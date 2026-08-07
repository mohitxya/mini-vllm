"""Load Qwen/Qwen3-0.6B Hugging Face weights into mini-vLLM.

Run from the repository root:

    .venv/bin/python example.py

The first run downloads the model's config and safetensors shards to the
Hugging Face cache. This script loads tensors one shard at a time, so it does
not create a second full Qwen model in RAM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parent


def require_huggingface_dependencies():
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        from safetensors import safe_open
        from transformers import AutoConfig
    except ImportError as exc:
        raise SystemExit(
            "Install the Hugging Face loader dependencies first:\n"
            "  .venv/bin/pip install transformers safetensors huggingface_hub\n"
            f"Missing module: {exc.name}"
        ) from exc
    return AutoConfig, hf_hub_download, safe_open, EntryNotFoundError


def init_single_process_group() -> None:
    """The current parallel layers require a one-rank process group."""
    if dist.is_initialized():
        return
    rendezvous = tempfile.NamedTemporaryFile(prefix="minivllm-dist-", delete=False)
    rendezvous.close()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous.name}",
        rank=0,
        world_size=1,
    )
    os.unlink(rendezvous.name)


def import_qwen_model():
    """Import the architecture whose filename is not a Python identifier."""
    path = ROOT / "minivllm" / "models" / "qwen3-0.6b.py"
    spec = importlib.util.spec_from_file_location("minivllm_qwen3_06b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model architecture from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3ForCausalLM


def hf_weight_names(num_layers: int) -> dict[str, tuple[str, int | str | None]]:
    """Map HF tensor names to mini-vLLM parameter names and packed shard IDs."""
    mapping = {
        "model.embed_tokens.weight": ("model.embed_tokens.weight", None),
        "model.norm.weight": ("model.norm.weight", None),
        "lm_head.weight": ("lm_head.weight", None),
    }
    for layer in range(num_layers):
        source = f"model.layers.{layer}"
        target = source
        mapping.update({
            f"{source}.input_layernorm.weight": (f"{target}.input_layernorm.weight", None),
            f"{source}.post_attention_layernorm.weight": (f"{target}.post_attention_layernorm.weight", None),
            f"{source}.self_attn.q_norm.weight": (f"{target}.self_attn.q_norm.weight", None),
            f"{source}.self_attn.k_norm.weight": (f"{target}.self_attn.k_norm.weight", None),
            f"{source}.self_attn.q_proj.weight": (f"{target}.self_attn.qkv_proj.weight", "q"),
            f"{source}.self_attn.k_proj.weight": (f"{target}.self_attn.qkv_proj.weight", "k"),
            f"{source}.self_attn.v_proj.weight": (f"{target}.self_attn.qkv_proj.weight", "v"),
            f"{source}.self_attn.o_proj.weight": (f"{target}.self_attn.o_proj.weight", None),
            f"{source}.mlp.gate_proj.weight": (f"{target}.mlp.gate_up_proj.weight", 0),
            f"{source}.mlp.up_proj.weight": (f"{target}.mlp.gate_up_proj.weight", 1),
            f"{source}.mlp.down_proj.weight": (f"{target}.mlp.down_proj.weight", None),
        })
    return mapping


def load_weights(model, download_file, safe_open, EntryNotFoundError) -> None:
    """Stream HF safetensors into the target parameters via their loaders."""
    try:
        index_path = download_file("model.safetensors.index.json")
    except (EntryNotFoundError, FileNotFoundError):
        # Qwen/Qwen3-0.6B is a single safetensors checkpoint. Keep support for
        # indexed checkpoints as well, since the mapping code is otherwise the
        # same for larger models.
        checkpoint = download_file("model.safetensors")
        with safe_open(checkpoint, framework="pt", device="cpu") as shard:
            weight_map = {name: "model.safetensors" for name in shard.keys()}
    else:
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]

    mapping = hf_weight_names(len(model.model.layers))
    required = set(mapping)
    missing = sorted(required - set(weight_map))

    # Qwen may tie lm_head and embedding weights, in which case safetensors
    # stores just the embedding parameter. The architecture shares that tensor.
    if missing == ["lm_head.weight"]:
        mapping.pop("lm_head.weight")
        missing = []
    if missing:
        raise KeyError(f"Checkpoint does not contain expected tensors: {missing}")

    params = dict(model.named_parameters())
    shard_to_sources: dict[str, list[str]] = {}
    for source_name in mapping:
        shard_to_sources.setdefault(weight_map[source_name], []).append(source_name)

    loaded = []
    with torch.no_grad():
        for shard_name, source_names in shard_to_sources.items():
            shard_path = download_file(shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for source_name in source_names:
                    target_name, shard_id = mapping[source_name]
                    param = params[target_name]
                    tensor = shard.get_tensor(source_name)
                    module_name, _, _ = target_name.rpartition(".")
                    module = model.get_submodule(module_name)
                    loader = getattr(module, "weight_loader", None)
                    if loader is None:
                        if shard_id is not None:
                            raise RuntimeError(
                                f"Packed parameter {target_name} needs a weight loader"
                            )
                        if param.shape != tensor.shape:
                            raise ValueError(
                                f"Shape mismatch for {target_name}: expected "
                                f"{tuple(param.shape)}, got {tuple(tensor.shape)}"
                            )
                        param.copy_(tensor)
                    elif shard_id is None:
                        loader(param, tensor)
                    else:
                        loader(param, tensor, shard_id)
                    loaded.append(source_name)

    if len(loaded) != len(mapping):
        raise RuntimeError(f"Loaded {len(loaded)} tensors; expected {len(mapping)}")


def allocate_kv_cache(model, max_seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Allocate the flat paged KV layout consumed by the attention layer."""
    from minivllm.layers.attention import Attention

    num_slots = math.ceil(max_seq_len / block_size) * block_size
    cache_dtype = model.model.embed_tokens.weight.dtype
    for module in model.modules():
        if isinstance(module, Attention):
            shape = (num_slots, module.num_kv_heads, module.head_dim)
            module.k_cache = torch.empty(shape, dtype=cache_dtype, device=device)
            module.v_cache = torch.empty_like(module.k_cache)
    return torch.arange(num_slots // block_size, dtype=torch.long, device=device).unsqueeze(0)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    """Minimal batch-one prefill/decode loop for validating mini-vLLM."""
    from minivllm.utils.context import Context, set_context

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Transformers 5 returns a BatchEncoding here, while older versions
    # returned the tensor directly.
    input_ids = (encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"])
    input_ids = input_ids.to(device).squeeze(0)
    max_seq_len = input_ids.numel() + max_new_tokens
    block_size = 16
    block_tables = allocate_kv_cache(model, max_seq_len, block_size, device)

    prompt_len = input_ids.numel()
    set_context(Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, prompt_len], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, prompt_len], dtype=torch.int32, device=device),
        max_seqlen_q=prompt_len,
        max_seqlen_k=prompt_len,
        slot_mapping=torch.arange(prompt_len, dtype=torch.long, device=device),
        block_size=block_size,
    ))
    hidden_states = model(input_ids, torch.arange(prompt_len, device=device))
    logits = model.compute_logits(hidden_states)[0]

    generated: list[int] = []
    for position in range(prompt_len, max_seq_len):
        if temperature == 0:
            next_token = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(0)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

        set_context(Context(
            is_prefill=False,
            slot_mapping=torch.tensor([position], dtype=torch.long, device=device),
            block_tables=block_tables,
            context_lens=torch.tensor([position + 1], dtype=torch.int32, device=device),
            block_size=block_size,
        ))
        hidden_states = model(next_token.reshape(1), torch.tensor([position], device=device))
        logits = model.compute_logits(hidden_states)[0]

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu", help="Target device after loading, e.g. cuda:0")
    parser.add_argument("--prompt", default="The capital of India is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")

    AutoConfig, hf_hub_download, safe_open, EntryNotFoundError = require_huggingface_dependencies()
    from transformers import AutoTokenizer
    init_single_process_group()
    model_source = Path(args.model).expanduser()
    if model_source.is_absolute() and not model_source.is_dir():
        raise SystemExit(
            f"Local model directory does not exist: {model_source}\n"
            "Use a real snapshot directory, or use the Hugging Face model ID:\n"
            f"  {Path(__file__).name} --model {MODEL_ID}"
        )
    is_local_model = model_source.is_dir()
    model_id_or_path = str(model_source) if is_local_model else args.model
    config = AutoConfig.from_pretrained(model_id_or_path)
    Qwen3ForCausalLM = import_qwen_model()

    dtype = getattr(torch, args.dtype)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.to_empty(device="cpu")
    # ``to_empty`` materializes parameters but intentionally leaves buffers
    # uninitialized. RoPE's non-persistent cosine/sine cache must be rebuilt.
    from minivllm.layers.rotary_embedding import RotaryEmbedding
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.rebuild_cache()
    model = model.to(dtype=dtype)

    def download_file(filename: str) -> str:
        if is_local_model:
            local_file = model_source / filename
            if not local_file.is_file():
                raise FileNotFoundError(f"Missing checkpoint file: {local_file}")
            return str(local_file)
        return hf_hub_download(args.model, filename)

    load_weights(model, download_file, safe_open, EntryNotFoundError)
    if config.tie_word_embeddings:
        # HF safetensors omits the duplicate lm_head tensor. Re-establish this
        # after to_empty(), which can otherwise break shared storage.
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
    model.to(args.device).eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {args.model} into mini-vLLM ({total_parameters:,} parameters, {args.dtype}, {args.device}).")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    completion = generate(
        model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
        torch.device(args.device),
    )
    print(f"Prompt: {args.prompt!r}\nCompletion: {completion!r}")


if __name__ == "__main__":
    main()
