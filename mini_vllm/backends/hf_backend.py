from __future__ import annotations
from mini_vllm.runtime.sampling import sample_next_token
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mini_vllm.backends.base import Backend
from mini_vllm.runtime.request import GenerationRequest


class HFBackend(Backend):
    """
    Hugging Face backend.

    This backend uses:
        - Hugging Face tokenizer
        - Hugging Face model weights
        - PyTorch execution
        - Hugging Face KV cache support

    But our runtime controls generation.

    Milestone 5 adds:
        - prefill(request)
        - decode_one(request)

    This is the core API needed for future scheduling.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str | None = None,
        stop_after_repeated_whitespace: int = 3,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_after_repeated_whitespace = stop_after_repeated_whitespace

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Backward-compatible full generation API
    # ------------------------------------------------------------------

    def generate_one(self, prompt: str, max_new_tokens: int = 30) -> str:
        """
        Generate one full response.

        This now uses the step API internally:

            request
            prefill(request)
            while not finished:
                decode_one(request)

        This proves that the step API can reproduce full generation.
        """

        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.prefill(request)

        while not request.is_finished():
            self.decode_one(request)

        if request.status.value == "FAILED":
            raise RuntimeError(request.error)

        return request.generated_text or ""

    # ------------------------------------------------------------------
    # Milestone 5: prefill/decode API
    # ------------------------------------------------------------------

    def prefill(self, request: GenerationRequest) -> None:
        """
        Process the full prompt and generate the first token.

        This is the "prefill" phase.

        Important:
            Prefill is usually more expensive than a single decode step
            because it processes the entire prompt.

        This method updates the request in-place.
        """

        if request.is_finished():
            return

        request.mark_running()

        try:
            encoded = self.tokenizer(
                request.prompt,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)

            request.input_ids = input_ids
            request.generated_ids = input_ids

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                )

                request.past_key_values = outputs.past_key_values

                logits = outputs.logits
                next_token_logits = logits[:, -1, :]

                next_token_id = self._select_next_token(
                    next_token_logits=next_token_logits,
                    request=request,
                )

            request.has_prefilled = True

            self._append_or_finish(
                request=request,
                next_token_id=next_token_id,
            )

        except Exception as exc:
            request.mark_failed(exc)

    def decode_one(self, request: GenerationRequest) -> None:
        """
        Generate exactly one token after prefill.

        This is the "decode" phase.

        Instead of passing the full sequence again, we pass only:

            request.last_token_id

        plus:

            request.past_key_values

        This is what makes KV-cache decoding efficient.
        """

        if request.is_finished():
            return

        if not request.has_prefilled:
            raise RuntimeError(
                "Cannot decode before prefill. "
                "Call backend.prefill(request) first."
            )

        if request.last_token_id is None:
            # This can happen if prefill generated EOS immediately.
            return

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=request.last_token_id,
                    past_key_values=request.past_key_values,
                    use_cache=True,
                )

                request.past_key_values = outputs.past_key_values

                logits = outputs.logits
                next_token_logits = logits[:, -1, :]

                next_token_id = self._select_next_token(
                    next_token_logits=next_token_logits,
                    request=request,
                )

            self._append_or_finish(
                request=request,
                next_token_id=next_token_id,
            )

        except Exception as exc:
            request.mark_failed(exc)
        # ------------------------------------------------------------------
    # Milestone 7: simplified batched recompute API
    # ------------------------------------------------------------------

    def prepare_for_batch_recompute(self, request: GenerationRequest) -> None:
        """
        Prepare one request for simplified continuous batching.

        This batching path does NOT use KV cache.

        Instead, every request stores its current full token sequence:

            request.generated_ids

        Each batch step pads all active requests into one tensor and runs
        the model on the full current sequence.

        This is less efficient than KV-cache decoding, but it teaches:
            - batch construction
            - variable-length padding
            - attention masks
            - dynamic active request sets
        """

        if request.is_finished():
            return

        request.mark_running()

        try:
            encoded = self.tokenizer(
                request.prompt,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)

            request.input_ids = input_ids
            request.generated_ids = input_ids
            request.last_token_id = None
            request.past_key_values = None
            request.has_prefilled = True
            request.num_generated_tokens = 0

            if request.max_new_tokens <= 0:
                self._finish_request(request)

        except Exception as exc:
            request.mark_failed(exc)

    def decode_batch_recompute(self, requests: list[GenerationRequest]) -> None:
        """
        Decode one token for multiple active requests using one batched
        forward pass.

        Important:
            This is true batching, but not KV-cache batching.

        For each request, we already have:

            request.generated_ids

        These sequences may have different lengths.

        Example:

            A: [10, 20, 30]
            B: [50, 60]
            C: [90, 91, 92, 93]

        We pad them:

            A: [10, 20, 30, PAD]
            B: [50, 60, PAD, PAD]
            C: [90, 91, 92, 93]

        And create attention mask:

            A: [1, 1, 1, 0]
            B: [1, 1, 0, 0]
            C: [1, 1, 1, 1]

        Then one model forward gives logits for the whole batch.

        For each request, we take logits from its last real token position.
        """

        active_requests = [
            request for request in requests
            if not request.is_finished()
        ]

        if not active_requests:
            return

        # Make sure every request has tokenized state.
        for request in active_requests:
            if request.generated_ids is None:
                self.prepare_for_batch_recompute(request)

        # Some requests may have failed during preparation.
        active_requests = [
            request for request in active_requests
            if not request.is_finished()
        ]

        if not active_requests:
            return

        pad_token_id = self.tokenizer.pad_token_id

        # Each request.generated_ids has shape [1, seq_len].
        sequence_lengths = [
            request.generated_ids.shape[1]
            for request in active_requests
        ]

        max_seq_len = max(sequence_lengths)
        batch_size = len(active_requests)

        batch_input_ids = torch.full(
            size=(batch_size, max_seq_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.zeros(
            size=(batch_size, max_seq_len),
            dtype=torch.long,
            device=self.device,
        )

        # Copy each request's current sequence into the batch tensor.
        for row_idx, request in enumerate(active_requests):
            seq = request.generated_ids[0]
            seq_len = seq.shape[0]

            batch_input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                logits = outputs.logits

                # logits shape:
                # [batch_size, max_seq_len, vocab_size]
                #
                # For each row, the next-token prediction is at
                # that request's last real token position.
                for row_idx, request in enumerate(active_requests):
                    seq_len = sequence_lengths[row_idx]
                    last_real_position = seq_len - 1

                    next_token_logits = logits[
                        row_idx: row_idx + 1,
                        last_real_position,
                        :,
                    ]

                    next_token_id = self._select_next_token(
                        next_token_logits=next_token_logits,
                        request=request,
                    )

                    self._append_or_finish(
                        request=request,
                        next_token_id=next_token_id,
                    )

        except Exception as exc:
            for request in active_requests:
                request.mark_failed(exc)
    # ------------------------------------------------------------------
    # Token selection
    # ------------------------------------------------------------------
    
    def _select_next_token_greedy(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding.

        next_token_logits shape:
            [batch_size, vocab_size]

        Returns:
            next_token_id shape [batch_size, 1]

        Greedy means:
            choose token with highest logit.
        """

        return torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )
    def _select_next_token( self, next_token_logits: torch.Tensor, request: GenerationRequest,
    ) -> torch.Tensor:
        """
        Select next token using the request's sampling configuration.

        next_token_logits shape:
            [batch_size, vocab_size]

        Returns:
            next_token_id shape [batch_size, 1]
        """

        return sample_next_token(
            logits=next_token_logits,
            config=request.sampling_config,
        )
    # ------------------------------------------------------------------
    # Request update helpers
    # ------------------------------------------------------------------

    def _append_or_finish(
        self,
        request: GenerationRequest,
        next_token_id: torch.Tensor,
    ) -> None:
        """
        Decide whether to append the token or finish the request.

        Stop conditions:
            1. EOS token generated
            2. max_new_tokens reached
            3. too many repeated whitespace tokens
        """

        if next_token_id.item() == self.tokenizer.eos_token_id:
            self._finish_request(request)
            return

        token_text = self.tokenizer.decode(next_token_id[0])

        # Track whitespace repetition to avoid ugly blank-line loops.
        if not hasattr(request, "consecutive_whitespace_tokens"):
            request.consecutive_whitespace_tokens = 0

        if token_text.strip() == "":
            request.consecutive_whitespace_tokens += 1
        else:
            request.consecutive_whitespace_tokens = 0

        if request.consecutive_whitespace_tokens >= self.stop_after_repeated_whitespace:
            self._finish_request(request)
            return

        request.generated_ids = torch.cat(
            [request.generated_ids, next_token_id],
            dim=1,
        )

        request.last_token_id = next_token_id
        request.num_generated_tokens += 1

        if request.num_generated_tokens >= request.max_new_tokens:
            self._finish_request(request)

    def _finish_request(self, request: GenerationRequest) -> None:
        """
        Decode generated token IDs and mark request as finished.
        """

        if request.generated_ids is None:
            request.mark_finished("")
            return

        generated_text = self.tokenizer.decode(
            request.generated_ids[0],
            skip_special_tokens=True,
        )

        request.mark_finished(generated_text)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def print_request_state(self, request: GenerationRequest) -> None:
        """
        Print useful debugging information about a request.
        """

        print(f"\nRequest {request.short_id()}")
        print(f"status: {request.status}")
        print(f"has_prefilled: {request.has_prefilled}")
        print(f"num_generated_tokens: {request.num_generated_tokens}")

        if request.input_ids is not None:
            print(f"input_ids shape: {tuple(request.input_ids.shape)}")

        if request.generated_ids is not None:
            print(f"generated_ids shape: {tuple(request.generated_ids.shape)}")

        if request.last_token_id is not None:
            print(f"last_token_id: {request.last_token_id.item()}")
            print(
                "last_token_text:",
                repr(self.tokenizer.decode(request.last_token_id[0])),
            )

        if request.past_key_values is not None:
            self._print_kv_cache_shapes(request.past_key_values)

    def _print_kv_cache_shapes(self, past_key_values) -> None:
        """
        Print shape information for Hugging Face KV cache.

        Transformers versions differ:

        Older versions:
            past_key_values is a tuple/list:
                past_key_values[layer_idx] = (key_tensor, value_tensor)

        Newer versions:
            past_key_values is often a DynamicCache object.

        This debug function should never crash the runtime.
        """

        print("\nKV Cache:")
        print("cache object type:", type(past_key_values))

        # ------------------------------------------------------------
        # Newer Hugging Face DynamicCache-style object
        # ------------------------------------------------------------
        if past_key_values.__class__.__name__ == "DynamicCache":
            print("cache format: DynamicCache")

            if hasattr(past_key_values, "get_seq_length"):
                try:
                    print("cached sequence length:", past_key_values.get_seq_length())
                except Exception as exc:
                    print("could not read sequence length:", exc)

            # Some versions expose key_cache/value_cache.
            if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
                key_cache = past_key_values.key_cache
                value_cache = past_key_values.value_cache

                print("number of layers cached:", len(key_cache))

                if len(key_cache) > 0:
                    print("first layer key shape:  ", tuple(key_cache[0].shape))
                    print("first layer value shape:", tuple(value_cache[0].shape))

                return

            # Other versions hide the internal tensors differently.
            # So we print available public methods/fields for debugging.
            public_attrs = [
                name for name in dir(past_key_values)
                if not name.startswith("_")
            ]

            print("DynamicCache does not expose key_cache/value_cache directly.")
            print("Available public attributes/methods:")
            print(public_attrs)

            return

        # ------------------------------------------------------------
        # Legacy tuple/list cache
        # ------------------------------------------------------------
        if isinstance(past_key_values, (tuple, list)):
            print("cache format: legacy tuple/list")
            print("number of layers cached:", len(past_key_values))

            first_layer = past_key_values[0]
            key_tensor = first_layer[0]
            value_tensor = first_layer[1]

            print("first layer key shape:  ", tuple(key_tensor.shape))
            print("first layer value shape:", tuple(value_tensor.shape))
            return

        # ------------------------------------------------------------
        # Unknown cache type
        # ------------------------------------------------------------
        print("Unknown cache format.")
        print("Available public attributes/methods:")
        public_attrs = [
            name for name in dir(past_key_values)
            if not name.startswith("_")
        ]
        print(public_attrs)

    # ------------------------------------------------------------------
    # Benchmark helper retained from Milestone 3
    # ------------------------------------------------------------------

    def compare_kv_cache_speed(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> None:
        """
        Compare full-prefix recomputation vs step-wise KV-cache generation.

        This is kept for educational benchmarking.
        """

        print("\nBenchmarking old-style full generation through step API...")
        start = time.perf_counter()
        text = self.generate_one(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - start

        print(f"Elapsed: {elapsed:.4f}s")
        print(text)