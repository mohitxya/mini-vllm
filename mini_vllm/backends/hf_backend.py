import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mini_vllm.backends.base import Backend


class HFBackend(Backend):
    """
        Milestone 1: model.generate(), which hides the decoding loop. 
        Milestone 2: We manually implement autoregressive decoding.
        Milestone 3: manual greedy decoding with KV Cache. 
        Important: 
            We still use Hugging Face for: 
            - loading model weights. 
            - loading tokenizer.
            - running the transformer forward pass. 

            But we now control: 
            - the token-by-token loop. 
            - how next token is selected. 
            - when generation stops
    """
    def __init__(
            self, 
            model_name: str = "sshleifer/tiny-gpt2", 
            device: str | None = None):

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval() # disables dropout, BatchNorm behaves differently. 

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_one(self, prompt: str, max_new_tokens: int = 30) -> str:
       
        return self.generate_one_with_kv_cache(
            prompt=prompt, 
            max_new_tokens = max_new_tokens,
        )
    def generate_one_without_kv_cache(
        self, 
        prompt: str,
        max_new_tokens: int=30, 
    ) -> str: 
        """
            Manually generate text token by token using greedy decoding.

            Greedy decoding means: 
                At every step, choose the token with the highest probability. 
            This is deterministic. 
            same prompt -> same output
        """
        # step 1: prompt text to token IDs. 
        encoded = self.tokenizer(
            prompt, 
            return_tensors = "pt",
        )

        input_ids = encoded["input_ids"].to(self.device)

        # input_ids shape: [batch_size, sequence_length]
        # batch_size = 1 for single prompt. 

        # step 2: generate one token at a time.
        with torch.no_grad():
            for step in range(max_new_tokens):
                # transformer forward pass. 
                outputs = self.model(input_ids=input_ids)

                # for every position in the sequence, 
                # the model predicts scores for every token in vocabulary.
                logits = outputs.logits

                next_token_logits = logits[:,-1,:]

                # shape: [batch_size, vocab_size]
                # for us: [1,50257]

                next_token_id = torch.argmax(
                    next_token_logits, 
                    dim = 1,
                    keepdim = True, 
                )
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        generated_text = self.tokenizer.decode(
            input_ids[0], 
            skip_special_tokens=True,
        )

        return generated_text

    def generate_one_with_kv_cache(self, prompt: str, max_new_tokens: int=30,) -> str: 
        """
            Manual greedy decoding with KV Cache. 
            Key idea: 
                - First forward pass processes the full prompt. 
                - The model returns past_key_values. 
                - Later forward passes only process the newly generated token. 
                - Old attention keys/values are reused from cache. 

            This avoids recomputing the whole prefix every step. 
        """
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        # We keep generated token IDs separately so decoding is easy. 
        generated_ids = input_ids

        past_key_values = None
        
        with torch.no_grad(): 
            for step in range(max_new_tokens):

                if step == 0: 
                    # feed full prompt
                    model_input_ids = input_ids
                else: 
                    # feed only the most recently generated token. 
                    model_input_ids = next_token_id
                outputs = self.model(
                    input_ids = model_input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True,
                )
                logits = outputs.logits

                # store cache for next step. 
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:,-1,:]

                next_token_id = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids = torch.cat([generated_ids, next_token_id],
                                          dim=1,)
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def generate_one_with_kv_cache_debug(
        self, 
        prompt: str, 
        max_new_tokens: int=5,
    ) -> str: 
        """
            debug version that prints shapes of: input ids, logits, past key values. 
            to understand internals of KV Cache. 
        """
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        generated_ids = input_ids
        past_key_values = None

        print("\n=== Initial Prompt ===")
        print(prompt)

        print("\n=== Initial input_ids ===")
        print(input_ids)
        print("input_ids shape:", tuple(input_ids.shape))
        
        with torch.no_grad(): 
            for step in range(max_new_tokens):
                print(f"\n========= Step {step + 1} =========")

                if step == 0: 
                    model_input_ids = input_ids
                    print("Feeding FULL prompt")
                else: 
                    model_input_ids = next_token_id
                    print("Feeding ONLY last generated token")
                
                print("model_input_ids:", model_input_ids)
                print("model_input_ids shape:", tuple(model_input_ids.shape))

                outputs = self.model(
                    input_ids = model_input_ids, 
                    past_key_values = past_key_values, 
                    use_cache = True,
                )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                print("logits shape:", tuple(logits.shape))

                self._print_kv_cache_shapes(past_key_values)

                next_token_logits = logits[:, -1, :]

                next_token_id = torch.argmax(
                    next_token_logits, 
                    dim = -1, 
                    keepdim=True, 
                )

                token_text = self.tokenizer.decode(next_token_id[0])

                print("next_token_id:", next_token_id.item())
                print("next_token_text:", repr(token_text))

                if next_token_id.item() == self.tokenizer.eos_token_id: 
                    print("EOS token generated. Stopping.")
                    break

                generated_ids = torch.cat(
                    [generated_ids, next_token_id], 
                    dim = 1,
                )

                print("generated_ids shape:", tuple(generated_ids.shape))

            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _print_kv_cache_shapes(self, past_key_values) -> None: 
        """
            Print shape information for hugging face past_key_values

            for gpt 2 style models it's a tuple usually

            length: number of transformer layers. 
            each layer contains: key tensor, value tensor
        """    
        print("\nKV Cache:")
        print("number of layers cached:", len(past_key_values))

        first_layer = past_key_values[0]

        key_tensor = first_layer[0]
        value_tensor = first_layer[1]

        print("first layer key shape: ", tuple(key_tensor.shape))
        print("first layer value shape: ", tuple(value_tensor.shape))

    def compare_kv_cache_speed(
        self, 
        prompt: str, 
        max_new_tokens: int=50,
    ) -> None: 
        """
            compare generation time: 
                - without KV cache. 
                - with KV cache. 
            On CPU with small models, speedup may be modest. With larger models, KV cache 
            matters a lot more. 
        """
        print("\nBenchmarking without KV cache...")
        start = time.perf_counter()
        text_no_cache = self.generate_one_without_kv_cache(
            prompt = prompt, 
            max_new_tokens = max_new_tokens, 
        )
        no_cache_time = time.perf_counter() - start
        print("Done.")

        print("\nBenchmarking with KV cache...")
        start = time.perf_counter()
        text_cache = self.generate_one_with_kv_cache(
            prompt = prompt, 
            max_new_tokens = max_new_tokens,
        )
        cache_time = time.perf_counter() - start

        print("Done. ")
        print("\n=== Benchmark Results ===")
        print(f"Without KV cache: {no_cache_time:.4f} seconds")
        print(f"With KV cache: {cache_time:.4f} seconds")

        if cache_time > 0: 
            print(f"speedup: {no_cache_time / cache_time:.2f}x")

        print("\n=== Output without KV cache ===")
        print(text_no_cache)

        print("\n=== Output with KV cache ===")
        print(text_cache)
