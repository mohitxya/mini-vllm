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
