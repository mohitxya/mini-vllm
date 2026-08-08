"""
    - vLLM config configures the Engine. 
    - InprocClient, DPLBAsyncMPClient (allows serving at scale)
    As per the blog anatomy of vllm: 
    Engine is made up of several sub components: 
        - Model executor
        - Structured output manager
        - Scheduler: policy, waiting and running queues, KV cache manager. 
    The KV cache manager maintains a `free_block_queue` (pool of available KV-cache blocks).
"""
import torch

class LLMEngine(model: str = None, speculative_config = None): 
    def __init__(self): 
        pass
    def generate(self, prompts, sampling_params): 
        pass
