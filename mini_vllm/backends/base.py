from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Abstract base class for all inference backends. 

    A backend is anything that knows how to run a model. 
    Later we may have: 
        - Hugging Face CPU backend
        - CUDA Backend
        - ONNX backend
        - TensorRT backend

        The scheduler/runtime should not care which backend is used. 
    """
    @abstractmethod
    def generate_one(self, prompt: str, max_new_tokens: int = 30) -> str:
        pass
