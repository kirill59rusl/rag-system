from abc import ABC, abstractmethod
from collections.abc import Iterator

import ollama


class LLM(ABC):

    @abstractmethod
    def generate(self, prompt:str) -> str:
        ...
    
    #@abstractmethod
    #def chat(self, messages) -> str:
    #    ...
    #
    #@abstractmethod
    #def stream(self, messages) -> Iterator(str):
    #    ...

class OllamaLLM(LLM):
    def __init__(self, model: str, host: str | None = None):
        self.model=model
        self.client = ollama.Client(host=host) if host else ollama

    def generate(self, prompt: str) -> str:
        response=self.client.generate(
            model=self.model,
            prompt=prompt
        )
        return response["response"]
