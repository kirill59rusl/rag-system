from abc import ABC, abstractmethod
from typing import Any

import ollama


class LLM(ABC):

    @abstractmethod
    def generate(self, prompt:str, format: dict[str, Any] | None = None) -> str:
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

    def generate(self, prompt: str, format: dict[str, Any] | None = None) -> str:
        response=self.client.generate(
            model=self.model,
            prompt=prompt,
            format=format,
            options={
                "think": True
            }
        )
        return response["thinking"]
