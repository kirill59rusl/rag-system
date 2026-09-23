from abc import ABC, abstractmethod
from typing import Any

import ollama
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from openai import AsyncOpenAI


class LLM(ABC):

    @abstractmethod
    async def generate(self, prompt:str, response_format: dict[str, Any] | None = None) -> str:
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
        self.client = ollama.AsyncClient(host=host)

    async def generate(self, prompt: str, response_format: dict[str, Any] | None = None) -> str:
        response=await self.client.generate(
            model=self.model,
            prompt=prompt,
            format=response_format,
        )
        return response["thinking"]

class GigaChatLLM(LLM):
    def __init__(
        self,
        credentials: str,
        model: str
    ) -> None:
        self.model = model

        self.client = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
        )

    async def generate(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        chat = Chat(
            model=self.model,
            messages=[
                Messages(
                    role=MessagesRole.USER,
                    content=prompt,
                )
            ],
        )

        if response_format is not None:
            chat.response_format = {
                "type": "json_schema",
                "schema": response_format,
                "strict": True,
            }
        response = await self.client.achat(chat)

        return response.choices[0].message.content

class OpenAILLM(LLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url
        )
        self.model = model

    async def generate(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_format,
                    "strict": True,
                },
            }

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""