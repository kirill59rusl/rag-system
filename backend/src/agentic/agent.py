import json
from typing import Any

from src.agentic.schemas import AgentResponse, Tool
from src.generator.llm import OpenAILLM


class OpenAIChat(OpenAILLM):

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None
    ) -> AgentResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

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
        msg = response.choices[0].message

        tool_calls = []
        raw_calls = []
        for tc in msg.tool_calls or []:
            if tc.type != "function":
                continue
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": tc.function.arguments}
            tool_calls.append(Tool(id=tc.id, name=tc.function.name, arguments=arguments))
            raw_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })

        message: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if raw_calls:
            message["tool_calls"] = raw_calls

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            message=message,
        )
