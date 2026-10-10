from src.agentic.agent import OpenAIChat
from src.agentic.prompt import SYSTEM_PROMPT
from src.agentic.schemas import AgentAnswer
from src.agentic.tools import SEARCH_DOCS, ToolContext, run_tool
from src.generator.generate import build_sources, clean_claims
from src.schemas.llmresponse import LLMResponse

MAX_STEPS = 5

async def run_agent(query: str, ctx: ToolContext, chat: OpenAIChat,
                    max_steps: int = MAX_STEPS, debug=False):
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    response_format=LLMResponse.model_json_schema()

    for step in range(max_steps):
        response=await chat.chat(messages, [SEARCH_DOCS], response_format)
        messages.append(response.message)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": await run_tool(ctx,call)
                }
            )

    else:
        response=await chat.chat(messages, response_format=response_format)

    res=LLMResponse.model_validate_json(response.content)
    claims=clean_claims(res.claims, len(ctx.sources))
    used_ids={s for c in claims for s in c.sources}


    return AgentAnswer(
        answer=res.answer,
        claims=claims,
        sources=build_sources(ctx.sources, used_ids, with_content=debug),
        messages=messages[1:] if debug else None,
    )