from src.core.config import settings
from src.generator.llm import LLM, GigaChatLLM, OllamaLLM, OpenAILLM


def get_llm() -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(
            model=settings.llm,
        )

    if settings.llm_provider == "gigachat":
        return GigaChatLLM(
            credentials=settings.gigachat_credentials,
            model=settings.llm,
        )
    if settings.llm_provider == "openai":                
        return OpenAILLM(
            model=settings.llm,
            api_key=settings.openai_api_key,             
            base_url=getattr(settings, "openai_base_url", None),
        )

    raise ValueError(
        f"Неизвестный провайдер LLM: {settings.llm_provider}"
    )