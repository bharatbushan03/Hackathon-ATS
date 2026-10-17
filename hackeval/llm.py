from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import AGENT_IDS, Settings

LlmMap = dict[str, BaseChatModel]


def azure_v1_base(url: str) -> str:
    """Turn a resource root or Foundry project URL into the OpenAI v1 base URL.

    Foundry project endpoints keep ``/api/projects/<name>`` and append ``/openai/v1/``.
    Classic Azure OpenAI resource roots just get ``/openai/v1/``.
    """
    u = url.strip().rstrip("/")
    if "/openai/v1" in u:
        return u.split("/openai/v1")[0] + "/openai/v1/"
    return u + "/openai/v1/"


def _as_llms(llm: BaseChatModel | LlmMap) -> LlmMap:
    if isinstance(llm, dict):
        return llm
    return {aid: llm for aid in AGENT_IDS}


def build_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    """Build a chat model. Azure uses one key + one project/resource URL for every deployment."""
    p = settings.llm_provider
    name = model or settings.llm_model
    kw = dict(model=name, temperature=settings.llm_temperature)
    if p == "azure":
        from langchain_openai import ChatOpenAI

        if not settings.azure_endpoint or not settings.azure_api_key:
            raise ValueError(
                "Set AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_PROJECT_URL) and AZURE_OPENAI_API_KEY."
            )
        return ChatOpenAI(
            **kw,
            timeout=settings.llm_timeout,
            max_retries=0,
            api_key=settings.azure_api_key,
            base_url=azure_v1_base(settings.azure_endpoint),
            default_headers={"api-key": settings.azure_api_key},
        )
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**kw, timeout=settings.llm_timeout, max_retries=0)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kw, timeout=settings.llm_timeout, max_retries=0)
    if p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**kw, timeout=settings.llm_timeout, max_retries=0)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(**kw)
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{p}'. Use azure, openai, anthropic, google, or ollama."
    )


def build_llms(settings: Settings) -> LlmMap:
    """One client per agent. Same Azure credentials; model name may differ per agent."""
    cache: dict[str, BaseChatModel] = {}
    out: LlmMap = {}
    for aid in AGENT_IDS:
        name = settings.agent_models[aid]
        if name not in cache:
            cache[name] = build_llm(settings, name)
        out[aid] = cache[name]
    return out
