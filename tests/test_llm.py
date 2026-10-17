from hackeval.llm import azure_v1_base, build_llm, build_llms
from hackeval.config import AGENT_IDS, Settings


def _settings(**kw) -> Settings:
    model = kw.get("llm_model", "gpt-4o-mini")
    base = dict(
        llm_provider="azure",
        llm_model=model,
        agent_models={aid: model for aid in AGENT_IDS},
        azure_endpoint="https://demo.openai.azure.com",
        azure_api_key="sk-test",
        llm_temperature=0.1,
        llm_timeout=30,
        max_retries=0,
        retry_backoff=0,
        max_doc_chars=1000,
        max_zip_mb=1,
        max_uncompressed_mb=1,
        max_files=1,
        data_dir=__import__("pathlib").Path("."),
        csv_path=__import__("pathlib").Path("evaluation.csv"),
        api_url="http://127.0.0.1:8000",
    )
    base.update(kw)
    return Settings(**base)


def test_azure_v1_base():
    assert azure_v1_base("https://x.openai.azure.com") == "https://x.openai.azure.com/openai/v1/"
    assert azure_v1_base("https://x.openai.azure.com/openai/v1") == "https://x.openai.azure.com/openai/v1/"
    assert (
        azure_v1_base("https://x.services.ai.azure.com/api/projects/hack")
        == "https://x.services.ai.azure.com/api/projects/hack/openai/v1/"
    )


def test_azure_build_llm_uses_v1_base():
    llm = build_llm(_settings())
    assert "openai/v1" in str(llm.openai_api_base or llm.base_url)
    project = build_llm(_settings(azure_endpoint="https://x.services.ai.azure.com/api/projects/hack"))
    assert "/api/projects/hack/openai/v1" in str(project.openai_api_base or project.base_url)


def test_azure_build_llms_shares_client_for_same_model():
    llms = build_llms(_settings())
    assert set(llms) == set(AGENT_IDS)
    assert len({id(v) for v in llms.values()}) == 1


def test_azure_build_llms_uses_per_agent_models():
    s = _settings()
    s = Settings(**{**s.__dict__, "agent_models": {
        "problem_innovation": "gpt-4o",
        "technical": "gpt-4.1",
        "impact_presentation": "gpt-4o-mini",
        "final": "gpt-4.1",
    }})
    llms = build_llms(s)
    assert llms["problem_innovation"].model_name == "gpt-4o"
    assert llms["technical"] is llms["final"]
    assert llms["impact_presentation"].model_name == "gpt-4o-mini"


def test_azure_requires_key_and_url():
    try:
        build_llm(_settings(azure_api_key="", azure_endpoint=""))
    except ValueError as e:
        assert "AZURE_OPENAI" in str(e)
    else:
        raise AssertionError("expected ValueError")
