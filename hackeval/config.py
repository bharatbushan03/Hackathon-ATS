from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "azure": "gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "google": "gemini-2.5-flash",
    "ollama": "llama3.1",
}

AGENT_IDS = ("problem_innovation", "technical", "impact_presentation", "final")
AGENT_MODEL_ENV = {
    "problem_innovation": "LLM_MODEL_PROBLEM_INNOVATION",
    "technical": "LLM_MODEL_TECHNICAL",
    "impact_presentation": "LLM_MODEL_IMPACT_PRESENTATION",
    "final": "LLM_MODEL_FINAL",
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    agent_models: dict[str, str]
    azure_endpoint: str
    azure_api_key: str
    llm_temperature: float
    llm_timeout: int
    max_retries: int
    retry_backoff: float
    max_doc_chars: int
    max_zip_mb: int
    max_uncompressed_mb: int
    max_files: int
    data_dir: Path
    csv_path: Path
    api_url: str
    google_drive_api_key: str = ""


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "azure").strip().lower()
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    default_model = (os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")).strip()
    agent_models = {
        aid: (os.getenv(env) or default_model).strip() for aid, env in AGENT_MODEL_ENV.items()
    }
    return Settings(
        llm_provider=provider,
        llm_model=default_model,
        agent_models=agent_models,
        azure_endpoint=(
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_PROJECT_URL")
            or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
            or ""
        ).strip(),
        azure_api_key=(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
        retry_backoff=float(os.getenv("RETRY_BACKOFF_SECONDS", "2")),
        max_doc_chars=int(os.getenv("MAX_DOC_CHARS", "40000")),
        max_zip_mb=int(os.getenv("MAX_ZIP_MB", "200")),
        max_uncompressed_mb=int(os.getenv("MAX_UNCOMPRESSED_MB", "500")),
        max_files=int(os.getenv("MAX_FILES", "100")),
        data_dir=data_dir,
        csv_path=Path(os.getenv("CSV_PATH", str(data_dir / "evaluation.csv"))).resolve(),
        api_url=os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/"),
        google_drive_api_key=os.getenv("GOOGLE_DRIVE_API_KEY", "").strip(),
    )
