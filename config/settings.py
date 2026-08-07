"""
Central configuration loader for ClimateGuard AI.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Phase 1: external data sources ---
    cds_api_url: str = Field(default="https://cds.climate.copernicus.eu/api")
    cds_api_key: Optional[str] = None
    nasa_firms_map_key: Optional[str] = None
    fred_api_key: Optional[str] = None

    # --- Phase 3: Azure core infra ---
    azure_subscription_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_resource_group: str = "climateguard-ai-rg"
    azure_storage_account_name: Optional[str] = None
    azure_storage_account_key: Optional[str] = None
    azure_key_vault_name: Optional[str] = None
    azure_databricks_workspace_url: Optional[str] = None
    azure_ml_workspace_name: Optional[str] = None
    azure_ml_mlflow_tracking_uri: Optional[str] = None

    # --- Phase 9: Azure OpenAI / AI Search ---
    azure_openai_endpoint: str = Field(...)
    azure_openai_api_key: str = Field(...)
    azure_openai_api_version: str = Field(default="2024-08-01-preview")
    azure_openai_chat_deployment: str = Field(default="gpt-4o-mini")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-small")

    # Optional public OpenAI fallback
    openai_api_key: Optional[str] = None
    llm_provider: Literal["azure", "openai"] = "azure"

    # RAG (local Chroma)
    rag_persist_directory: str = Field(default="rag_index/chroma_db")
    rag_collection_name: str = Field(default="climateguard_docs")
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_top_k: int = 6

    # Optional Azure AI Search (for later)
    azure_ai_search_endpoint: Optional[str] = None
    azure_ai_search_api_key: Optional[str] = None
    azure_ai_search_index_name: str = "climateguard-rag-index"

    # Model-serving API (Phase 8/10)
    model_api_base_url: str = Field(default="http://localhost:8000/v1")
    model_api_timeout: float = 30.0

    # Phase 10
    api_env: str = "local"
    api_log_level: str = "INFO"

    def require(self, field_name: str) -> str:
        value = getattr(self, field_name, None)
        if not value:
            raise RuntimeError(
                f"Missing required setting '{field_name}'. "
                f"Set it in your .env file (see .env.example)."
            )
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()