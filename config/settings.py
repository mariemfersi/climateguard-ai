"""
Central configuration loader for ClimateGuard AI.

Design principle (per Roadmap Task 0.1.3):
    Settings needed by the CURRENT phase are required (raise on missing).
    Settings needed by LATER phases are optional at this stage of the project
    and default to None, so the app runs cleanly before Phase 3/9 begin.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.fred_api_key)
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Phase 1: external data sources (required once Phase 1 starts) ---
    cds_api_url: str = Field(default="https://cds.climate.copernicus.eu/api")
    cds_api_key: str | None = None
    nasa_firms_map_key: str | None = None
    fred_api_key: str | None = None

    # --- Phase 3: Azure core infra (optional until Phase 3) ---
    azure_subscription_id: str | None = None
    azure_tenant_id: str | None = None
    azure_resource_group: str = "climateguard-ai-rg"
    azure_storage_account_name: str | None = None
    azure_storage_account_key: str | None = None
    azure_key_vault_name: str | None = None
    azure_databricks_workspace_url: str | None = None
    azure_ml_workspace_name: str | None = None
    azure_ml_mlflow_tracking_uri: str | None = None

    # --- Phase 9: Azure OpenAI / AI Search (optional until Phase 9) ---
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment_chat: str | None = None
    azure_openai_deployment_embedding: str | None = None
    azure_ai_search_endpoint: str | None = None
    azure_ai_search_api_key: str | None = None
    azure_ai_search_index_name: str = "climateguard-rag-index"

    # --- Phase 10: serving ---
    api_env: str = "local"
    api_log_level: str = "INFO"

    def require(self, field_name: str) -> str:
        """
        Fetch a setting that is required for the CURRENT operation, raising a
        clear, actionable error if it is missing rather than failing later
        with a confusing None-related exception deep in a data pipeline.
        """
        value = getattr(self, field_name, None)
        if not value:
            raise RuntimeError(
                f"Missing required setting '{field_name}'. "
                f"Set it in your .env file (see .env.example for the full list)."
            )
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing .env on every import."""
    return Settings()
