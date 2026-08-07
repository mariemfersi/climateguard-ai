"""
Shared MLflow tracking configuration. Import and call configure_mlflow()
at the top of any training entrypoint before mlflow.start_run().
"""
import logging
import os

import mlflow

from config.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MLFLOW_DB_URI = "sqlite:///D:/dev/climateguard-ai/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "climateguard-ai"


def _load_azureml_mlflow_plugin() -> bool:
    """Attempt to import the Azure ML MLflow plugin."""
    try:
        from azureml import mlflow as azureml_mlflow  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import azureml.mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def configure_mlflow() -> None:
    settings = get_settings()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.azure_ml_mlflow_tracking_uri

    if tracking_uri:
        if tracking_uri.startswith("azureml://"):
            if _load_azureml_mlflow_plugin():
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    logger.info(f"MLflow Azure ML tracking → {tracking_uri}")
                except Exception as e:
                    logger.warning(
                        "Failed to set Azure ML tracking URI %s: %s. Falling back to local SQLite.",
                        tracking_uri,
                        e,
                    )
                    tracking_uri = None
            else:
                logger.warning(
                    "Azure ML MLflow integration is unavailable. Falling back to local SQLite tracking."
                )
                tracking_uri = None
        else:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking → {tracking_uri}")
            except Exception as e:
                logger.warning(
                    "Failed to set MLFLOW_TRACKING_URI %s: %s. Falling back to local SQLite.",
                    tracking_uri,
                    e,
                )
                tracking_uri = None

    if not tracking_uri:
        if os.environ.get("MLFLOW_TRACKING_URI"):
            logger.warning(
                "Ignoring invalid MLFLOW_TRACKING_URI and using local SQLite fallback."
            )
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_DB_URI)
        logger.info("Using local SQLite MLflow tracking")

    try:
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
        logger.info("MLflow experiment set to %s", DEFAULT_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("Failed to set MLflow experiment: %s", e)
