"""
Shared MLflow tracking configuration. Import and call configure_mlflow()
at the top of any training entrypoint before mlflow.start_run().
"""
import logging

import mlflow

# Import azureml-mlflow to register the azureml:// scheme plugin
try:
    from azureml import mlflow as azureml_mlflow
except ImportError:
    try:
        import azureml.mlflow
    except ImportError:
        pass  # azureml-mlflow not installed, will fall back to local

from config.settings import get_settings

logger = logging.getLogger(__name__)


def configure_mlflow() -> None:
    settings = get_settings()
    
    # Azure ML MLflow tracking doesn't support full model registry API (404 on logged-models endpoint)
    # Root cause: azureml-mlflow==1.62.0.post5 predates MLflow 3.x's reworked Logged Model
    # tracking API (we're on mlflow==3.14.0). No compatible azureml-mlflow release exists yet.
    # Use local SQLite for now to enable model logging.
    # TODO: Re-check azureml-mlflow compatibility with MLflow 3.x in Phase 11 (MLOps hardening).
    if False and settings.azure_ml_mlflow_tracking_uri:  # Disabled - model logging fails with 404
        try:
            mlflow.set_tracking_uri(settings.azure_ml_mlflow_tracking_uri)
            logger.info(f"MLflow tracking → {settings.azure_ml_mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to set Azure ML tracking URI: {e}. Falling back to local SQLite.")
            mlflow.set_tracking_uri("sqlite:///D:/dev/climateguard-ai/mlflow.db")
    else:
        mlflow.set_tracking_uri("sqlite:///D:/dev/climateguard-ai/mlflow.db")
        logger.info("Using local SQLite MLflow tracking")