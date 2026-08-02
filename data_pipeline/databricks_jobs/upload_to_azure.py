"""
Upload local Bronze/Silver/Gold files to their corresponding Azure Data
Lake Storage Gen2 containers.

COST/INFRASTRUCTURE NOTE: rather than provisioning an actual Azure Data
Factory pipeline for this, we use a direct, authenticated upload script.
At this project's scale (a handful of files, run on-demand rather than on
a real production schedule), a full ADF pipeline is infrastructure
overhead without a corresponding benefit — consistent with the same
cost-conscious reasoning already applied to the Databricks decision (see
data_pipeline/databricks_jobs module docstrings). ADF orchestration is
architecturally described in the design doc as the intended production
pattern; this script is the pragmatic vertical-slice substitute.

AUTHENTICATION: uses DefaultAzureCredential (delegates to your `az login`
session) rather than a storage account key in .env — no long-lived secret
needs to be stored anywhere for this to work locally.

Usage:
    python -m data_pipeline.databricks_jobs.upload_to_azure --layer gold
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from config.settings import get_settings

logger = logging.getLogger(__name__)

LAYER_LOCAL_ROOTS = {
    "bronze": Path("data_pipeline/bronze"),
    "silver": Path("data_pipeline/silver"),
    "gold": Path("data_pipeline/gold"),
}


def get_blob_service_client() -> BlobServiceClient:
    settings = get_settings()
    account_name = settings.require("azure_storage_account_name")
    account_url = f"https://{account_name}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=credential)


def upload_layer(layer: str, blob_service_client: BlobServiceClient | None = None) -> list[str]:
    """
    Upload every file under the local layer's root directory to the
    matching Azure container (container name == layer name, e.g. "gold").

    Args:
        layer: one of "bronze", "silver", "gold".
        blob_service_client: optionally inject a client (used by tests to
            pass a mock); if None, a real authenticated client is created.

    Returns:
        List of blob names successfully uploaded.

    Raises:
        ValueError: if `layer` is not a recognized layer name.
        FileNotFoundError: if the local layer root directory doesn't exist
            or contains no files.
    """
    if layer not in LAYER_LOCAL_ROOTS:
        raise ValueError(
            f"Unknown layer '{layer}' — must be one of {list(LAYER_LOCAL_ROOTS.keys())}"
        )

    local_root = LAYER_LOCAL_ROOTS[layer]
    if not local_root.exists():
        raise FileNotFoundError(
            f"Local layer root {local_root} does not exist — nothing to upload."
        )

    files = [p for p in local_root.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found under {local_root} — nothing to upload.")

    client = blob_service_client or get_blob_service_client()
    container_client = client.get_container_client(layer)

    uploaded = []
    for file_path in files:
        blob_name = str(file_path.relative_to(local_root)).replace("\\", "/")
        logger.info("Uploading %s -> container '%s', blob '%s'", file_path, layer, blob_name)
        with open(file_path, "rb") as f:
            container_client.upload_blob(name=blob_name, data=f, overwrite=True)
        uploaded.append(blob_name)

    logger.info("Uploaded %d file(s) to the '%s' container", len(uploaded), layer)
    return uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer",
        required=True,
        choices=list(LAYER_LOCAL_ROOTS.keys()),
        help="Which layer to upload",
    )
    args = parser.parse_args()
    upload_layer(args.layer)