"""MLflow model registry client and artifact caching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.recsys.bundle import Bundle, load_bundle

if TYPE_CHECKING:
    from src.common.config import Settings


def alias_version(settings: Settings, alias: str) -> int | None:
    """Return model version pointed to by alias, or None if alias does not exist."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        mv = client.get_model_version_by_alias(settings.model_name, alias)
        return int(mv.version)
    except (MlflowException, Exception):
        return None


def fetch_bundle(settings: Settings, version: int) -> Bundle:
    """Download and cache model bundle artifact from MLflow run for specific version."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    cache_dir = Path("/tmp/model_cache") / str(version)
    bundle_path = cache_dir / "bundle" if (cache_dir / "bundle").exists() else cache_dir

    if (bundle_path / "factors.npz").exists() and (bundle_path / "meta.json").exists():
        return load_bundle(bundle_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    mv = client.get_model_version(settings.model_name, str(version))
    downloaded_path = client.download_artifacts(
        run_id=mv.run_id, path="bundle", dst_path=str(cache_dir)
    )
    return load_bundle(downloaded_path)


def fetch_alias(settings: Settings, alias: str) -> tuple[Bundle, int]:
    """Fetch model bundle and version number for registered alias."""
    version = alias_version(settings, alias)
    if version is None:
        raise ValueError(f"Alias '{alias}' not found for model '{settings.model_name}'")
    bundle = fetch_bundle(settings, version)
    return bundle, version
