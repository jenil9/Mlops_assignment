"""Project configuration and parameter loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    database_url: str
    mlflow_tracking_uri: str
    model_name: str
    simulator_key: str
    admin_key: str
    replica_name: str
    api_reload_targets: list[str]
    api_headless_host: str | None


def get_settings() -> Settings:
    """Read application settings from environment variables."""
    reload_targets_raw = os.getenv("API_RELOAD_TARGETS", "")
    reload_targets = (
        [t.strip() for t in reload_targets_raw.split(",") if t.strip()]
        if reload_targets_raw
        else []
    )
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./local.db"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        model_name=os.getenv("MODEL_NAME", "movie-recommender"),
        simulator_key=os.getenv("SIMULATOR_KEY", "sim-secret-change-me"),
        admin_key=os.getenv("ADMIN_KEY", "admin-secret-change-me"),
        replica_name=os.getenv("REPLICA_NAME", "local"),
        api_reload_targets=reload_targets,
        api_headless_host=os.getenv("API_HEADLESS_HOST") or None,
    )


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load hyperparameters from YAML file (default params.yaml)."""
    if path is None:
        path = os.getenv("PARAMS_PATH", "params.yaml")
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Params file not found: {filepath}")
    with filepath.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
