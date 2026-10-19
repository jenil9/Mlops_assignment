"""In-memory state management for active recommendation model bundle."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.recsys.bundle import Bundle
from src.recsys.registry import fetch_alias

if TYPE_CHECKING:
    from src.common.config import Settings


@dataclass(frozen=True)
class LoadedModel:
    bundle: Bundle
    version: int


def load_champion(settings: Settings) -> LoadedModel:
    """Fetch current champion bundle and version from MLflow registry."""
    bundle, version = fetch_alias(settings, "champion")
    return LoadedModel(bundle=bundle, version=version)


class ModelHolder:
    """Thread-safe container supporting atomic model swapping during reload."""

    def __init__(self, initial_model: LoadedModel | None = None) -> None:
        self._model = initial_model
        self._lock = threading.Lock()

    def get(self) -> LoadedModel | None:
        return self._model

    def swap(self, new_model: LoadedModel) -> None:
        with self._lock:
            self._model = new_model
