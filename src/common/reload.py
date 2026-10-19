"""Replication reload helpers to notify API replicas of model swaps."""

from __future__ import annotations

import socket
from typing import Any

import requests

from src.common.config import get_settings


def reload_replicas(admin_key: str | None = None) -> list[tuple[str, bool, Any]]:
    """Call POST /reload with X-Admin-Key on each replica sequentially."""
    settings = get_settings()
    key = admin_key or settings.admin_key
    targets: list[str] = []

    if settings.api_headless_host:
        try:
            parts = settings.api_headless_host.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8000
            addr_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
            unique_ips = sorted({ai[4][0] for ai in addr_info})
            targets = [f"http://{ip}:{port}" for ip in unique_ips]
        except Exception:
            targets = [f"http://{settings.api_headless_host}"]

    if not targets:
        targets = settings.api_reload_targets or ["http://127.0.0.1:8000"]

    results: list[tuple[str, bool, Any]] = []
    for target in targets:
        url = f"{target.rstrip('/')}/reload"
        try:
            res = requests.post(url, headers={"X-Admin-Key": key}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                version = data.get("model_version")
                results.append((target, True, version))
            else:
                results.append((target, False, None))
        except Exception:
            results.append((target, False, None))

    return results
