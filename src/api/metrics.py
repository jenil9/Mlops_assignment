"""Prometheus metrics definitions and instrumentation for the API."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from src.common.config import get_settings

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled by API",
    ["endpoint", "method", "status", "source"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
)

recommendations_served_total = Counter(
    "recommendations_served_total",
    "Total recommendations served by recommendation type",
    ["kind"],
)

feedback_total = Counter(
    "feedback_total",
    "Total feedback submissions received",
    ["source"],
)

feedback_rating_sum = Counter(
    "feedback_rating_sum",
    "Sum of feedback rating scores received",
    ["source"],
)

model_version_gauge = Gauge(
    "model_version",
    "Currently served model version number",
)


def set_model_version(version: int) -> None:
    """Set Prometheus gauge to active model version."""
    model_version_gauge.set(version)


def record_recommendation(kind: str) -> None:
    """Increment recommendation served counter."""
    recommendations_served_total.labels(kind=kind).inc()


def record_feedback(source: str, rating: float) -> None:
    """Increment feedback count and feedback score sum counters."""
    feedback_total.labels(source=source).inc()
    feedback_rating_sum.labels(source=source).inc(rating)


def metrics_response() -> Response:
    """Return raw Prometheus metrics string with appropriate content-type."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware recording HTTP request counts and latency histograms."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Check source header
        settings = get_settings()
        sim_header = request.headers.get("x-simulator-key")
        source = "simulator" if sim_header == settings.simulator_key else "web"

        response = await call_next(request)
        duration = time.perf_counter() - start_time

        # Match route template for Prometheus endpoint label
        endpoint = "unmatched"
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                endpoint = route.path
                break

        status_str = str(response.status_code)
        http_requests_total.labels(
            endpoint=endpoint,
            method=request.method,
            status=status_str,
            source=source,
        ).inc()

        http_request_duration_seconds.labels(endpoint=endpoint).observe(duration)
        return response
