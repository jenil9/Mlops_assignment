"""Pandera validation schemas for raw MovieLens data."""

from __future__ import annotations

from pandera import Check, Column, DataFrameSchema

ratings_schema = DataFrameSchema(
    columns={
        "userId": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "movieId": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "rating": Column(float, Check.in_range(0.5, 5.0), nullable=False),
        "timestamp": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    },
    strict=True,
    coerce=True,
)

movies_schema = DataFrameSchema(
    columns={
        "movieId": Column(int, unique=True, nullable=False),
        "title": Column(str, nullable=False),
        "genres": Column(str, nullable=False),
    },
    strict=True,
    coerce=True,
)
