"""Tests for drift detection logic (pure functions, no MLflow, no Docker)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.drift.drift_check import build_current, decide, live_hit_rate

# ---------------------------------------------------------------------------
# decide() tests
# ---------------------------------------------------------------------------


class TestDecide:
    """Decision table for decide(share, n_rows, params)."""

    PARAMS = {
        "drift": {
            "min_rows": 500,
            "share_threshold": 0.3,
        }
    }

    def test_below_min_rows_no_drift(self):
        assert decide(1.0, 499, self.PARAMS) is False

    def test_at_min_rows_above_threshold(self):
        assert decide(0.33, 500, self.PARAMS) is True

    def test_above_min_rows_below_threshold(self):
        # share < 0.3 should be False
        assert decide(0.0, 1000, self.PARAMS) is False

    def test_above_min_rows_at_threshold(self):
        assert decide(0.33, 1000, self.PARAMS) is True

    def test_full_drift(self):
        assert decide(1.0, 1000, self.PARAMS) is True

    def test_high_threshold_66_percent(self):
        params = {"drift": {"min_rows": 500, "share_threshold": 0.67}}
        # share=0.33 should not trigger with threshold=0.67
        assert decide(0.33, 1000, params) is False
        # share=0.67 should trigger
        assert decide(0.67, 1000, params) is True

    def test_exactly_at_boundary(self):
        # n_rows exactly at min_rows with share exactly at threshold
        assert decide(0.3, 500, self.PARAMS) is True

    def test_zero_rows(self):
        assert decide(0.33, 0, self.PARAMS) is False


# ---------------------------------------------------------------------------
# build_current() tests
# ---------------------------------------------------------------------------


class FakeBundle:
    """Minimal bundle stub for build_current tests."""

    def __init__(self, movies: pd.DataFrame, popularity: pd.DataFrame):
        self.movies = movies
        self.popularity = popularity


def _make_bundle(
    movie_ids: list[int],
    genres_list: list[list[str]],
    n_ratings_list: list[int],
) -> FakeBundle:
    movies = pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Movie {m}" for m in movie_ids],
            "genres": genres_list,
        }
    )
    popularity = pd.DataFrame({"movieId": movie_ids, "n_ratings": n_ratings_list})
    return FakeBundle(movies=movies, popularity=popularity)


class TestBuildCurrent:
    """Tests for build_current mapping feedback rows to drift columns."""

    def test_first_genre_extracted(self):
        bundle = _make_bundle([1], [["Action", "Drama"]], [50])
        window = pd.DataFrame(
            {
                "user_id": [1],
                "movie_id": [1],
                "rating": [4.0],
                "source": ["simulator"],
                "created_at": [datetime.now(timezone.utc)],
            }
        )
        result = build_current(window, bundle)
        assert result["main_genre"].iloc[0] == "Action"

    def test_unknown_movie_uses_none_and_zero(self):
        bundle = _make_bundle([1], [["Comedy"]], [10])
        window = pd.DataFrame(
            {
                "user_id": [1],
                "movie_id": [999],  # not in bundle
                "rating": [3.0],
                "source": ["web"],
                "created_at": [datetime.now(timezone.utc)],
            }
        )
        result = build_current(window, bundle)
        assert result["main_genre"].iloc[0] == "none"
        assert result["item_popularity"].iloc[0] == 0

    def test_correct_columns(self):
        bundle = _make_bundle([42], [["Horror"]], [100])
        window = pd.DataFrame(
            {
                "user_id": [7],
                "movie_id": [42],
                "rating": [5.0],
                "source": ["simulator"],
                "created_at": [datetime.now(timezone.utc)],
            }
        )
        result = build_current(window, bundle)
        assert set(result.columns) == {"rating", "main_genre", "item_popularity"}
        assert result["rating"].iloc[0] == 5.0
        assert result["item_popularity"].iloc[0] == 100


# ---------------------------------------------------------------------------
# live_hit_rate() tests
# ---------------------------------------------------------------------------


class TestLiveHitRate:
    """Tests for live_hit_rate on hand-crafted examples."""

    def _make_window(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def _make_logs(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_hit_counts(self):
        t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)

        window = self._make_window(
            [
                # user 1: liked movie 10, which IS in the recommend list -> hit
                {"user_id": 1, "movie_id": 10, "rating": 5.0, "source": "web", "created_at": t2},
                # user 1: liked movie 99, which is NOT in the recommend list -> miss
                {"user_id": 1, "movie_id": 99, "rating": 4.0, "source": "web", "created_at": t2},
                # user 2: liked movie 20, but no earlier log -> excluded
                {"user_id": 2, "movie_id": 20, "rating": 5.0, "source": "web", "created_at": t2},
            ]
        )

        logs = self._make_logs(
            [
                {
                    "user_id": 1,
                    "kind": "personalized",
                    "movie_ids": json.dumps([10, 11, 12]),
                    "created_at": t1,
                },
            ]
        )

        rate = live_hit_rate(window, logs)
        # 1 hit out of 2 qualifying (user 1 has 2 liked rated with an earlier log)
        # user 2 has no earlier log, so excluded
        assert rate == pytest.approx(0.5)

    def test_no_qualifying_rows_returns_none(self):
        t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        # No liked ratings
        window = self._make_window(
            [{"user_id": 1, "movie_id": 5, "rating": 2.0, "source": "web", "created_at": t1}]
        )
        logs = self._make_logs(
            [
                {
                    "user_id": 1,
                    "kind": "popular",
                    "movie_ids": json.dumps([5, 6]),
                    "created_at": t1,
                }
            ]
        )
        assert live_hit_rate(window, logs) is None

    def test_empty_inputs_return_none(self):
        assert live_hit_rate(pd.DataFrame(), pd.DataFrame()) is None
