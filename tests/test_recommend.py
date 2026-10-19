"""Tests for recommendation rules, popular fallback, fold-in, and because rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recsys.bundle import build_bundle
from src.recsys.recommend import because_rows, recommend_for_user, recommend_popular, user_vector


@pytest.fixture
def sample_bundle() -> tuple:
    user_factors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    item_factors = np.array(
        [[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]], dtype=np.float64
    )
    user_ids = np.array([1, 2], dtype=np.int64)
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    # 10 ratings for items 10, 20, 30, 40 to pass min_item_ratings; item 50 has only 2 ratings
    ratings_rows = []
    for mid in [10, 20, 30, 40]:
        for uid in range(1, 11):
            ratings_rows.append(
                {"userId": uid, "movieId": mid, "rating": 4.0, "timestamp": 1000 + uid}
            )
    ratings_rows.append({"userId": 1, "movieId": 50, "rating": 3.0, "timestamp": 1050})
    ratings_rows.append({"userId": 2, "movieId": 50, "rating": 3.0, "timestamp": 1051})
    ratings_df = pd.DataFrame(ratings_rows)

    movies_df = pd.DataFrame(
        [
            {"movieId": 10, "title": "Movie 10", "genres": ["Action"]},
            {"movieId": 20, "title": "Movie 20", "genres": ["Action"]},
            {"movieId": 30, "title": "Movie 30", "genres": ["Comedy"]},
            {"movieId": 40, "title": "Movie 40", "genres": ["Comedy"]},
            {"movieId": 50, "title": "Movie 50", "genres": ["Drama"]},
        ]
    )

    params = {
        "model": {"factors": 2, "reg": 0.1, "min_item_ratings": 5},
        "serving": {
            "min_ratings_personalize": 3,
            "liked_threshold": 4.0,
            "because_rows": 3,
        },
    }

    model = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "global_mean": 3.5,
    }

    bundle = build_bundle(model, ratings_df, movies_df, params)
    return bundle, ratings_df


def test_popular_skips_seen_and_low_count(sample_bundle) -> None:
    bundle, _ = sample_bundle
    seen = {10}
    recs = recommend_popular(bundle, seen, n=10)
    rec_ids = [m["movie_id"] for m in recs]
    # Movie 10 is seen, movie 50 has only 2 ratings (< min_item_ratings 5)
    assert 10 not in rec_ids
    assert 50 not in rec_ids
    assert 20 in rec_ids


def test_fewer_than_three_ratings_gets_popular(sample_bundle) -> None:
    bundle, _ = sample_bundle
    history = pd.DataFrame(
        [
            {"userId": 999, "movieId": 10, "rating": 4.0, "timestamp": 100},
            {"userId": 999, "movieId": 20, "rating": 4.5, "timestamp": 200},
        ]
    )
    kind, recs = recommend_for_user(bundle, history, n=5)
    assert kind == "popular"
    rec_ids = [m["movie_id"] for m in recs]
    assert 10 not in rec_ids and 20 not in rec_ids


def test_saved_vector_vs_fold_in(sample_bundle) -> None:
    bundle, _ = sample_bundle
    # User 1 exists in model; bundle user 1 rated 10, 20, 30, 40, 50
    # Perfect match with bundle history
    hist_exact = bundle.user_history[bundle.user_history["userId"] == 1].copy()
    vec_exact = user_vector(bundle, hist_exact)
    saved_vec = bundle.user_factors[bundle.user_to_idx[1]]
    np.testing.assert_allclose(vec_exact, saved_vec, rtol=1e-5)

    # Extra rating -> fold in
    extra_row = pd.DataFrame([{"userId": 1, "movieId": 30, "rating": 1.0, "timestamp": 9999}])
    hist_modified = pd.concat([hist_exact, extra_row], ignore_index=True)
    vec_folded = user_vector(bundle, hist_modified)
    assert not np.allclose(vec_folded, saved_vec)


def test_because_rows_constraints(sample_bundle) -> None:
    bundle, _ = sample_bundle
    # User liked movies 10, 20, 30, 40 with 4.5+ ratings
    history = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 100},
            {"userId": 1, "movieId": 20, "rating": 4.5, "timestamp": 200},
            {"userId": 1, "movieId": 30, "rating": 4.0, "timestamp": 300},
            {"userId": 1, "movieId": 40, "rating": 4.5, "timestamp": 400},
        ]
    )
    rows = because_rows(bundle, history, n=5)
    # At most 3 rows
    assert len(rows) <= 3

    seen = {10, 20, 30, 40}
    for row in rows:
        liked_id = row["because_you_liked"]["movie_id"]
        assert liked_id in seen
        rec_ids = [m["movie_id"] for m in row["movies"]]
        # Never includes the liked movie itself or any seen movies
        assert liked_id not in rec_ids
        for s in seen:
            assert s not in rec_ids
