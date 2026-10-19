"""Tests for ranking evaluation metrics and NDCG@K computation."""

from __future__ import annotations

import pandas as pd

from src.recsys.bundle import build_bundle
from src.recsys.evaluate import _compute_dcg, _compute_idcg, evaluate


def test_ndcg_perfect_match_and_zero_hits() -> None:
    """Check that perfect recommendation ranking yields NDCG@K=1.0 and disjoint gives 0.0."""
    relevant_items = {10, 20, 30}
    k = 5

    # 1. Perfect ordering of recommendations
    perfect_recs = [10, 20, 30, 40, 50]
    dcg_perfect = _compute_dcg(perfect_recs, relevant_items)
    idcg_perfect = _compute_idcg(len(relevant_items), k)
    assert idcg_perfect > 0
    assert abs(dcg_perfect / idcg_perfect - 1.0) < 1e-9

    # 2. No hits ordering
    zero_recs = [90, 91, 92, 93, 94]
    dcg_zero = _compute_dcg(zero_recs, relevant_items)
    assert dcg_zero == 0.0


def test_evaluate_metric_values() -> None:
    """Check evaluate function returns expected metric keys and bounded numbers."""
    import numpy as np

    user_factors = np.array([[1.0, 0.0]], dtype=np.float64)
    item_factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 100},
            {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 200},
            {"userId": 1, "movieId": 3, "rating": 3.0, "timestamp": 300},
        ]
    )
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "M1", "genres": ["Action"]},
            {"movieId": 2, "title": "M2", "genres": ["Action"]},
            {"movieId": 3, "title": "M3", "genres": ["Action"]},
        ]
    )
    params = {
        "model": {"factors": 2, "reg": 0.1, "min_item_ratings": 1},
        "serving": {"min_ratings_personalize": 2, "liked_threshold": 4.0},
    }
    model = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_ids": np.array([1]),
        "item_ids": np.array([1, 2, 3]),
        "global_mean": 4.0,
    }
    bundle = build_bundle(model, ratings, movies, params)

    seen_df = ratings.copy()
    eval_df = pd.DataFrame([{"userId": 1, "movieId": 2, "rating": 4.5, "timestamp": 400}])
    results = evaluate(bundle, eval_df, seen_df, k=2, threshold=4.0)

    assert "rmse" in results
    assert "precision_at_k" in results
    assert "ndcg_at_k" in results
    assert 0.0 <= results["ndcg_at_k"] <= 1.0
