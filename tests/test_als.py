"""Tests for explicit ALS matrix factorization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recsys.als import train_als


def test_als_training_rmse_and_reproducibility() -> None:
    """Check that ALS improves over global mean on low-rank data and seed is reproducible."""
    rng = np.random.default_rng(123)
    # Synthetic low-rank data: 20 users, 15 items, rank 2
    u_true = rng.normal(size=(20, 2))
    v_true = rng.normal(size=(15, 2))
    r_matrix = 3.0 + u_true @ v_true.T
    r_matrix = np.clip(r_matrix, 1.0, 5.0)

    rows = []
    for u in range(20):
        for i in range(15):
            rows.append({"userId": u + 1, "movieId": i + 1, "rating": float(r_matrix[u, i])})
    df = pd.DataFrame(rows)

    model1 = train_als(df, factors=4, reg=0.01, iterations=10, seed=42)
    model2 = train_als(df, factors=4, reg=0.01, iterations=10, seed=42)

    # Check reproducibility
    np.testing.assert_allclose(model1["user_factors"], model2["user_factors"], rtol=1e-5)
    np.testing.assert_allclose(model1["item_factors"], model2["item_factors"], rtol=1e-5)

    # Check that model predictions beat global mean baseline
    global_mean = float(df["rating"].mean())
    baseline_rmse = float(np.sqrt(np.mean((df["rating"].values - global_mean) ** 2)))

    u_indices = np.array([u - 1 for u in df["userId"].values])
    i_indices = np.array([i - 1 for i in df["movieId"].values])
    preds = np.clip(
        global_mean
        + np.sum(model1["user_factors"][u_indices] * model1["item_factors"][i_indices], axis=1),
        0.5,
        5.0,
    )
    final_rmse = float(np.sqrt(np.mean((df["rating"].values - preds) ** 2)))

    assert final_rmse < baseline_rmse
