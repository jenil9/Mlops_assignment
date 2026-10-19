"""Explicit Alternating Least Squares (ALS) matrix factorization in NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def solve_vector(other_factors: np.ndarray, centered_ratings: np.ndarray, reg: float) -> np.ndarray:
    """Solve ridge regression least-squares for one user or item vector."""
    n_ratings, factors = other_factors.shape
    if n_ratings == 0:
        return np.zeros(factors, dtype=np.float64)

    # (V.T @ V + reg * n * I) x = V.T @ r
    diag_reg = reg * float(n_ratings) * np.eye(factors, dtype=np.float64)
    a_mat = other_factors.T @ other_factors + diag_reg
    b_vec = other_factors.T @ centered_ratings
    try:
        return np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]


def predict(user_vec: np.ndarray, item_factors: np.ndarray, global_mean: float) -> np.ndarray:
    """Predict ratings for a user vector across item factors, clipped to [0.5, 5.0]."""
    scores = global_mean + item_factors @ user_vec
    return np.clip(scores, 0.5, 5.0)


def train_als(
    ratings: pd.DataFrame,
    factors: int,
    reg: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Train explicit ALS model on ratings DataFrame."""
    user_ids = np.unique(ratings["userId"].values)
    item_ids = np.unique(ratings["movieId"].values)

    n_users = len(user_ids)
    n_items = len(item_ids)

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {mid: idx for idx, mid in enumerate(item_ids)}

    u_indices = np.array([user_map[u] for u in ratings["userId"].values], dtype=np.int64)
    i_indices = np.array([item_map[m] for m in ratings["movieId"].values], dtype=np.int64)
    raw_ratings = ratings["rating"].values.astype(np.float64)

    global_mean = float(np.mean(raw_ratings))
    centered_ratings = raw_ratings - global_mean

    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0.0, 0.1, size=(n_users, factors)).astype(np.float64)
    item_factors = rng.normal(0.0, 0.1, size=(n_items, factors)).astype(np.float64)

    # Sorted arrays and index pointers for fast slicing
    user_sort_idx = np.argsort(u_indices)
    u_sorted = u_indices[user_sort_idx]
    user_items = i_indices[user_sort_idx]
    user_ratings = centered_ratings[user_sort_idx]
    user_ptrs = np.searchsorted(u_sorted, np.arange(n_users + 1))

    item_sort_idx = np.argsort(i_indices)
    i_sorted = i_indices[item_sort_idx]
    item_users = u_indices[item_sort_idx]
    item_ratings = centered_ratings[item_sort_idx]
    item_ptrs = np.searchsorted(i_sorted, np.arange(n_items + 1))

    for it in range(1, iterations + 1):
        # Update user factors
        for u in range(n_users):
            start = user_ptrs[u]
            end = user_ptrs[u + 1]
            if end > start:
                v_slice = item_factors[user_items[start:end]]
                r_slice = user_ratings[start:end]
                user_factors[u] = solve_vector(v_slice, r_slice, reg)

        # Update item factors
        for i in range(n_items):
            start = item_ptrs[i]
            end = item_ptrs[i + 1]
            if end > start:
                u_slice = user_factors[item_users[start:end]]
                r_slice = item_ratings[start:end]
                item_factors[i] = solve_vector(u_slice, r_slice, reg)

        # Compute training RMSE
        preds = np.clip(
            global_mean + np.sum(user_factors[u_indices] * item_factors[i_indices], axis=1),
            0.5,
            5.0,
        )
        rmse = float(np.sqrt(np.mean((raw_ratings - preds) ** 2)))
        print(f"Iteration {it:2d}/{iterations} - Training RMSE: {rmse:.4f}")

    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "global_mean": global_mean,
    }
