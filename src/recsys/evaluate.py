"""Evaluation metrics for recommendation models: RMSE, Precision@K, NDCG@K."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.recsys.bundle import Bundle
from src.recsys.recommend import recommend_for_user, recommend_popular, user_vector


def _compute_dcg(rec_ids: list[int], relevant_set: set[int]) -> float:
    dcg = 0.0
    for rank, mid in enumerate(rec_ids):
        if mid in relevant_set:
            dcg += 1.0 / math.log2(rank + 2)
    return dcg


def _compute_idcg(n_relevant: int, k: int) -> float:
    n_ideal = min(n_relevant, k)
    return sum(1.0 / math.log2(rank + 2) for rank in range(n_ideal))


def evaluate(
    bundle: Bundle,
    eval_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    min_ratings: int = 10,
) -> dict[str, Any]:
    """Compute RMSE, Precision@K, and NDCG@K for model against evaluation split."""
    squared_errors: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []

    seen_grouped = {uid: group for uid, group in seen_df.groupby("userId")}
    eval_grouped = {uid: group for uid, group in eval_df.groupby("userId")}

    all_users = set(eval_grouped.keys())
    evaluated_ranking_users = 0

    for uid, u_eval in eval_grouped.items():
        u_seen = seen_grouped.get(uid, pd.DataFrame(columns=["movieId", "rating", "timestamp"]))

        # RMSE evaluation
        u_vec = user_vector(bundle, u_seen)
        if u_vec is not None:
            for r in u_eval.itertuples():
                mid = int(r.movieId)
                if mid in bundle.item_to_idx:
                    pred = float(
                        np.clip(
                            bundle.global_mean
                            + u_vec @ bundle.item_factors[bundle.item_to_idx[mid]],
                            0.5,
                            5.0,
                        )
                    )
                    squared_errors.append((float(r.rating) - pred) ** 2)

        # Ranking evaluation: user has >= 1 rating in seen and >= 1 relevant in eval
        if len(u_seen) > 0:
            relevant_items = set(u_eval[u_eval["rating"] >= threshold]["movieId"].astype(int))
            if relevant_items:
                evaluated_ranking_users += 1
                _, rec_movies = recommend_for_user(bundle, u_seen, k)
                rec_ids = [int(m["movie_id"]) for m in rec_movies[:k]]

                hits = sum(1 for m in rec_ids if m in relevant_items)
                precisions.append(hits / float(k))

                dcg = _compute_dcg(rec_ids, relevant_items)
                idcg = _compute_idcg(len(relevant_items), k)
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
    precision = float(np.mean(precisions)) if precisions else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return {
        "rmse": rmse,
        "precision_at_k": precision,
        "ndcg_at_k": ndcg,
        "n_users": len(all_users),
        "n_ranking_users": evaluated_ranking_users,
        "n_rows": int(len(eval_df)),
    }


def evaluate_popularity(
    bundle: Bundle,
    eval_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
) -> dict[str, Any]:
    """Compute baseline popularity metrics (RMSE against global mean)."""
    squared_errors: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []

    seen_grouped = {uid: group for uid, group in seen_df.groupby("userId")}
    eval_grouped = {uid: group for uid, group in eval_df.groupby("userId")}

    for uid, u_eval in eval_grouped.items():
        u_seen = seen_grouped.get(uid, pd.DataFrame(columns=["movieId", "rating", "timestamp"]))

        for r in u_eval.itertuples():
            mid = int(r.movieId)
            if mid in bundle.item_to_idx:
                pred = bundle.global_mean
                squared_errors.append((float(r.rating) - pred) ** 2)

        if len(u_seen) > 0:
            relevant_items = set(u_eval[u_eval["rating"] >= threshold]["movieId"].astype(int))
            if relevant_items:
                seen_ids = set(u_seen["movieId"].astype(int))
                rec_movies = recommend_popular(bundle, seen_ids, k)
                rec_ids = [int(m["movie_id"]) for m in rec_movies[:k]]

                hits = sum(1 for m in rec_ids if m in relevant_items)
                precisions.append(hits / float(k))

                dcg = _compute_dcg(rec_ids, relevant_items)
                idcg = _compute_idcg(len(relevant_items), k)
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
    precision = float(np.mean(precisions)) if precisions else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return {
        "rmse": rmse,
        "precision_at_k": precision,
        "ndcg_at_k": ndcg,
        "n_users": len(eval_grouped),
        "n_rows": int(len(eval_df)),
    }
