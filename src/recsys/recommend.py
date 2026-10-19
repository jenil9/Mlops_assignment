"""Core recommendation algorithms: personalized, popular, and because-you-liked."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.recsys.als import predict, solve_vector
from src.recsys.bundle import Bundle


def user_vector(bundle: Bundle, history: pd.DataFrame) -> np.ndarray | None:
    """Compute or lookup user factor vector according to contract rules."""
    params = bundle.meta.get("params", {})
    serving_params = params.get("serving", {})
    model_params = params.get("model", {})

    min_personalize = int(serving_params.get("min_ratings_personalize", 3))
    if history is None or len(history) < min_personalize:
        return None

    # Check if user already exists in model and history matches saved history
    user_id = None
    if "userId" in history.columns and len(history["userId"].dropna()) > 0:
        unique_uids = history["userId"].dropna().unique()
        if len(unique_uids) == 1:
            user_id = int(unique_uids[0])

    if user_id is not None and user_id in bundle.user_to_idx:
        saved_ratings = bundle.user_history_ratings.get(user_id, {})
        if len(history) == len(saved_ratings):
            all_matched = True
            for r in history.itertuples():
                mid = int(r.movieId)
                if mid not in saved_ratings or abs(saved_ratings[mid] - float(r.rating)) > 1e-6:
                    all_matched = False
                    break
            if all_matched:
                user_idx = bundle.user_to_idx[user_id]
                return bundle.user_factors[user_idx].copy()

    # Otherwise fold in: solve ridge regression using items that exist in the bundle
    reg = float(model_params.get("reg", 0.1))
    valid_mask = history["movieId"].astype(int).isin(bundle.item_to_idx)
    valid_hist = history[valid_mask]
    if len(valid_hist) < min_personalize:
        return None

    item_indices = [bundle.item_to_idx[int(mid)] for mid in valid_hist["movieId"]]
    v_mat = bundle.item_factors[item_indices]
    centered_r = valid_hist["rating"].values.astype(np.float64) - bundle.global_mean
    return solve_vector(v_mat, centered_r, reg)


def recommend_popular(bundle: Bundle, seen: set[int], n: int) -> list[dict[str, Any]]:
    """Return top-N popular movies not yet rated by the user."""
    min_ratings = int(bundle.meta.get("params", {}).get("model", {}).get("min_item_ratings", 10))
    movies: list[dict[str, Any]] = []

    for mid in bundle.popularity_order:
        if mid in seen:
            continue
        if bundle.popularity_counts.get(mid, 0) < min_ratings:
            continue
        if mid not in bundle.item_to_idx:
            continue

        meta = bundle.movie_meta.get(mid, {"title": f"Movie {mid}", "genres": []})
        movies.append(
            {
                "movie_id": mid,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": round(float(bundle.global_mean), 2),
            }
        )
        if len(movies) >= n:
            break

    return movies


def recommend_for_user(
    bundle: Bundle, history: pd.DataFrame, n: int
) -> tuple[str, list[dict[str, Any]]]:
    """Recommend movies for a user, returning ('personalized' | 'popular', movie_list)."""
    seen = set(history["movieId"].astype(int)) if len(history) > 0 else set()
    u_vec = user_vector(bundle, history)

    if u_vec is None:
        return "popular", recommend_popular(bundle, seen, n)

    min_ratings = int(bundle.meta.get("params", {}).get("model", {}).get("min_item_ratings", 10))
    scores = predict(u_vec, bundle.item_factors, bundle.global_mean)

    candidates: list[tuple[int, float]] = []
    for mid, idx in bundle.item_to_idx.items():
        if mid in seen:
            continue
        if bundle.popularity_counts.get(mid, 0) < min_ratings:
            continue
        candidates.append((mid, float(scores[idx])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:n]

    movies: list[dict[str, Any]] = []
    for mid, score in top_candidates:
        meta = bundle.movie_meta.get(mid, {"title": f"Movie {mid}", "genres": []})
        movies.append(
            {
                "movie_id": mid,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": round(score, 2),
            }
        )

    return "personalized", movies


def because_rows(bundle: Bundle, history: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    """Generate 'Because you liked X' rows for up to 3 most recently liked movies."""
    if history is None or len(history) == 0:
        return []

    serving_params = bundle.meta.get("params", {}).get("serving", {})
    model_params = bundle.meta.get("params", {}).get("model", {})
    liked_thresh = float(serving_params.get("liked_threshold", 4.0))
    max_because_rows = int(serving_params.get("because_rows", 3))
    min_ratings = int(model_params.get("min_item_ratings", 10))

    seen = set(history["movieId"].astype(int))
    liked = history[history["rating"] >= liked_thresh].sort_values(by="timestamp", ascending=False)

    rows: list[dict[str, Any]] = []
    processed_liked: set[int] = set()

    for r in liked.itertuples():
        x_id = int(r.movieId)
        if x_id in processed_liked:
            continue
        if x_id not in bundle.item_to_idx:
            continue

        processed_liked.add(x_id)
        x_idx = bundle.item_to_idx[x_id]
        norm_x = bundle.normalized_item_factors[x_idx]
        sims = bundle.normalized_item_factors @ norm_x

        sim_candidates: list[tuple[int, float]] = []
        for mid, idx in bundle.item_to_idx.items():
            if mid == x_id or mid in seen:
                continue
            if bundle.popularity_counts.get(mid, 0) < min_ratings:
                continue
            sim_candidates.append((mid, float(sims[idx])))

        sim_candidates.sort(key=lambda x: x[1], reverse=True)
        top_sim = sim_candidates[:n]

        rec_movies: list[dict[str, Any]] = []
        for mid, score in top_sim:
            meta = bundle.movie_meta.get(mid, {"title": f"Movie {mid}", "genres": []})
            rec_movies.append(
                {
                    "movie_id": mid,
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": round(score, 2),
                }
            )

        if rec_movies:
            x_meta = bundle.movie_meta.get(x_id, {"title": f"Movie {x_id}"})
            rows.append(
                {
                    "because_you_liked": {"movie_id": x_id, "title": x_meta["title"]},
                    "movies": rec_movies,
                }
            )

        if len(rows) >= max_because_rows:
            break

    return rows
