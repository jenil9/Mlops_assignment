"""Model bundle packaging, persistence, and fast in-memory indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Bundle:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    global_mean: float
    movies: pd.DataFrame
    popularity: pd.DataFrame
    user_history: pd.DataFrame
    reference: pd.DataFrame
    meta: dict[str, Any]

    # Pre-computed fast lookup caches
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    normalized_item_factors: np.ndarray
    popularity_order: list[int]
    popularity_counts: dict[int, int]
    user_history_map: dict[int, set[int]]
    movie_meta: dict[int, dict[str, Any]]


def build_bundle(
    model: dict[str, Any],
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    params: dict[str, Any],
    feedback_rows: int = 0,
    parent_version: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> Bundle:
    """Build a complete model bundle from trained factors, datasets, and metadata."""
    user_factors = model["user_factors"]
    item_factors = model["item_factors"]
    user_ids = model["user_ids"]
    item_ids = model["item_ids"]
    global_mean = float(model["global_mean"])

    # Compute popularity from the training ratings
    pop = (
        ratings.groupby("movieId", as_index=False)
        .size()
        .rename(columns={"size": "n_ratings"})
        .sort_values(by="n_ratings", ascending=False)
        .reset_index(drop=True)
    )
    pop["movieId"] = pop["movieId"].astype("int64")
    pop["n_ratings"] = pop["n_ratings"].astype("int64")

    # Build reference dataset for drift detection
    movie_genre_map: dict[int, str] = {}
    for _, row in movies.iterrows():
        mid = int(row["movieId"])
        g_list = row["genres"]
        if isinstance(g_list, (list, np.ndarray)) and len(g_list) > 0 and g_list[0]:
            movie_genre_map[mid] = str(g_list[0])
        else:
            movie_genre_map[mid] = "none"

    pop_map: dict[int, int] = dict(zip(pop["movieId"], pop["n_ratings"]))

    ref_rows = []
    for _, r in ratings.iterrows():
        mid = int(r["movieId"])
        ref_rows.append(
            {
                "rating": float(r["rating"]),
                "main_genre": movie_genre_map.get(mid, "none"),
                "item_popularity": int(pop_map.get(mid, 0)),
            }
        )
    ref_df = pd.DataFrame(ref_rows)

    hist_df = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    hist_df["userId"] = hist_df["userId"].astype("int64")
    hist_df["movieId"] = hist_df["movieId"].astype("int64")
    hist_df["rating"] = hist_df["rating"].astype("float64")
    hist_df["timestamp"] = hist_df["timestamp"].astype("int64")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics or {},
        "train_rows": int(len(ratings)),
        "feedback_rows": int(feedback_rows),
        "parent_version": parent_version,
    }

    return _create_bundle_instance(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
        global_mean=global_mean,
        movies=movies,
        popularity=pop,
        user_history=hist_df,
        reference=ref_df,
        meta=meta,
    )


def _create_bundle_instance(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    global_mean: float,
    movies: pd.DataFrame,
    popularity: pd.DataFrame,
    user_history: pd.DataFrame,
    reference: pd.DataFrame,
    meta: dict[str, Any],
) -> Bundle:
    user_to_idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {int(mid): idx for idx, mid in enumerate(item_ids)}

    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    normalized_item_factors = item_factors / np.maximum(norms, 1e-9)

    popularity_order = [int(x) for x in popularity["movieId"].tolist()]
    popularity_counts = {int(r.movieId): int(r.n_ratings) for r in popularity.itertuples()}

    user_history_map: dict[int, set[int]] = {}
    for uid, group in user_history.groupby("userId"):
        user_history_map[int(uid)] = set(group["movieId"].astype(int))

    movie_meta: dict[int, dict[str, Any]] = {}
    for r in movies.itertuples():
        mid = int(r.movieId)
        genres = list(r.genres) if isinstance(r.genres, (list, np.ndarray)) else []
        movie_meta[mid] = {"title": str(r.title), "genres": genres}

    return Bundle(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
        global_mean=global_mean,
        movies=movies,
        popularity=popularity,
        user_history=user_history,
        reference=reference,
        meta=meta,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        normalized_item_factors=normalized_item_factors,
        popularity_order=popularity_order,
        popularity_counts=popularity_counts,
        user_history_map=user_history_map,
        movie_meta=movie_meta,
    )


def save_bundle(bundle: Bundle, folder: str | Path) -> None:
    """Save all bundle artifacts to target folder."""
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "factors.npz",
        user_factors=bundle.user_factors,
        item_factors=bundle.item_factors,
        user_ids=bundle.user_ids,
        item_ids=bundle.item_ids,
        global_mean=np.float64(bundle.global_mean),
    )
    bundle.movies.to_parquet(out_dir / "movies.parquet", index=False)
    bundle.popularity.to_parquet(out_dir / "popularity.parquet", index=False)
    bundle.user_history.to_parquet(out_dir / "user_history.parquet", index=False)
    bundle.reference.to_parquet(out_dir / "reference.parquet", index=False)

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.meta, f, indent=2)


def load_bundle(folder: str | Path) -> Bundle:
    """Load model bundle from disk and construct fast in-memory indexes."""
    in_dir = Path(folder)
    factors_npz = np.load(in_dir / "factors.npz")
    user_factors = factors_npz["user_factors"]
    item_factors = factors_npz["item_factors"]
    user_ids = factors_npz["user_ids"]
    item_ids = factors_npz["item_ids"]
    global_mean = float(factors_npz["global_mean"])

    movies = pd.read_parquet(in_dir / "movies.parquet")
    popularity = pd.read_parquet(in_dir / "popularity.parquet")
    user_history = pd.read_parquet(in_dir / "user_history.parquet")
    reference = pd.read_parquet(in_dir / "reference.parquet")

    with (in_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return _create_bundle_instance(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
        global_mean=global_mean,
        movies=movies,
        popularity=popularity,
        user_history=user_history,
        reference=reference,
        meta=meta,
    )
