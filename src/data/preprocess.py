"""Data preprocessing and time-based splitting pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import load_params
from src.data.schemas import movies_schema, ratings_schema


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid movie rows, remove duplicates, and parse genres into lists."""
    df = movies.dropna(subset=["movieId", "title"]).drop_duplicates(subset=["movieId"]).copy()

    def parse_genres(g: Any) -> list[str]:
        if not isinstance(g, str) or not g.strip() or g.strip() == "(no genres listed)":
            return []
        parts = [p.strip() for p in g.split("|") if p.strip()]
        return [] if parts == ["(no genres listed)"] else parts

    df["genres"] = df["genres"].apply(parse_genres)
    df["movieId"] = df["movieId"].astype("int64")
    df["title"] = df["title"].astype(str)
    return df.reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame, movie_ids: set[int] | list[int]) -> pd.DataFrame:
    """Drop nulls, keep latest rating per user-movie pair, filter known movies, and add datetime."""
    df = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()

    # Per (userId, movieId) keep only the latest by timestamp
    df = df.sort_values(by=["userId", "movieId", "timestamp"]).drop_duplicates(
        subset=["userId", "movieId"], keep="last"
    )

    valid_ids = set(movie_ids)
    df = df[df["movieId"].isin(valid_ids)].copy()

    df["userId"] = df["userId"].astype("int64")
    df["movieId"] = df["movieId"].astype("int64")
    df["rating"] = df["rating"].astype("float64")
    df["timestamp"] = df["timestamp"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    df = df.sort_values(by=["timestamp", "userId", "movieId"]).reset_index(drop=True)
    return df


def split_by_time(ratings: pd.DataFrame, fractions: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Split ratings sequentially by time into train, val, test, and stream partitions."""
    required_keys = {"train", "val", "test", "stream"}
    if not required_keys.issubset(fractions.keys()):
        raise ValueError(f"Fractions must contain all keys: {required_keys}")

    total_frac = sum(fractions[k] for k in ["train", "val", "test", "stream"])
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1.0 (got {total_frac})")

    df = ratings.sort_values(by=["timestamp", "userId", "movieId"]).reset_index(drop=True)
    n = len(df)

    idx_train = round(n * fractions["train"])
    idx_val = round(n * (fractions["train"] + fractions["val"]))
    idx_test = round(n * (fractions["train"] + fractions["val"] + fractions["test"]))

    return {
        "train": df.iloc[:idx_train].reset_index(drop=True),
        "val": df.iloc[idx_train:idx_val].reset_index(drop=True),
        "test": df.iloc[idx_val:idx_test].reset_index(drop=True),
        "stream": df.iloc[idx_test:].reset_index(drop=True),
    }


def compute_metrics(splits: dict[str, pd.DataFrame], movies: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics and unseen-user counts for each split."""
    train_users = set(splits["train"]["userId"])
    metrics: dict[str, Any] = {"n_movies": len(movies)}

    for name in ["train", "val", "test", "stream"]:
        split_df = splits[name]
        split_metrics: dict[str, Any] = {
            "rows": int(len(split_df)),
            "users": int(split_df["userId"].nunique()) if len(split_df) > 0 else 0,
            "start": str(split_df["datetime"].min()) if len(split_df) > 0 else "",
            "end": str(split_df["datetime"].max()) if len(split_df) > 0 else "",
        }
        if name != "train":
            split_users = set(split_df["userId"])
            split_metrics["users_not_in_train"] = int(len(split_users - train_users))
        metrics[name] = split_metrics

    return metrics


def main() -> None:
    """Preprocess raw ratings and movies, write parquet splits, and output metrics."""
    params = load_params()
    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_ratings = pd.read_csv(raw_dir / "ratings.csv")
    raw_movies = pd.read_csv(raw_dir / "movies.csv")

    validated_ratings = ratings_schema.validate(raw_ratings, lazy=True)
    validated_movies = movies_schema.validate(raw_movies, lazy=True)

    movies_clean = clean_movies(validated_movies)
    movie_ids = set(movies_clean["movieId"])
    ratings_clean = clean_ratings(validated_ratings, movie_ids)

    splits = split_by_time(ratings_clean, params["split"])
    metrics = compute_metrics(splits, movies_clean)

    for name, df in splits.items():
        df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    movies_clean.to_parquet(processed_dir / "movies.parquet", index=False)

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "preprocess.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Preprocessing completed successfully.")
    print(f"Total cleaned movies: {len(movies_clean)}")
    for name, df in splits.items():
        print(f"Split {name}: {len(df)} rows, {df['userId'].nunique()} users")


if __name__ == "__main__":
    main()
