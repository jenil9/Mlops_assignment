"""Unit tests for preprocessing logic and data splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import clean_ratings, split_by_time


def test_latest_duplicate_wins() -> None:
    """Check that for duplicate (userId, movieId), the one with latest timestamp is kept."""
    raw_ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 3.0, "timestamp": 100},
            {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 200},
            {"userId": 1, "movieId": 20, "rating": 4.0, "timestamp": 150},
        ]
    )
    cleaned = clean_ratings(raw_ratings, movie_ids={10, 20})
    assert len(cleaned) == 2
    row = cleaned[(cleaned["userId"] == 1) & (cleaned["movieId"] == 10)].iloc[0]
    assert row["rating"] == 5.0
    assert row["timestamp"] == 200


def test_unknown_movies_dropped() -> None:
    """Check that ratings for movies not in the known movie list are dropped."""
    raw_ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 3.0, "timestamp": 100},
            {"userId": 1, "movieId": 999, "rating": 4.0, "timestamp": 110},
        ]
    )
    cleaned = clean_ratings(raw_ratings, movie_ids={10})
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["movieId"] == 10


def test_split_sizes_and_time_order() -> None:
    """Check that split sizes sum to total and timestamps are monotonically non-decreasing."""
    rows = [{"userId": i, "movieId": 10, "rating": 4.0, "timestamp": 1000 + i} for i in range(100)]
    df = clean_ratings(pd.DataFrame(rows), movie_ids={10})
    fractions = {"train": 0.70, "val": 0.10, "test": 0.10, "stream": 0.10}
    splits = split_by_time(df, fractions)

    total_split_rows = sum(len(s) for s in splits.values())
    assert total_split_rows == 100
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert len(splits["stream"]) == 10

    # max timestamp of each piece <= next piece min timestamp
    split_order = ["train", "val", "test", "stream"]
    for i in range(len(split_order) - 1):
        s_curr = splits[split_order[i]]
        s_next = splits[split_order[i + 1]]
        assert s_curr["timestamp"].max() <= s_next["timestamp"].min()


def test_no_duplicate_pairs_across_splits() -> None:
    """Check that no (userId, movieId) pair exists across multiple splits."""
    rows = [
        {"userId": i % 5 + 1, "movieId": (i * 7) % 20 + 1, "rating": 3.5, "timestamp": 1000 + i}
        for i in range(50)
    ]
    df = clean_ratings(pd.DataFrame(rows), movie_ids=set(range(1, 30)))
    fractions = {"train": 0.70, "val": 0.10, "test": 0.10, "stream": 0.10}
    splits = split_by_time(df, fractions)

    seen_pairs: set[tuple[int, int]] = set()
    for s_name, s_df in splits.items():
        pairs = set(zip(s_df["userId"], s_df["movieId"]))
        # No duplicates within or across splits
        assert len(pairs) == len(s_df), f"Duplicate within split {s_name}"
        assert seen_pairs.isdisjoint(pairs), f"Overlap found in split {s_name}"
        seen_pairs.update(pairs)


def test_real_processed_files() -> None:
    """Real-files integration test skipped when data/processed/train.parquet is missing."""
    train_path = Path("data/processed/train.parquet")
    if not train_path.exists():
        pytest.skip("data/processed/train.parquet does not exist yet")

    splits = {}
    for name in ["train", "val", "test", "stream"]:
        p = Path(f"data/processed/{name}.parquet")
        assert p.exists()
        splits[name] = pd.read_parquet(p)

    # 1. No duplicate pairs across splits
    seen_pairs: set[tuple[int, int]] = set()
    for s_name, s_df in splits.items():
        pairs = set(zip(s_df["userId"], s_df["movieId"]))
        assert len(pairs) == len(s_df), f"Duplicates in {s_name}"
        assert seen_pairs.isdisjoint(pairs), f"Overlap in {s_name}"
        seen_pairs.update(pairs)

    # 2. Time order across splits
    split_order = ["train", "val", "test", "stream"]
    for i in range(len(split_order) - 1):
        s_curr = splits[split_order[i]]
        s_next = splits[split_order[i + 1]]
        assert s_curr["timestamp"].max() <= s_next["timestamp"].min()

    # 3. Ratings between 0.5 and 5.0
    for s_name, s_df in splits.items():
        assert (s_df["rating"] >= 0.5).all()
        assert (s_df["rating"] <= 5.0).all()
