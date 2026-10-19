"""Traffic simulator for live ratings and drift simulation."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def wait_for_api(api_url: str, timeout: int = 60) -> bool:
    """Poll GET /health until 200 or timeout. Returns True if healthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def retry_request(fn, retries: int = 3, delay: float = 0.5):
    """Retry fn() up to `retries` times with `delay` between attempts."""
    for attempt in range(retries):
        try:
            result = fn()
            return result, None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, e
    return None, Exception("Max retries exceeded")


def simulate(
    mode: str,
    rate: float,
    limit: int | None,
    genre: str,
    drift_fraction: float,
    drift_pool_size: int,
    api_url: str,
    simulator_key: str,
    data_dir: str = "data/processed",
) -> int:
    """Run the simulator. Returns 0 on success, 1 if >5% errors."""
    data_path = Path(data_dir)

    # Load data
    stream_df = pd.read_parquet(data_path / "stream.parquet")
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)
    movies_df = pd.read_parquet(data_path / "movies.parquet")
    train_df = pd.read_parquet(data_path / "train.parquet")

    if limit is not None:
        stream_df = stream_df.head(limit)

    # Build drift pool: top N most-rated movies of the chosen genre by count in train
    drift_pool: list[int] = []
    if mode == "drift":
        genre_movies = movies_df[movies_df["genres"].str.contains(genre, na=False)][
            "movieId"
        ].tolist()
        genre_set = set(genre_movies)
        counts = train_df[train_df["movieId"].isin(genre_set)]["movieId"].value_counts()
        drift_pool = counts.head(drift_pool_size).index.tolist()
        rng = random.Random(42)
    else:
        rng = random.Random(42)

    # Wait for API
    print(f"Waiting for API at {api_url} ...")
    if not wait_for_api(api_url, timeout=60):
        print(f"ERROR: API at {api_url} did not become healthy within 60 seconds.", file=sys.stderr)
        sys.exit(1)
    print("API is up. Starting simulation ...")

    headers = {"X-Simulator-Key": simulator_key}
    events_sent = 0
    recommend_calls = 0
    feedback_calls = 0
    errors = 0
    served_by: dict[str, int] = defaultdict(int)

    start_time = time.monotonic()
    interval = 1.0 / rate  # seconds per event
    next_event_time = time.monotonic()

    events = stream_df.to_dict("records")
    total = len(events)

    for i, event in enumerate(events):
        user_id = int(event["userId"])
        movie_id = int(event["movieId"])
        rating = float(event["rating"])

        # Drift substitution
        if mode == "drift" and drift_pool and rng.random() < drift_fraction:
            movie_id = rng.choice(drift_pool)
            rating = rng.choice([4.0, 4.5, 5.0])

        # GET /recommend
        def do_recommend():
            return requests.get(
                f"{api_url}/recommend",
                params={"n": 10, "user_id": user_id},
                headers=headers,
                timeout=10,
            )

        rec_resp, rec_err = retry_request(do_recommend)
        recommend_calls += 1
        if rec_err or rec_resp is None or rec_resp.status_code != 200:
            errors += 1
        else:
            xsb = rec_resp.headers.get("X-Served-By", "unknown")
            served_by[xsb] += 1

        # POST /feedback
        def do_feedback():
            return requests.post(
                f"{api_url}/feedback",
                json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
                headers=headers,
                timeout=10,
            )

        fb_resp, fb_err = retry_request(do_feedback)
        feedback_calls += 1
        if fb_err or fb_resp is None or fb_resp.status_code not in (200, 201):
            errors += 1

        events_sent += 1

        # Rate limiting using monotonic clock
        next_event_time += interval
        sleep_time = next_event_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.monotonic() - start_time
            print(
                f"  {i+1}/{total} events  errors={errors}  "
                f"elapsed={elapsed:.1f}s  rate={events_sent/elapsed:.1f} ev/s",
                flush=True,
            )

    elapsed = time.monotonic() - start_time
    achieved_rate = events_sent / elapsed if elapsed > 0 else 0
    total_requests = recommend_calls + feedback_calls
    error_rate = errors / total_requests if total_requests > 0 else 0

    print("\n=== Simulation Summary ===")
    print(f"Events sent:       {events_sent}")
    print(f"Recommend calls:   {recommend_calls}")
    print(f"Feedback calls:    {feedback_calls}")
    print(f"Errors:            {errors}")
    print(f"Elapsed (s):       {elapsed:.2f}")
    print(f"Achieved (ev/s):   {achieved_rate:.2f}")
    print("X-Served-By counts:")
    for replica, count in sorted(served_by.items()):
        print(f"  {replica}: {count}")

    if error_rate > 0.05:
        print(f"\nFAIL: error rate {error_rate:.1%} exceeds 5%", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    params = load_params()
    sim_cfg = params.get("simulator", {})

    parser = argparse.ArgumentParser(description="Traffic simulator for movie-recsys")
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--rate", type=float, default=sim_cfg.get("rate", 20))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--genre", default=sim_cfg.get("drift_genre", "Horror"))
    parser.add_argument("--drift-fraction", type=float, default=sim_cfg.get("drift_fraction", 0.8))
    args = parser.parse_args()

    api_url = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
    simulator_key = os.getenv("SIMULATOR_KEY", "sim-secret-change-me")

    exit_code = simulate(
        mode=args.mode,
        rate=args.rate,
        limit=args.limit,
        genre=args.genre,
        drift_fraction=args.drift_fraction,
        drift_pool_size=sim_cfg.get("drift_pool_size", 30),
        api_url=api_url,
        simulator_key=simulator_key,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
