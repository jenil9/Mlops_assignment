"""Evidently-based data drift detection service."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def build_current(window: pd.DataFrame, bundle: Any) -> pd.DataFrame:
    """Map feedback rows to the three drift columns (rating, main_genre, item_popularity).

    Args:
        window: DataFrame with columns user_id, movie_id, rating, created_at
        bundle: loaded Bundle with movies and popularity data

    Returns:
        DataFrame with columns rating, main_genre, item_popularity
    """
    # Build lookup maps from the bundle
    pop_map: dict[int, int] = dict(
        zip(bundle.popularity["movieId"], bundle.popularity["n_ratings"])
    )
    genre_map: dict[int, str] = {}
    for _, row in bundle.movies.iterrows():
        mid = int(row["movieId"])
        genres = row["genres"]
        if isinstance(genres, (list,)) and len(genres) > 0 and genres[0]:
            genre_map[mid] = str(genres[0])
        else:
            genre_map[mid] = "none"

    rows = []
    for _, r in window.iterrows():
        mid = int(r["movie_id"])
        rows.append(
            {
                "rating": float(r["rating"]),
                "main_genre": genre_map.get(mid, "none"),
                "item_popularity": int(pop_map.get(mid, 0)),
            }
        )
    return pd.DataFrame(rows)


def decide(share: float, n_rows: int, params: dict) -> bool:
    """Return True if drift is detected based on share and row count.

    Args:
        share: fraction of drifted columns (0, 0.33, 0.67, or 1.0)
        n_rows: number of rows in the current window
        params: drift section of params.yaml

    Returns:
        True if drift is detected
    """
    drift_cfg = params.get("drift", params)
    min_rows = drift_cfg.get("min_rows", 500)
    share_threshold = drift_cfg.get("share_threshold", 0.3)
    return (n_rows >= min_rows) and (share >= share_threshold)


def live_hit_rate(window: pd.DataFrame, logs: pd.DataFrame) -> float | None:
    """Compute fraction of liked feedback movies that were recommended.

    Among window feedback rows with rating >= serving.liked_threshold that have
    an earlier /recommend log row for the same user, the share whose movie_id
    was in that recommendation list.

    Returns None if there are no qualifying rows.
    """
    if window.empty or logs.empty:
        return None

    hits = 0
    total = 0
    liked = window[window["rating"] >= 4.0]

    for _, fb_row in liked.iterrows():
        uid = int(fb_row["user_id"])
        mid = int(fb_row["movie_id"])
        fb_time = fb_row["created_at"]

        # Find the most recent recommend log for this user BEFORE this feedback
        user_logs = logs[
            (logs["user_id"] == uid)
            & (logs["kind"].isin(["popular", "personalized"]))
            & (logs["created_at"] < fb_time)
        ]
        if user_logs.empty:
            continue

        most_recent = user_logs.sort_values("created_at").iloc[-1]
        try:
            rec_ids = set(json.loads(most_recent["movie_ids"]))
        except Exception:
            continue

        total += 1
        if mid in rec_ids:
            hits += 1

    if total == 0:
        return None
    return hits / total


def run_check(
    engine: Any,
    session_factory: Any,
    settings: Any,
    params: dict,
    gauges: dict,
    reports_dir: Path,
    bundle_cache: dict,
) -> None:
    """Perform one full drift check cycle."""
    from sqlalchemy.orm import sessionmaker

    from src.common.db import DriftReport, Feedback, RecommendationLog
    from src.recsys.registry import alias_version, fetch_bundle

    drift_cfg = params.get("drift", {})
    min_rows = drift_cfg.get("min_rows", 500)
    reference_sample = drift_cfg.get("reference_sample", 10000)
    keep_reports = drift_cfg.get("keep_reports", 20)

    # Step 1: Find champion
    champ_ver = alias_version(settings, "champion")
    if champ_ver is None:
        log.info("No champion model found. Skipping drift check.")
        return

    # Step 2: Load bundle (cache by version)
    if bundle_cache.get("version") != champ_ver:
        log.info("Loading bundle for champion v%s", champ_ver)
        bundle = fetch_bundle(settings, champ_ver)
        bundle_cache["bundle"] = bundle
        bundle_cache["version"] = champ_ver
    else:
        bundle = bundle_cache["bundle"]

    # Step 3: Select window
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        from datetime import datetime, timezone

        created_at_str = bundle.meta.get("created_at", "")
        try:
            model_created_at = datetime.fromisoformat(created_at_str)
        except Exception:
            model_created_at = datetime.fromtimestamp(0, tz=timezone.utc)

        feedback_rows = db.query(Feedback).filter(Feedback.created_at > model_created_at).all()
        window_data = [
            {
                "user_id": fb.user_id,
                "movie_id": fb.movie_id,
                "rating": fb.rating,
                "source": fb.source,
                "created_at": fb.created_at,
            }
            for fb in feedback_rows
        ]
        window = pd.DataFrame(window_data)

        n_rows = len(window)
        rows_by_source: dict[str, int] = {}
        if not window.empty:
            rows_by_source = window.groupby("source").size().to_dict()

        # Step 4: Insufficient rows case
        if n_rows < min_rows:
            log.info("Insufficient rows for drift check: %d < %d", n_rows, min_rows)
            report = DriftReport(
                model_version=champ_ver,
                n_rows=n_rows,
                rows_by_source=json.dumps(rows_by_source),
                drift_share=None,
                drift_detected=False,
                live_hit_rate=None,
                report_path=None,
                note="insufficient rows",
            )
            db.add(report)
            db.commit()

            # Update gauges
            if gauges:
                gauges["drift_share"].set(0)
                gauges["drift_detected"].set(0)
                gauges["drift_window_rows"].set(n_rows)
                for src, cnt in rows_by_source.items():
                    gauges["drift_rows_by_source"].labels(source=src).set(cnt)
                gauges["drift_last_check_timestamp"].set(time.time())
            return

        # Step 5: Run Evidently
        from evidently.metric_preset import DataDriftPreset
        from evidently.pipeline.column_mapping import ColumnMapping
        from evidently.report import Report

        current_df = build_current(window, bundle)

        ref_df = bundle.reference.copy()
        if len(ref_df) > reference_sample:
            ref_df = ref_df.sample(n=reference_sample, random_state=42)

        column_mapping = ColumnMapping(
            numerical_features=["rating", "item_popularity"],
            categorical_features=["main_genre"],
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        report_dict = report.as_dict()

        # Extract drifted column count from Evidently 0.4.x structure
        try:
            metrics_list = report_dict.get("metrics", [])
            n_drifted = 0
            for m in metrics_list:
                result = m.get("result", {})
                if "number_of_drifted_columns" in result:
                    n_drifted = int(result["number_of_drifted_columns"])
                    break
                # fallback: look for per-column drift
                drift_by_columns = result.get("drift_by_columns", {})
                if drift_by_columns:
                    n_drifted = sum(
                        1
                        for col_data in drift_by_columns.values()
                        if col_data.get("drift_detected", False)
                    )
                    break
        except Exception as e:
            log.warning("Error parsing Evidently report: %s", e)
            n_drifted = 0

        share = n_drifted / 3
        drift_detected_flag = decide(share, n_rows, params)

        # Step 6: Save reports
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        report_stem = f"drift_{ts}_v{champ_ver}"
        reports_dir.mkdir(parents=True, exist_ok=True)
        html_path = reports_dir / f"{report_stem}.html"
        json_path = reports_dir / f"{report_stem}.json"

        try:
            report.save_html(str(html_path))
        except Exception as e:
            log.warning("Could not save HTML report: %s", e)
            html_path = None

        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": ts,
                    "champion_version": champ_ver,
                    "n_rows": n_rows,
                    "n_drifted": n_drifted,
                    "drift_share": share,
                    "drift_detected": drift_detected_flag,
                },
                f,
                indent=2,
            )

        # Copy latest HTML
        if html_path and html_path.exists():
            shutil.copy(html_path, reports_dir / "latest.html")

        # Prune old reports
        all_html = sorted(reports_dir.glob("drift_*.html"), key=lambda p: p.name)
        for old in all_html[:-keep_reports]:
            old.unlink(missing_ok=True)
        all_json = sorted(reports_dir.glob("drift_*.json"), key=lambda p: p.name)
        for old in all_json[:-keep_reports]:
            old.unlink(missing_ok=True)

        # Step 7: Compute live hit rate
        log_rows = db.query(RecommendationLog).all()
        logs_data = [
            {
                "user_id": lr.user_id,
                "kind": lr.kind,
                "movie_ids": lr.movie_ids,
                "created_at": lr.created_at,
            }
            for lr in log_rows
        ]
        logs_df = pd.DataFrame(logs_data) if logs_data else pd.DataFrame()

        hit_rate = live_hit_rate(window, logs_df)

        drift_db_report = DriftReport(
            model_version=champ_ver,
            n_rows=n_rows,
            rows_by_source=json.dumps(rows_by_source),
            drift_share=share,
            drift_detected=drift_detected_flag,
            live_hit_rate=hit_rate,
            report_path=str(html_path) if html_path else None,
            note=None,
        )
        db.add(drift_db_report)
        db.commit()

        log.info(
            "Drift check complete: v%d, rows=%d, share=%.2f, detected=%s",
            champ_ver,
            n_rows,
            share,
            drift_detected_flag,
        )

        # Update Prometheus gauges
        if gauges:
            gauges["drift_share"].set(share)
            gauges["drift_detected"].set(1 if drift_detected_flag else 0)
            gauges["drift_window_rows"].set(n_rows)
            if hit_rate is not None:
                gauges["live_hit_rate"].set(hit_rate)
            for src, cnt in rows_by_source.items():
                gauges["drift_rows_by_source"].labels(source=src).set(cnt)
            gauges["drift_last_check_timestamp"].set(time.time())

    except Exception as e:
        log.error("Drift check failed: %s", e, exc_info=True)
        db.rollback()
    finally:
        db.close()


def main() -> None:
    """Entry point: --once for a single check, default for continuous loop."""
    parser = argparse.ArgumentParser(description="Drift detection service")
    parser.add_argument("--once", action="store_true", help="Run one check and exit")
    args = parser.parse_args()

    from src.common.config import get_settings, load_params
    from src.common.db import get_engine, init_db

    settings = get_settings()
    params = load_params()
    drift_cfg = params.get("drift", {})
    interval = drift_cfg.get("interval_seconds", 60)

    engine = get_engine(settings.database_url)
    init_db(engine)

    reports_dir = Path("reports/drift")
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle_cache: dict = {}
    gauges: dict = {}

    if not args.once:
        # Start Prometheus metrics server
        from prometheus_client import Gauge, start_http_server

        gauges["drift_share"] = Gauge("drift_share", "Share of drifted columns")
        gauges["drift_detected"] = Gauge("drift_detected", "1 if drift is detected")
        gauges["drift_window_rows"] = Gauge("drift_window_rows", "Rows in current drift window")
        gauges["live_hit_rate"] = Gauge("live_hit_rate", "Live recommendation hit rate")
        gauges["drift_rows_by_source"] = Gauge(
            "drift_rows_by_source", "Rows by source in drift window", ["source"]
        )
        gauges["drift_last_check_timestamp"] = Gauge(
            "drift_last_check_timestamp", "Unix timestamp of last drift check"
        )
        start_http_server(8001)
        log.info("Prometheus metrics server started on port 8001")

        log.info("Starting drift detection loop (interval=%ds)", interval)
        while True:
            try:
                run_check(engine, None, settings, params, gauges, reports_dir, bundle_cache)
            except Exception as e:
                log.error("Unexpected error in drift loop: %s", e, exc_info=True)
            time.sleep(interval)
    else:
        run_check(engine, None, settings, params, gauges, reports_dir, bundle_cache)
        log.info("Single drift check complete.")


if __name__ == "__main__":
    main()
