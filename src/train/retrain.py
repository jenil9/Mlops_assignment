"""Retraining pipeline with automatic trigger, quality gate, and promotion logic."""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common.config import get_settings, load_params
from src.common.db import DriftReport, Feedback, RetrainRun, get_engine, get_session, init_db
from src.common.reload import reload_replicas
from src.recsys.als import train_als
from src.recsys.bundle import build_bundle, save_bundle
from src.recsys.evaluate import evaluate
from src.recsys.registry import alias_version, fetch_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------


def should_trigger(
    latest_drift_row: Any | None,
    champion_version: int,
    new_rows: int,
    params: dict,
) -> bool:
    """Return True if an automatic retrain should be triggered.

    Trigger only when:
    - latest_drift_row is for THIS champion version
    - drift_detected is True
    - new_rows >= retrain.min_new_rows (stops endless retraining after rejection)
    """
    if latest_drift_row is None:
        return False
    if latest_drift_row.model_version != champion_version:
        return False
    if not latest_drift_row.drift_detected:
        return False
    retrain_cfg = params.get("retrain", {})
    min_new_rows = retrain_cfg.get("min_new_rows", 500)
    return new_rows >= min_new_rows


def merge_training_data(train_df: pd.DataFrame, feedback_df: pd.DataFrame) -> pd.DataFrame:
    """Combine train.parquet and feedback rows; latest rating per (user, movie) wins."""
    fb_cols = (
        feedback_df[["userId", "movieId", "rating", "timestamp"]]
        if not feedback_df.empty
        else pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    )
    combined = pd.concat(
        [
            train_df[["userId", "movieId", "rating", "timestamp"]],
            fb_cols,
        ],
        ignore_index=True,
    )
    # Keep the latest rating per (userId, movieId)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )
    return combined


def split_holdout(feedback_df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: oldest (1-fraction) as training, newest fraction as holdout."""
    if feedback_df.empty:
        return feedback_df.copy(), feedback_df.copy()
    sorted_df = feedback_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_df)
    n_holdout = max(1, int(n * fraction))
    n_train = n - n_holdout
    older = sorted_df.iloc[:n_train].copy()
    holdout = sorted_df.iloc[n_train:].copy()
    return older, holdout


def decide_promotion(
    v1_holdout: float,
    v2_holdout: float,
    v1_test: float,
    v2_test: float,
    params: dict,
) -> tuple[bool, str]:
    """Returns (promote, reason) based on NDCG@K values.

    Promote if:
    - (v2_holdout - v1_holdout) / max(v1_holdout, 1e-9) >= min_improvement
    - AND (v1_test - v2_test) / max(v1_test, 1e-9) <= max_test_drop
    """
    retrain_cfg = params.get("retrain", {})
    min_improvement = retrain_cfg.get("min_improvement", 0.01)
    max_test_drop = retrain_cfg.get("max_test_drop", 0.10)

    holdout_gain = (v2_holdout - v1_holdout) / max(v1_holdout, 1e-9)
    test_drop = (v1_test - v2_test) / max(v1_test, 1e-9)

    if holdout_gain < min_improvement:
        return False, f"holdout gain {holdout_gain:.4f} < {min_improvement}"
    if test_drop > max_test_drop:
        return False, f"test drop {test_drop:.4f} > {max_test_drop}"
    return True, f"holdout gain {holdout_gain:.4f}, test drop {test_drop:.4f}"


# ---------------------------------------------------------------------------
# Core retrain step
# ---------------------------------------------------------------------------


def run_retrain(trigger: str, settings: Any, params: dict, force: bool = False) -> None:
    """Execute one retrain cycle and update the database with results."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    engine = get_engine(settings.database_url)
    init_db(engine)
    db = next(get_session(engine))

    retrain_run_row: RetrainRun | None = None

    try:
        # Step 1: Check for already-running retrain
        running = db.query(RetrainRun).filter(RetrainRun.status == "running").all()
        for r in running:
            age = (datetime.now(timezone.utc) - r.started_at).total_seconds()
            if age < 30 * 60:
                log.info(
                    "Another retrain is already running (started %.0f min ago). Skipping.", age / 60
                )
                return

        # Insert retrain row
        retrain_run_row = RetrainRun(
            trigger=trigger,
            status="running",
        )
        db.add(retrain_run_row)
        db.commit()
        db.refresh(retrain_run_row)

        # Step 2: Load champion bundle (v1)
        champ_ver = alias_version(settings, "champion")
        if champ_ver is None:
            log.info("No champion model found. Skipping retrain.")
            retrain_run_row.status = "failed"
            retrain_run_row.note = "No champion model"
            retrain_run_row.finished_at = datetime.now(timezone.utc)
            db.commit()
            return

        v1_bundle = fetch_bundle(settings, champ_ver)

        # Step 3: Read all feedback rows
        feedback_rows_db = db.query(Feedback).all()
        if not feedback_rows_db:
            log.info("No feedback rows. Skipping retrain.")
            retrain_run_row.status = "done"
            retrain_run_row.decision = "rejected"
            retrain_run_row.note = "not enough feedback"
            retrain_run_row.finished_at = datetime.now(timezone.utc)
            db.commit()
            return

        feedback_data = []
        for fb in feedback_rows_db:
            feedback_data.append(
                {
                    "userId": fb.user_id,
                    "movieId": fb.movie_id,
                    "rating": fb.rating,
                    "timestamp": int(fb.created_at.timestamp()),
                }
            )
        all_feedback_df = pd.DataFrame(feedback_data)
        # Latest rating per (user, movie)
        all_feedback_df = (
            all_feedback_df.sort_values("timestamp")
            .drop_duplicates(subset=["userId", "movieId"], keep="last")
            .reset_index(drop=True)
        )

        # Step 4: Split holdout
        retrain_cfg = params.get("retrain", {})
        holdout_fraction = retrain_cfg.get("holdout_fraction", 0.2)
        eval_cfg = params.get("eval", {})
        relevant_threshold = float(eval_cfg.get("relevant_threshold", 4.0))

        older_feedback, holdout_df = split_holdout(all_feedback_df, holdout_fraction)

        has_relevant = (holdout_df["rating"] >= relevant_threshold).any()
        if not has_relevant:
            log.info("Holdout has no relevant ratings. Rejecting.")
            retrain_run_row.status = "done"
            retrain_run_row.decision = "rejected"
            retrain_run_row.note = "not enough feedback"
            retrain_run_row.holdout_rows = len(holdout_df)
            retrain_run_row.finished_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Step 5: Train v2
        processed_dir = Path(params["data"]["processed_dir"])
        train_df = pd.read_parquet(processed_dir / "train.parquet")

        training_df = merge_training_data(train_df, older_feedback)

        model_cfg = params.get("model", {})
        log.info(
            "Training v2 model on %d rows (%d feedback rows)...",
            len(training_df),
            len(older_feedback),
        )
        model = train_als(
            ratings=training_df,
            factors=int(model_cfg.get("factors", 32)),
            reg=float(model_cfg.get("reg", 0.1)),
            iterations=int(model_cfg.get("iterations", 15)),
            seed=int(model_cfg.get("seed", 42)),
        )

        movies_df = pd.read_parquet(processed_dir / "movies.parquet")
        v2_bundle = build_bundle(
            model=model,
            ratings=training_df,
            movies=movies_df,
            params=params,
            feedback_rows=len(older_feedback),
            parent_version=champ_ver,
        )

        # Step 6: Evaluate v1 and v2
        k = int(eval_cfg.get("k", 10))
        min_item_ratings = int(model_cfg.get("min_item_ratings", 10))
        test_df = pd.read_parquet(processed_dir / "test.parquet")

        v2_holdout_metrics = evaluate(
            v2_bundle, holdout_df, training_df, k, relevant_threshold, min_item_ratings
        )
        v1_holdout_metrics = evaluate(
            v1_bundle, holdout_df, train_df, k, relevant_threshold, min_item_ratings
        )

        v2_test_metrics = evaluate(
            v2_bundle, test_df, train_df, k, relevant_threshold, min_item_ratings
        )
        v1_test_metrics = evaluate(
            v1_bundle, test_df, train_df, k, relevant_threshold, min_item_ratings
        )

        v1_holdout_ndcg = v1_holdout_metrics["ndcg_at_k"]
        v2_holdout_ndcg = v2_holdout_metrics["ndcg_at_k"]
        v1_test_ndcg = v1_test_metrics["ndcg_at_k"]
        v2_test_ndcg = v2_test_metrics["ndcg_at_k"]

        # Step 7: Decide promotion
        if force:
            promote = True
            reason = "forced"
            decision = "forced"
        else:
            promote, reason = decide_promotion(
                v1_holdout_ndcg, v2_holdout_ndcg, v1_test_ndcg, v2_test_ndcg, params
            )
            decision = "promoted" if promote else "rejected"

        # Step 8: Log to MLflow
        mlflow.set_experiment("movie-recsys")
        with mlflow.start_run(tags={"stage": "retrain", "trigger": trigger}) as mlflow_run:
            mlflow.log_params(model_cfg)
            mlflow.log_metrics(
                {
                    "v1_holdout_ndcg_at_10": v1_holdout_ndcg,
                    "v2_holdout_ndcg_at_10": v2_holdout_ndcg,
                    "v1_test_ndcg_at_10": v1_test_ndcg,
                    "v2_test_ndcg_at_10": v2_test_ndcg,
                    "v1_holdout_rmse": v1_holdout_metrics["rmse"],
                    "v2_holdout_rmse": v2_holdout_metrics["rmse"],
                    "v1_test_rmse": v1_test_metrics["rmse"],
                    "v2_test_rmse": v2_test_metrics["rmse"],
                }
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                save_bundle(v2_bundle, tmpdir)
                mlflow.log_artifacts(tmpdir, artifact_path="bundle")

            # Save feedback snapshot
            with tempfile.TemporaryDirectory() as snapdir:
                snap_path = Path(snapdir) / "feedback_snapshot.parquet"
                all_feedback_df.to_parquet(snap_path, index=False)
                mlflow.log_artifact(str(snap_path))

            mlflow.log_artifact("params.yaml")

            run_id_mlflow = mlflow_run.info.run_id

        # Register v2
        model_uri = f"runs:/{run_id_mlflow}/bundle"
        reg = mlflow.register_model(model_uri, settings.model_name)
        v2_version = int(reg.version)

        if promote:
            client.set_registered_model_alias(settings.model_name, "previous", str(champ_ver))
            client.set_registered_model_alias(settings.model_name, "champion", str(v2_version))
            log.info("Promoted v%d -> champion (v%d -> previous)", v2_version, champ_ver)
            results = reload_replicas()
            for target, success, ver in results:
                log.info("Reload %s: %s (version=%s)", target, "ok" if success else "failed", ver)
        else:
            client.set_model_version_tag(
                settings.model_name, str(v2_version), "decision", "rejected"
            )
            log.info("v2 rejected: %s", reason)

        # Step 10: Update retrain row
        retrain_run_row.finished_at = datetime.now(timezone.utc)
        retrain_run_row.status = "done"
        retrain_run_row.rows_used = len(training_df)
        retrain_run_row.holdout_rows = len(holdout_df)
        retrain_run_row.v1_version = champ_ver
        retrain_run_row.v2_version = v2_version
        retrain_run_row.v1_ndcg_holdout = v1_holdout_ndcg
        retrain_run_row.v2_ndcg_holdout = v2_holdout_ndcg
        retrain_run_row.v1_ndcg_test = v1_test_ndcg
        retrain_run_row.v2_ndcg_test = v2_test_ndcg
        retrain_run_row.decision = decision
        retrain_run_row.note = reason
        db.commit()

        log.info(
            "Retrain complete: decision=%s v1_ndcg=%.4f v2_ndcg=%.4f",
            decision,
            v1_holdout_ndcg,
            v2_holdout_ndcg,
        )

    except Exception as e:
        log.error("Retrain failed: %s", e, exc_info=True)
        if retrain_run_row is not None:
            retrain_run_row.status = "failed"
            retrain_run_row.note = str(e)
            retrain_run_row.finished_at = datetime.now(timezone.utc)
            try:
                db.commit()
            except Exception:
                pass
    finally:
        db.close()


def _count_new_rows(settings: Any, params: dict, champion_version: int) -> int:
    """Count feedback rows created after the last finished retrain run.

    Falls back to the champion's created_at when no finished run is available.
    """
    engine = get_engine(settings.database_url)
    db = next(get_session(engine))
    try:
        from src.recsys.registry import fetch_bundle

        last_run = (
            db.query(RetrainRun)
            .filter(RetrainRun.status.in_(["done", "failed"]))
            .order_by(RetrainRun.finished_at.desc())
            .first()
        )

        if last_run and last_run.finished_at:
            cutoff = last_run.finished_at
        else:
            # Use champion's created_at
            bundle = fetch_bundle(settings, champion_version)
            cutoff_str = bundle.meta.get("created_at", "")
            try:
                cutoff = datetime.fromisoformat(cutoff_str)
            except Exception:
                cutoff = datetime.fromtimestamp(0, tz=timezone.utc)

        count = db.query(Feedback).filter(Feedback.created_at > cutoff).count()
        return int(count)
    finally:
        db.close()


def main() -> None:
    """Entry point with --loop, --auto, --once, --force modes."""
    parser = argparse.ArgumentParser(description="Retraining pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--loop", action="store_true", help="Run auto check on a loop (Compose default)"
    )
    group.add_argument(
        "--auto", action="store_true", help="One pass with trigger rules (k8s CronJob)"
    )
    group.add_argument("--once", action="store_true", help="One retrain now, skip trigger rules")
    group.add_argument(
        "--force", action="store_true", help="One retrain, skip trigger rules AND quality gate"
    )
    args = parser.parse_args()

    settings = get_settings()
    params = load_params()
    retrain_cfg = params.get("retrain", {})
    interval = retrain_cfg.get("interval_seconds", 60)

    engine = get_engine(settings.database_url)
    init_db(engine)

    def do_auto() -> None:
        """Check trigger conditions and retrain if needed."""
        champ_ver = alias_version(settings, "champion")
        if champ_ver is None:
            log.info("No champion model yet. Waiting ...")
            return

        db = next(get_session(engine))
        try:
            latest_drift = db.query(DriftReport).order_by(DriftReport.created_at.desc()).first()
        finally:
            db.close()

        new_rows = _count_new_rows(settings, params, champ_ver)
        triggered = should_trigger(latest_drift, champ_ver, new_rows, params)
        if triggered:
            log.info("Trigger condition met: drift detected, %d new rows. Retraining ...", new_rows)
            run_retrain("drift", settings, params, force=False)
        else:
            log.info(
                "No trigger: drift=%s, new_rows=%d",
                latest_drift.drift_detected if latest_drift else "N/A",
                new_rows,
            )

    if args.force:
        run_retrain("forced", settings, params, force=True)
    elif args.once:
        run_retrain("manual", settings, params, force=False)
    elif args.auto:
        do_auto()
    else:
        # --loop (default)
        log.info("Starting retrainer loop (interval=%ds)", interval)
        while True:
            try:
                do_auto()
            except Exception as e:
                log.error("Error in retrain loop: %s", e, exc_info=True)
            time.sleep(interval)


if __name__ == "__main__":
    main()
