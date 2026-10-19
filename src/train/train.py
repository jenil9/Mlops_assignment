"""Initial model training script with evaluation and MLflow registry integration."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common.config import get_settings, load_params
from src.recsys.als import train_als
from src.recsys.bundle import build_bundle, save_bundle
from src.recsys.evaluate import evaluate, evaluate_popularity
from src.recsys.registry import alias_version


def get_git_commit() -> str:
    """Return current git commit sha or 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Train ALS baseline and registered model, evaluating and logging to MLflow."""
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip training if champion model alias exists in MLflow",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote trained model as champion even if champion exists",
    )
    args = parser.parse_args()

    settings = get_settings()
    params = load_params()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if args.if_missing:
        champ = alias_version(settings, "champion")
        if champ is not None:
            print(f"Champion model already exists (version {champ}). Exiting.")
            return

    processed_dir = Path(params["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    movies_df = pd.read_parquet(processed_dir / "movies.parquet")

    model_cfg = params["model"]
    eval_cfg = params["eval"]

    print("Training ALS matrix factorization model...")
    model = train_als(
        ratings=train_df,
        factors=int(model_cfg["factors"]),
        reg=float(model_cfg["reg"]),
        iterations=int(model_cfg["iterations"]),
        seed=int(model_cfg["seed"]),
    )

    k = int(eval_cfg["k"])
    threshold = float(eval_cfg["relevant_threshold"])
    min_ratings = int(model_cfg["min_item_ratings"])

    print("Evaluating models on validation and test sets...")
    bundle_temp = build_bundle(
        model=model,
        ratings=train_df,
        movies=movies_df,
        params=params,
        feedback_rows=0,
        parent_version=None,
    )

    val_metrics = evaluate(bundle_temp, val_df, train_df, k, threshold, min_ratings)
    test_metrics = evaluate(bundle_temp, test_df, train_df, k, threshold, min_ratings)

    base_val = evaluate_popularity(bundle_temp, val_df, train_df, k, threshold)
    base_test = evaluate_popularity(bundle_temp, test_df, train_df, k, threshold)

    metrics = {
        "val_rmse": val_metrics["rmse"],
        f"val_precision_at_{k}": val_metrics["precision_at_k"],
        f"val_ndcg_at_{k}": val_metrics["ndcg_at_k"],
        "test_rmse": test_metrics["rmse"],
        f"test_precision_at_{k}": test_metrics["precision_at_k"],
        f"test_ndcg_at_{k}": test_metrics["ndcg_at_k"],
        "baseline_val_rmse": base_val["rmse"],
        f"baseline_val_precision_at_{k}": base_val["precision_at_k"],
        f"baseline_val_ndcg_at_{k}": base_val["ndcg_at_k"],
        "baseline_test_rmse": base_test["rmse"],
        f"baseline_test_precision_at_{k}": base_test["precision_at_k"],
        f"baseline_test_ndcg_at_{k}": base_test["ndcg_at_k"],
    }

    # Final bundle with metrics included
    bundle = build_bundle(
        model=model,
        ratings=train_df,
        movies=movies_df,
        params=params,
        feedback_rows=0,
        parent_version=None,
        metrics=metrics,
    )

    experiment = mlflow.set_experiment("movie-recsys")
    print(
        f"Logging run to MLflow experiment '{experiment.name}' at {settings.mlflow_tracking_uri}..."
    )

    with mlflow.start_run() as run:
        mlflow.log_params(model_cfg)
        mlflow.log_params({"eval_k": k, "eval_threshold": threshold})
        mlflow.log_metrics(metrics)

        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("stage", "initial")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_bundle(bundle, tmpdir)
            mlflow.log_artifacts(tmpdir, artifact_path="bundle")

        mlflow.log_artifact("params.yaml")
        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock")

        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_uri = f"runs:/{run_id}/bundle"
    reg_model = mlflow.register_model(model_uri, settings.model_name)
    version = int(reg_model.version)

    current_champ = alias_version(settings, "champion")
    if current_champ is None:
        client.set_registered_model_alias(settings.model_name, "champion", str(version))
        print(f"Registered version {version} and set alias 'champion'.")
    elif args.promote:
        client.set_registered_model_alias(settings.model_name, "champion", str(version))
        print(f"Registered version {version} and promoted to alias 'champion'.")
    else:
        print(f"Registered version {version} (not promoted, champion is version {current_champ}).")

    print("\n=== Model Training Summary ===")
    print(f"Model Name:      {settings.model_name} (v{version})")
    print(f"Val RMSE:        {val_metrics['rmse']:.4f} (Baseline: {base_val['rmse']:.4f})")
    print(
        f"Val NDCG@{k}:     {val_metrics['ndcg_at_k']:.4f} (Baseline: {base_val['ndcg_at_k']:.4f})"
    )
    print(f"Test RMSE:       {test_metrics['rmse']:.4f} (Baseline: {base_test['rmse']:.4f})")
    test_ndcg = test_metrics["ndcg_at_k"]
    base_test_ndcg = base_test["ndcg_at_k"]
    print(f"Test NDCG@{k}:    {test_ndcg:.4f} (Baseline: {base_test_ndcg:.4f})")


if __name__ == "__main__":
    main()
