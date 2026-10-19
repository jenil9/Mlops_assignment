"""Administrative CLI commands for model alias management and feedback cleanup."""

from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import get_settings
from src.common.db import (
    DriftReport,
    Feedback,
    RecommendationLog,
    RetrainRun,
    get_engine,
    get_session,
)
from src.common.reload import reload_replicas
from src.recsys.registry import alias_version


def cmd_check_champion() -> int:
    """Check if model has an active champion alias, exiting 1 if missing."""
    settings = get_settings()
    champ = alias_version(settings, "champion")
    if champ is not None:
        print(f"Champion model exists: version {champ}")
        return 0
    print("No champion model. Run: make train")
    return 1


def cmd_reset_feedback() -> int:
    """Delete simulator rows from feedback and logs, and clear drift and retrain tables."""
    settings = get_settings()
    engine = get_engine(settings.database_url)
    session = next(get_session(engine))
    try:
        del_fb = session.query(Feedback).filter(Feedback.source == "simulator").delete()
        del_logs = (
            session.query(RecommendationLog)
            .filter(RecommendationLog.source == "simulator")
            .delete()
        )
        del_drift = session.query(DriftReport).delete()
        del_retrain = session.query(RetrainRun).delete()
        session.commit()
        print(
            f"Reset feedback complete: removed {del_fb} simulator feedback rows, "
            f"{del_logs} logs, {del_drift} drift reports, {del_retrain} retrain runs."
        )
        return 0
    except Exception as e:
        session.rollback()
        print(f"Error resetting feedback: {e}", file=sys.stderr)
        return 1
    finally:
        session.close()


def cmd_rollback() -> int:
    """Roll back champion model to previous version and notify replicas."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    prev_ver = alias_version(settings, "previous")
    if prev_ver is None:
        print("Error: No 'previous' model alias found. Cannot roll back.", file=sys.stderr)
        return 1

    champ_ver = alias_version(settings, "champion")

    # Swap aliases: champion goes to previous, previous goes to old champion
    client.set_registered_model_alias(settings.model_name, "champion", str(prev_ver))
    if champ_ver is not None:
        client.set_registered_model_alias(settings.model_name, "previous", str(champ_ver))

    print(f"Rolled back champion to version {prev_ver} (previous alias set to {champ_ver})")
    results = reload_replicas()
    for target, success, ver in results:
        status_text = f"success (version {ver})" if success else "failed"
        print(f"Reload {target}: {status_text}")
    return 0


def cmd_set_champion(version: int) -> int:
    """Set champion alias to specified version and notify replicas."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    client.set_registered_model_alias(settings.model_name, "champion", str(version))
    print(f"Champion alias set to version {version}")
    results = reload_replicas()
    for target, success, ver in results:
        status_text = f"success (version {ver})" if success else "failed"
        print(f"Reload {target}: {status_text}")
    return 0


def cmd_reload() -> int:
    """Trigger reload on all registered API replicas."""
    results = reload_replicas()
    for target, success, ver in results:
        status_text = f"success (version {ver})" if success else "failed"
        print(f"Reload {target}: {status_text}")
    return 0


def main() -> None:
    """Parse and dispatch admin CLI subcommands."""
    parser = argparse.ArgumentParser(description="Model administration and database management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-champion", help="Verify champion model exists")
    subparsers.add_parser("reset-feedback", help="Clear simulator feedback and drift history")
    subparsers.add_parser("rollback", help="Roll back champion to previous alias")
    subparsers.add_parser("reload", help="Trigger reload across replicas")

    set_champ_p = subparsers.add_parser("set-champion", help="Point champion to version")
    set_champ_p.add_argument("--version", type=int, required=True, help="Model version number")

    args = parser.parse_args()

    if args.command == "check-champion":
        sys.exit(cmd_check_champion())
    elif args.command == "reset-feedback":
        sys.exit(cmd_reset_feedback())
    elif args.command == "rollback":
        sys.exit(cmd_rollback())
    elif args.command == "set-champion":
        sys.exit(cmd_set_champion(args.version))
    elif args.command == "reload":
        sys.exit(cmd_reload())


if __name__ == "__main__":
    main()
