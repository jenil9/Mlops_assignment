# Movie RecSys — MLOps Pipeline

A full end-to-end MLOps project: a movie recommendation system with data versioning, model training, serving, drift detection, retraining, monitoring, and Kubernetes deployment.

## What it is

An ALS matrix factorisation recommendation engine trained on MovieLens-latest-small, served via FastAPI behind Nginx, with:

- **DVC** for data pipeline versioning
- **MLflow** for experiment tracking and model registry
- **Prometheus + Grafana** for live monitoring
- **Evidently** for drift detection
- **Automatic retraining** when drift is detected
- **Docker Compose** and **Kubernetes (kind)** deployment

## Quick Start (6 commands)

```bash
make data          # download & preprocess MovieLens data
make build         # build all Docker images
make infra         # start Postgres + MLflow
make train         # train model v1 and register in MLflow
make up            # start API, Nginx, Prometheus, Grafana, drift, retrainer
```

Then open:
- **App:** http://localhost:8080
- **MLflow:** http://localhost:5000
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (dashboard loads automatically)

## Demo

```bash
make simulate LIMIT=1500        # normal traffic
make simulate-drift LIMIT=2000  # drift traffic (watch Grafana drift panel)
make retrain-force              # force retrain (or wait for automatic retrainer)
make rollback                   # roll back to version 1
make demo-reset                 # reset demo state
```

## Kubernetes

```bash
make down && make k8s-up        # deploy to local kind cluster
bash scripts/smoke_test.sh
make k8s-status
make k8s-simulate LIMIT=1500
make k8s-retrain FORCE=1
make k8s-rollout                # rolling update, zero downtime
make k8s-down
```

## Development

```bash
make test     # run all tests
make lint     # ruff check
```

## Team

| Member | Steps | Area |
|--------|-------|------|
| 1 | 1, 2 | Data and ML |
| 2 | 3, 4, 5, 6 | Serving and infra |
| 3 | 10, 11 | CI/CD and Kubernetes |
| 4 | 7, 8, 9 | Monitoring, drift, retraining |

## Dataset

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.

## GitHub Settings (one-time)

1. **Branch protection** (`main`): require PR, status checks `lint-test`, `build`, `smoke`; block direct pushes
2. **Packages**: after first `cd.yml` run, make images public under the repo's Packages tab
3. **Team workflow**: branch `feature/<step>-<name>`, open PR, green checks, review, merge
