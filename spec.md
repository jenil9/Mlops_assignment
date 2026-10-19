# PROJECT_SPEC.md: MLOps Movie Recommendation System

Spec version 1.0 (2026-10-17). Team of 4. Everything runs locally: Docker Compose first, then Kubernetes (kind). No cloud, no cost.

---

## 0. How the coding agent must use this file

1. **Build strictly in step order (Step 0 to Step 12).** Every step ends with a **CHECKPOINT**. Do not start the next step until every item of its checklist passes. If an item fails, fix it first.
2. **This file contains no code.** Write the code from the "must do" descriptions. Keep every file name, function name, endpoint, table, column, metric name, make target and environment variable **exactly as written**, because other steps depend on them.
3. **Use the pinned versions exactly.** If a pin cannot be installed, use the nearest available version in the same minor line, run `pip freeze`, update the pin, and write one line about it in `docs/DEVIATIONS.md`.
4. **Never hard-code** paths, URLs, passwords or thresholds. Settings come from `params.yaml` (thresholds, model settings) and `.env` (addresses, secrets).
5. **Simplicity rule.** Add no libraries, frameworks, endpoints or features that this file does not list. If something is not covered, choose the simplest option and note it in `docs/DEVIATIONS.md`.
6. All commands are **bash**. Windows teammates run everything inside **WSL2 (Ubuntu)**, with the project in the Linux home folder (`~/movie-recsys`), never under `/mnt/c`.
7. Code style: type hints on function signatures, a one-line docstring per module, `ruff` clean. `make lint` and `make test` must pass at every checkpoint.
8. Datetimes are stored and compared in **UTC**.

---

## 1. Project goal

The grade is on the MLOps lifecycle, not on model accuracy. The model stays simple and the system around it is production-style:

- ingest and version data (DVC)
- train, track and register models (MLflow)
- serve a full web app with accounts (FastAPI + one HTML page)
- run behind a load balancer (Nginx, two API copies)
- deploy automatically (GitHub Actions, GitHub Container Registry)
- monitor (Prometheus, Grafana) and detect drift (Evidently)
- retrain, compare and promote models with no downtime
- run the same system on Kubernetes (kind)

**Out of scope:** deep learning, cloud spending, email, password reset, HTTPS, autoscaling, Helm, Ingress controllers, poster images.

## 2. What the user sees (web app)

1. **Guest (not logged in):** sees a **Popular movies** row and a Sign up / Log in box. Guests can browse but cannot rate. Clicking a star opens the signup form.
2. **Logged in:** username and Log out button show at the top. Each movie card has 1 to 5 stars. Clicking a star saves the rating for that account and refreshes the rows.
3. After the **first rating of 4 stars or more**, a **"Because you liked X"** row appears (up to the 3 most recent liked movies, one row each).
4. After **3 ratings**, a **"Recommended for you"** row appears.
5. Movies the user has already rated never show again in any row. History is stored on the server, so another device shows the same rows.
6. **Demo helper:** a dropdown "Preview a MovieLens user" (IDs 1 to 610) shows read-only rows for an existing user with history. No rating in preview mode.
7. Movie cards show title and genre tags. No posters.

## 3. Architecture

```
Browser ──> nginx (host port 8080) ──> api1 ─┐
                                   └─> api2 ─┼──> postgres  (database "app": users, sessions, feedback, logs, drift_reports, retrain_runs)
                                             └──> mlflow (port 5000) ──> postgres (database "mlflow")
simulator ──> nginx
drift     ──> postgres + mlflow ──> reports/drift/*.html, metrics on :8001
retrainer ──> postgres + mlflow ──> POST /reload on api1 and api2
prometheus ──> api1, api2, drift ──> grafana
```

| Component | Tool | Port (host) |
|---|---|---|
| Gateway and load balancer | Nginx | 8080 |
| API and web page (2 copies) | FastAPI | internal 8000 |
| Experiment tracking and model registry | MLflow | 5000 |
| Database | Postgres 16 | internal 5432 |
| Metrics | Prometheus | 9090 |
| Dashboards | Grafana | 3000 |
| Drift check | Evidently (own container) | internal 8001 |

## 4. Decisions already made (do not revisit)

| Area | Decision |
|---|---|
| Dataset | MovieLens `ml-latest-small`; only `ratings.csv` and `movies.csv` |
| Split | By global time: 70% train, 10% val, 10% test, 10% stream (replayed as live traffic) |
| Data versioning | DVC pipeline for data only. No shared DVC remote. `dvc.lock` committed. Teammates run `dvc repro`. |
| Model | Matrix factorization (explicit ALS, plain NumPy) + popularity baseline (also the cold-start fallback) |
| Cold start | Fewer than 3 ratings: popular movies. 3 or more: fold-in (least-squares solve from the user's ratings) |
| Liked | Rating of 4.0 or higher |
| Metrics | RMSE, Precision@10, NDCG@10 (relevant = rating of 4.0 or higher) |
| MLflow | Postgres backend, files via MLflow's artifact proxy. Model name `movie-recommender`, alias `champion` (current), alias `previous` (rollback) |
| Accounts | Username + password only. No email. scrypt hashes. Database-backed sessions (7 days). |
| IDs | MovieLens users 1 to 610 (cannot log in). New accounts start at 100000. |
| Traffic flag | Every feedback and recommendation row has `source` = `simulator` or `web`, set by the server |
| Images | Separate images: api, trainer, simulator, drift, mlflow. The retrainer reuses the trainer image. |
| Data in containers | `data/processed` is mounted read-only from the project folder |
| Gateway | Two named API services (`api1`, `api2`), round robin, failover, `/metrics` and `/reload` hidden from outside |
| Monitoring | Prometheus scrape every 5 s, one auto-provisioned Grafana dashboard, alerts visible in Prometheus only |
| Drift | At least 30% of columns drifted AND at least 500 new rows. Checked every 60 s. |
| Retrain | Promote only if NDCG@10 on recent feedback improves by at least 1% and the fixed test set does not drop by more than 10% |
| CI/CD | GitHub Actions, images to ghcr.io, smoke test on every pull request, `main` protected. Repo assumed public. |
| Kubernetes | kind, one node, plain YAML, Nginx stays as gateway, images loaded into kind, retrainer is a CronJob |

## 5. Final project folder tree (every file in the project)

```
movie-recsys/
├── .env.example                    # committed; real .env is not
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                      # lint, test, data, build, smoke test
│   ├── cd.yml                      # publish images to ghcr.io on main
│   └── k8s-deploy.yml              # kind cluster + smoke test on main
├── Makefile
├── README.md
├── dvc.yaml  dvc.lock  params.yaml
├── pytest.ini  ruff.toml
├── requirements.txt                # one environment for local development and CI
├── requirements/                   # per-image requirements
│   ├── api.txt  trainer.txt  simulator.txt  drift.txt  mlflow.txt
├── docker/
│   ├── api.Dockerfile  trainer.Dockerfile  simulator.Dockerfile
│   ├── drift.Dockerfile  mlflow.Dockerfile
├── docker-compose.yml
├── docs/DEVIATIONS.md
├── infra/
│   ├── postgres/init.sql
│   ├── nginx/nginx.conf  nginx.k8s.conf
│   ├── prometheus/prometheus.yml  prometheus.k8s.yml  alerts.yml
│   └── grafana/
│       ├── provisioning/datasources/datasource.yml
│       ├── provisioning/dashboards/dashboards.yml
│       └── dashboards/movie-recsys.json
├── k8s/
│   ├── kind-config.yaml
│   ├── infra/   kustomization.yaml  namespace.yaml  postgres.yaml  mlflow.yaml
│   ├── app/     kustomization.yaml  api.yaml  nginx.yaml  drift.yaml  retrainer-cronjob.yaml  prometheus.yaml  grafana.yaml
│   └── jobs/    job-trainer-image.yaml.tpl  job-simulator.yaml.tpl
├── scripts/smoke_test.sh
├── src/
│   ├── __init__.py
│   ├── common/   __init__.py  config.py  db.py  reload.py
│   ├── data/     __init__.py  download.py  schemas.py  preprocess.py
│   ├── recsys/   __init__.py  als.py  bundle.py  registry.py  recommend.py  evaluate.py
│   ├── train/    __init__.py  train.py  retrain.py  admin.py
│   ├── api/      __init__.py  main.py  auth.py  state.py  metrics.py  static/index.html
│   ├── simulator/ __init__.py  simulate.py
│   └── drift/    __init__.py  drift_check.py
├── tests/
│   ├── test_preprocess.py  test_als.py  test_recommend.py  test_evaluate.py
│   ├── test_api.py  test_drift_logic.py  test_retrain_logic.py
├── data/            (generated by DVC, not in git)  raw/  processed/
├── metrics/preprocess.json
├── reports/drift/   (generated, not in git)
```

Compose and Kubernetes never run at the same time (same host ports). Run `make down` before `make k8s-up`, and the reverse.

---

# STEP 0: Machine setup and shared files

**Goal:** every teammate has the same tools, and the shared config files exist before any feature work.

## 0.1 Tools to install (once per machine)

| Tool | Needed from | Install |
|---|---|---|
| Git, make, curl, jq | Step 1 | Ubuntu/WSL: `sudo apt install -y git make curl jq`. macOS: `brew install git make curl jq` |
| Python 3.11 + venv | Step 1 | Ubuntu: `sudo apt install -y python3.11 python3.11-venv`. macOS: `brew install python@3.11` |
| Docker Desktop (Windows: WSL2 backend and WSL integration turned on) | Step 4 | https://docs.docker.com/desktop/ . In Docker Desktop settings give it **at least 6 GB memory** (estimate; raise it if containers get killed). |
| kubectl | Step 11 | https://kubernetes.io/docs/tasks/tools/ |
| kind | Step 11 | https://kind.sigs.k8s.io/docs/user/quick-start/#installation |

Checks (all must print a version): `git --version`, `make --version`, `jq --version`, `python3.11 --version` (must say 3.11.x), `docker version`, `docker compose version`, `kubectl version --client`, `kind version`.

## 0.2 Create the repo and environment

```bash
mkdir movie-recsys && cd movie-recsys
git init
python3.11 -m venv .venv
source .venv/bin/activate        # run again in every new terminal, from the project folder
mkdir -p docs && printf "# Deviations from the spec\n" > docs/DEVIATIONS.md
```

## 0.3 Shared files to create now

### requirements.txt (one environment for development and CI)
```
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
pyyaml==6.0.1
requests==2.32.3
pandera==0.20.4
dvc==3.51.2
mlflow==2.14.1
sqlalchemy==2.0.31
psycopg2-binary==2.9.9
fastapi==0.111.0
uvicorn==0.30.1
prometheus-client==0.20.0
evidently==0.4.30
httpx==0.27.0
pytest==8.2.2
ruff==0.5.0
pre-commit==3.7.1
```
Run `pip install -r requirements.txt`. **It must succeed with no conflict.** If pip reports a conflict, fix the pins (see rule 3) before going on. Evidently must stay on the 0.4.x line, because later versions changed the API this spec relies on.

### .gitignore
```
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.env
local.db
mlflow.db
mlartifacts/
reports/
```
DVC adds `data/raw` and `data/processed` to `.gitignore` itself. Do not add them by hand.

### .env.example (copy to `.env`, change the secrets, never commit `.env`)
```
POSTGRES_USER=recsys
POSTGRES_PASSWORD=change-me
POSTGRES_DB=app
DATABASE_URL=postgresql+psycopg2://recsys:change-me@postgres:5432/app
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_NAME=movie-recommender
SIMULATOR_KEY=sim-secret-change-me
ADMIN_KEY=admin-secret-change-me
GRAFANA_ADMIN_PASSWORD=change-me
```
The password inside `DATABASE_URL` must equal `POSTGRES_PASSWORD` (no variable substitution is done). Running Python **outside** Docker (local development) overrides two values: `DATABASE_URL=sqlite:///./local.db` and `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. The Makefile's `*-local` targets set them inline.

### params.yaml (every tunable setting lives here)
```yaml
data:
  url: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
  raw_dir: data/raw
  processed_dir: data/processed
split:
  train: 0.70
  val: 0.10
  test: 0.10
  stream: 0.10
model:
  factors: 32
  reg: 0.1
  iterations: 15
  seed: 42
  min_item_ratings: 10
eval:
  k: 10
  relevant_threshold: 4.0
serving:
  min_ratings_personalize: 3
  liked_threshold: 4.0
  because_rows: 3
  default_n: 10
  max_n: 50
drift:
  interval_seconds: 60
  min_rows: 500
  share_threshold: 0.3
  reference_sample: 10000
  keep_reports: 20
retrain:
  interval_seconds: 60
  min_new_rows: 500
  holdout_fraction: 0.2
  min_improvement: 0.01
  max_test_drop: 0.10
simulator:
  rate: 20
  drift_genre: Horror
  drift_fraction: 0.8
  drift_pool_size: 30
```
If the dataset URL fails, copy the current "ml-latest-small" zip link from https://grouplens.org/datasets/movielens/latest/ into `data.url`.

### pytest.ini
```
[pytest]
pythonpath = .
testpaths = tests
```

### ruff.toml
```
line-length = 100
[lint]
select = ["E", "F", "I"]
```

### .pre-commit-config.yaml (must do)
Two repos: `pre-commit/pre-commit-hooks` with hooks `trailing-whitespace`, `end-of-file-fixer`, `check-yaml` (with `--allow-multiple-documents`), `check-added-large-files`; and `astral-sh/ruff-pre-commit` (rev `v0.5.0`) with hooks `ruff` (with `--fix`) and `ruff-format`. Install once with `pre-commit install`.

## 0.4 Python package rules

- `src/` is a package (`src/__init__.py`, empty) and so is every folder below it. Keep `src/common/__init__.py` empty (no imports), because the simulator image has no database libraries and must still be able to import `src.common.config`. Run code as modules from the project root: `python -m src.train.train`. Never run a file by path.
- In Docker, the working folder is `/app` and `PYTHONPATH=/app`.
- `src/common/config.py` must provide one `get_settings()` returning: database URL, MLflow URI, model name, simulator key, admin key, replica name (`REPLICA_NAME`, default `local`), reload targets (`API_RELOAD_TARGETS`, comma separated URLs), headless host (`API_HEADLESS_HOST`), and a `load_params()` function that reads `params.yaml` (path from env `PARAMS_PATH`, default `params.yaml`).

## 0.5 Shared contracts (every step relies on these)

### A. Model bundle (a folder logged to MLflow as artifact path `bundle`)

| File | Content |
|---|---|
| `factors.npz` | `user_factors` (users x factors), `item_factors` (items x factors), `user_ids`, `item_ids` (the id of each row), `global_mean` (float) |
| `movies.parquet` | `movieId`, `title`, `genres` (list of strings) for all movies |
| `popularity.parquet` | `movieId`, `n_ratings` (count in the training data), ordered by `n_ratings` descending |
| `user_history.parquet` | `userId`, `movieId`, `rating`, `timestamp` for every rating the model was trained on |
| `reference.parquet` | drift reference, one row per training rating: `rating`, `main_genre` (first genre of the movie, or `none`), `item_popularity` (that movie's `n_ratings`) |
| `meta.json` | `created_at` (UTC ISO), `params` (model settings), `metrics`, `train_rows`, `feedback_rows` (0 for v1), `parent_version` (null for v1) |

### B. Recommending rules (one implementation in `src/recsys/recommend.py`, used by the API, the evaluator and the retrainer)

- **Candidates:** movies with `n_ratings >= model.min_item_ratings`, minus movies the user already rated.
- **Score:** predicted rating = `global_mean + user_vector · item_vector`, clipped to 0.5 to 5.0.
- **User vector** (`user_vector(bundle, history)`):
  1. Fewer than `serving.min_ratings_personalize` ratings in total: return nothing (caller uses popular).
  2. If the user exists in the model **and** every rating in the history is already part of the model's own `user_history`: use the saved vector.
  3. Otherwise **fold in**: solve the same ridge least-squares as ALS (`(VᵀV + reg·n·I) x = Vᵀ(r − global_mean)`, using only rated movies that exist in the model).
- **Popular:** movies in `popularity.parquet` order, filtered by the candidate rule.
- **Because rows:** take the user's `serving.because_rows` most recent ratings of `serving.liked_threshold` or higher (by timestamp). For each such movie X that exists in the model, return the `n` candidates with the highest **cosine similarity** between item vectors (X itself excluded).
- **User history** = the bundle's `user_history` rows for that user plus the user's `feedback` rows from the database; for the same movie the **latest** rating wins.

### C. Database tables (database `app`; SQLAlchemy models in `src/common/db.py`; every service calls `create_all` at startup, which is safe to repeat)

| Table | Columns |
|---|---|
| `users` | `id` (auto key), `username` (lowercase, unique), `password_hash`, `created_at`. The public `user_id` is `100000 + id`. |
| `sessions` | `token_hash` (SHA-256 hex of the token, key), `user_id`, `created_at`, `expires_at` |
| `feedback` | `id`, `user_id`, `movie_id`, `rating` (float), `source` (`web` or `simulator`), `created_at`. Insert-only. Index on `user_id` and on `created_at`. |
| `recommendation_logs` | `id`, `user_id`, `kind` (`popular`, `personalized`, `because`), `model_version` (int), `movie_ids` (JSON text), `scores` (JSON text), `source`, `created_at`. Index on `user_id`. |
| `drift_reports` | `id`, `created_at`, `model_version`, `n_rows`, `rows_by_source` (JSON text), `drift_share` (nullable), `drift_detected` (bool), `live_hit_rate` (nullable), `report_path` (nullable), `note` (nullable) |
| `retrain_runs` | `id`, `started_at`, `finished_at` (nullable), `trigger` (`drift`, `manual`, `forced`), `status` (`running`, `done`, `failed`), `rows_used`, `holdout_rows`, `v1_version`, `v2_version`, `v1_ndcg_holdout`, `v2_ndcg_holdout`, `v1_ndcg_test`, `v2_ndcg_test`, `decision` (`promoted`, `rejected`, `forced`, `failed`), `note` |

### D. API contract (all JSON; errors use FastAPI's `{"detail": ...}`)

| Endpoint | Auth | Request | Response |
|---|---|---|---|
| `POST /signup` | none | `{username, password}` | `{user_id, username, token}`. 409 if the name exists. 422 if rules fail. |
| `POST /login` | none | `{username, password}` | `{user_id, username, token}`. 401 if wrong. |
| `POST /logout` | Bearer | none | `{ok: true}`, deletes the session |
| `GET /me` | Bearer | none | `{user_id, username, n_ratings}`. 401 if token missing or expired. |
| `GET /popular?n=` | optional Bearer | none | `{kind: "popular", model_version, movies: [...]}` (movies already rated by the logged-in user are removed) |
| `GET /recommend?n=` | Bearer, or simulator key + `user_id` | none | `{kind: "popular" or "personalized", model_version, movies: [...]}` (logged) |
| `GET /because?n=` | Bearer, or simulator key + `user_id` | none | `{model_version, rows: [{because_you_liked: {movie_id, title}, movies: [...]}]}` (logged). Empty `rows` if no liked movie. |
| `GET /preview/{user_id}` | none | none | `{recommended: {kind, movies}, because: {rows}}` for MovieLens users that exist in the model. 404 otherwise. Read-only, not logged. |
| `POST /feedback` | Bearer, or simulator key | `{movie_id, rating}` (+ `user_id` with the simulator key) | `{ok: true}`. Rating must be 0.5 to 5.0 in steps of 0.5. 404 if the movie is unknown. |
| `GET /health` | none | none | `{status: "ok", model_version, replica}`. 503 until a model is loaded. |
| `GET /metrics` | none | none | Prometheus text format |
| `POST /reload` | header `X-Admin-Key` | none | `{model_version}`. 403 if the key is wrong. |
| `GET /` | none | none | the web page |

- A **movie object** is `{movie_id, title, genres: [..], score}`. Default `n` is `serving.default_n`, maximum `serving.max_n`.
- Bearer token goes in the header `Authorization: Bearer <token>`.
- **Simulator key:** header `X-Simulator-Key` equal to `SIMULATOR_KEY`. With a valid key, the request acts as MovieLens user `user_id` (must be 1 to 99999, else 403) and rows get `source = simulator`. Every other request gets `source = web`. A wrong key returns 403. A normal user can never act as another user.
- Every response carries the header `X-Served-By: <REPLICA_NAME>`.

### E. Prometheus metric names

| Source | Metric | Type and labels |
|---|---|---|
| API | `http_requests_total` | counter: `endpoint` (route path such as `/recommend`), `method`, `status`, `source` |
| API | `http_request_duration_seconds` | histogram: `endpoint` |
| API | `recommendations_served_total` | counter: `kind` (`popular`, `personalized`, `because`) |
| API | `feedback_total` | counter: `source` |
| API | `feedback_rating_sum` | counter (exposed as `feedback_rating_sum_total`): `source`. Adds the rating value. |
| API | `model_version` | gauge: loaded model version number |
| Drift | `drift_share`, `drift_detected` (0 or 1), `drift_window_rows`, `live_hit_rate`, `drift_last_check_timestamp` | gauges |
| Drift | `drift_rows_by_source` | gauge: `source` |

## 0.6 Makefile (all targets; create the Makefile in the step where each target first works)

| Target | What it does | Step |
|---|---|---|
| `make data` | `dvc repro` | 1 |
| `make test` / `make lint` | `pytest -q` / `ruff check .` | 1 |
| `make mlflow-local` | `mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts` (own terminal) | 2 |
| `make train-local` | trainer against local MLflow, with `--if-missing` | 2 |
| `make api-local` | `uvicorn src.api.main:app --port 8000` with `DATABASE_URL=sqlite:///./local.db`, local MLflow, `REPLICA_NAME=local`, dev values for `SIMULATOR_KEY` and `ADMIN_KEY` | 3 |
| `make build` | `docker compose build` | 4 |
| `make infra` | `docker compose up -d --wait postgres mlflow` | 4 |
| `make train` | one-off trainer container with `--if-missing` | 4 |
| `make train-new` | one-off trainer container that always registers a new version (not promoted if a champion exists) | 4 |
| `make up` | check that a champion exists, then `docker compose up -d --wait api1 api2 nginx prometheus grafana drift retrainer` (Nginx added in Step 5, the rest in the steps that create them) | 4 |
| `make down` | `docker compose down` | 4 |
| `make clean` | `docker compose down -v` (deletes all volumes) | 4 |
| `make logs` | `docker compose logs -f --tail=100` | 4 |
| `make simulate` / `make simulate-drift` | one-off simulator container, mode `normal` / `drift`; optional `LIMIT=2000` | 6 |
| `make reset-feedback` | delete simulator rows from `feedback` and `recommendation_logs`, and clear `drift_reports` and `retrain_runs` | 6 |
| `make drift-check` | one drift check now | 8 |
| `make retrain` / `make retrain-force` | one retrain now (manual / skip the quality gate) | 9 |
| `make rollback` | `champion` back to `previous`, then reload both API copies | 9 |
| `make demo-reset` | `reset-feedback`, set champion to version 1, reload | 9 |
| `make k8s-up` `k8s-down` `k8s-status` `k8s-simulate` `k8s-simulate-drift` `k8s-retrain` `k8s-rollback` `k8s-demo-reset` `k8s-rollout` | see Step 11 | 11 |

Add `.PHONY` for all targets. Every target that needs data first checks that `data/processed/train.parquet` exists and otherwise prints "run make data first" and stops.

## CHECKPOINT 0

- [ ] All tool checks in 0.1 print versions
- [ ] `pip install -r requirements.txt` succeeds in the 3.11 venv
- [ ] `.env` exists (copied from `.env.example`) and is not tracked by git
- [ ] `pre-commit install` ran

---

# STEP 1: Data pipeline (preprocessing with DVC)

**Goal:** turn raw `ratings.csv` and `movies.csv` into validated, cleaned, time-split Parquet files. `dvc repro` rebuilds everything.
**Needs:** Step 0.

**Locked choices:** global time split 70/10/10/10. Some val/test/stream users have no history in train: this is intended (cold-start story), document it, do not fix it. No shared DVC remote: data is rebuilt by `dvc repro`, `dvc.lock` is committed, `data/` never is.

## Files added

```
dvc.yaml
src/__init__.py  src/data/__init__.py  src/common/__init__.py  (all empty)
src/common/config.py  (minimal for now: load_params)
Makefile  (targets data, test, lint)
src/data/download.py  src/data/schemas.py  src/data/preprocess.py
tests/test_preprocess.py
data/raw/  data/processed/  metrics/preprocess.json   (generated)
```

## dvc.yaml
```yaml
stages:
  download:
    cmd: python -m src.data.download
    deps:
      - src/data/download.py
    params:
      - data.url
      - data.raw_dir
    outs:
      - data/raw

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/schemas.py
      - src/data/preprocess.py
      - data/raw
    params:
      - data.raw_dir
      - data.processed_dir
      - split
    outs:
      - data/processed
    metrics:
      - metrics/preprocess.json:
          cache: false
```

## What each file must do

All scripts run from the project root and read `params.yaml` through `load_params()` (create a minimal `src/common/config.py` now with just `load_params`; Step 2 extends it).

**src/data/download.py**
- `download_raw(url, raw_dir)`: create `raw_dir`; if `ratings.csv` and `movies.csv` already exist, print a message and return. Otherwise download the zip with `requests` (timeout 60 s, raise on HTTP error), extract **only** those two files and flatten the zip's inner folder so they land directly in `raw_dir`. Raise an error if either is missing afterwards.
- `main()` reads `data` params and calls it.

**src/data/schemas.py** (Pandera, `strict`, no extra columns)
- `ratings_schema`: `userId` int >= 1; `movieId` int >= 1; `rating` float 0.5 to 5.0; `timestamp` int >= 0.
- `movies_schema`: `movieId` int, unique; `title` string, not null; `genres` string, not null.

**src/data/preprocess.py** (keep these names; later steps import them)

| Function | Behavior |
|---|---|
| `clean_movies(movies)` | drop rows with null `movieId` or `title`; drop duplicate `movieId`; turn `genres` `"A\|B"` into a list of strings (`"(no genres listed)"` becomes an empty list); fresh index |
| `clean_ratings(ratings, movie_ids)` | drop rows with any null; per (`userId`, `movieId`) keep only the latest by `timestamp`; keep only movies in `movie_ids`; add `datetime` (from `timestamp`, unix seconds); sort by (`timestamp`, `userId`, `movieId`); fresh index |
| `split_by_time(ratings, fractions)` | `fractions` has keys `train`, `val`, `test`, `stream`; raise `ValueError` unless they sum to 1.0 (tolerance 1e-9); sort by (`timestamp`, `userId`, `movieId`); cut by row position at cumulative fractions using `round(n * cumulative)`; return dict of 4 DataFrames in time order |
| `compute_metrics(splits, movies)` | `n_movies`; per split `rows`, `users`, `start`, `end` (strings); for val/test/stream also `users_not_in_train` |
| `main()` | read params and CSVs; validate both with `lazy=True`; clean; split; write `train/val/test/stream.parquet` and `movies.parquet` (`index=False`) to `processed_dir`; write `metrics/preprocess.json` (indented); print row counts |

**tests/test_preprocess.py** (synthetic data, no download; 5 tests): latest duplicate wins; unknown movies dropped; split sizes add up and `max(timestamp)` of each piece is `<=` the next piece's `min` (ties allowed); no (user, movie) pair in two splits; a real-files test that is skipped when `data/processed/train.parquet` is missing (no duplicate pairs, time order across splits, ratings 0.5 to 5.0).

## Commands

```bash
dvc init
dvc config core.analytics false
make data            # = dvc repro
make test            # expect: 5 passed
git add . && git commit -m "Step 1: data pipeline"
dvc repro            # second run: "Data and pipelines are up to date."
```
Teammate on a fresh clone: create venv, `pip install -r requirements.txt`, `dvc repro`.

## Output contract (later steps may rely on this)

| File | Columns |
|---|---|
| `data/processed/train.parquet`, `val.parquet`, `test.parquet`, `stream.parquet` | `userId` int64, `movieId` int64, `rating` float64, `timestamp` int64 (unix seconds), `datetime` datetime64 |
| `data/processed/movies.parquet` | `movieId` int64, `title` str, `genres` list of str |
| `metrics/preprocess.json` | counts, dates, unseen-user counts |

## CHECKPOINT 1

- [ ] `dvc repro` succeeds; a second run says up to date
- [ ] 5 Parquet files in `data/processed/`; `metrics/preprocess.json` shows about 100,836 rows in total (may differ slightly if GroupLens updates the file), split roughly 70585 / 10084 / 10083 / 10084, about 9,700 movies
- [ ] `users_not_in_train` is greater than 0 for `stream`
- [ ] `make test` passes; `git status` shows `data/raw` and `data/processed` untracked-and-ignored
- [ ] `dvc.lock` and `metrics/preprocess.json` committed

---

# STEP 2: Model, evaluation and MLflow

**Goal:** train the popularity baseline and the matrix factorization model from `train.parquet`, evaluate both on val and test, log everything to MLflow, and register the result as `movie-recommender` with alias `champion`.
**Needs:** Step 1.

## Files added

```
src/common/__init__.py  src/common/config.py (extend)
src/recsys/__init__.py  als.py  bundle.py  registry.py  recommend.py  evaluate.py
src/train/__init__.py   train.py
tests/test_als.py  tests/test_recommend.py  tests/test_evaluate.py
```

## What each file must do

**src/common/config.py**: complete `get_settings()` and `load_params()` as described in Step 0.4.

**src/recsys/als.py**: explicit ALS in plain NumPy. No other ML library.
- `train_als(ratings, factors, reg, iterations, seed) -> dict` with `user_factors`, `item_factors`, `user_ids`, `item_ids`, `global_mean`. `ratings` is a DataFrame with `userId`, `movieId`, `rating`.
  - Center ratings by subtracting the global mean. Initialize both factor matrices from a normal distribution (mean 0, std 0.1) using `seed`.
  - Each iteration: for every user solve `(VᵀV + reg·n_u·I) x = Vᵀ r` over the movies that user rated (`n_u` = their rating count); then the same for every movie over its users. Use sorted arrays and index pointers, not a Python loop over DataFrame rows.
  - Print the training RMSE after each iteration.
- `solve_vector(other_factors, centered_ratings, reg) -> vector`: the one ridge solve used by training and by fold-in.
- `predict(user_vec, item_factors, global_mean)`: clipped to 0.5 to 5.0.

**src/recsys/bundle.py**: build, save, load the model bundle (Step 0.5 A).
- `build_bundle(model, ratings, movies, params, feedback_rows, parent_version) -> Bundle` (a small class or dataclass). Computes popularity, `reference.parquet` data and `meta.json`.
- `save_bundle(bundle, folder)` / `load_bundle(folder) -> Bundle`. `Bundle` exposes fast lookups built at load time: item id to row index, normalized item vectors for cosine similarity, popularity order, per-user history dict, movie title and genres by id.

**src/recsys/registry.py**: read helpers for the MLflow registry, used by the API, the drift check, the retrainer and the admin commands. `alias_version(settings, alias) -> int or None`; `fetch_bundle(settings, version) -> Bundle` (download that version's run artifact folder `bundle` into `/tmp/model_cache/<version>`, reuse the folder if it is already there, then `load_bundle`); `fetch_alias(settings, alias) -> (Bundle, version)`. The tracking address comes from settings.

**src/recsys/recommend.py**: the recommending rules of Step 0.5 B.
- `user_vector(bundle, history)`, `recommend_popular(bundle, seen, n)`, `recommend_for_user(bundle, history, n) -> (kind, list)`, `because_rows(bundle, history, n)`. `history` is a DataFrame of `movieId`, `rating`, `timestamp`.

**src/recsys/evaluate.py**
- `evaluate(bundle, eval_df, seen_df, k, threshold, min_ratings) -> dict` with `rmse`, `precision_at_k`, `ndcg_at_k`, `n_users`, `n_rows`.
  - **RMSE:** over eval rows whose user and movie exist in the model; prediction uses `user_vector` built from the user's `seen_df` ratings; skip rows where no vector exists.
  - **Ranking metrics:** for each user with at least one relevant eval rating (`rating >= threshold`) **and** at least one rating in `seen_df`: recommend with `recommend_for_user` using `seen_df` as the history (so users with fewer than 3 ratings get the popular list, like in the API). Precision@K = hits / K. NDCG@K = binary-relevance DCG divided by ideal DCG. Average over users.
- `evaluate_popularity(bundle, eval_df, seen_df, k, threshold) -> dict`: same ranking metrics with the popular list for every user; RMSE predicts the global mean.

**src/train/train.py** (`python -m src.train.train [--if-missing] [--promote]`)
1. Read params, `train.parquet`, `val.parquet`, `test.parquet`, `movies.parquet`.
2. If `--if-missing` and the registry already has a `champion`, print a message and exit 0.
3. Train ALS on `train.parquet`; build the bundle with `feedback_rows=0`, `parent_version=None`.
4. Evaluate on val and test (`seen_df` = `train.parquet`), model and popularity baseline.
5. Set MLflow tracking URI from settings, experiment `movie-recsys`. In one run log: the model params, all metrics (`val_rmse`, `val_precision_at_10`, `val_ndcg_at_10`, same for `test_`, and `baseline_val_*`, `baseline_test_*`), the bundle folder as artifact path `bundle`, `params.yaml` and `dvc.lock`, and the tags `git_commit` (if available) and `stage=initial`.
6. Register `runs:/<run_id>/bundle` under the model name from settings. If there is no `champion` alias yet, set it to this version. If one exists, set it only when `--promote` is given; otherwise print that the version was registered but not promoted.
7. Print a short summary table.

## Tests (all use small synthetic data, no MLflow)

- `test_als.py`: on a small synthetic low-rank rating table, training RMSE after the run is lower than predicting the global mean; same seed gives identical factors.
- `test_recommend.py`: popular list skips seen movies and low-count movies; user with fewer than 3 ratings gets kind `popular`; a user whose history is entirely inside the model uses the saved vector, while one with an extra rating is folded in (vectors differ); because rows return at most 3 rows and never include the liked movie itself or seen movies.
- `test_evaluate.py`: a model whose recommendations exactly match the relevant items gives NDCG@K of 1.0; a list with no hits gives 0.

## Commands (local, no Docker yet)

```bash
make test
make mlflow-local          # terminal 1, leave running; MLflow UI at http://127.0.0.1:5000
make train-local           # terminal 2
```
Expected: training RMSE printed per iteration, then a summary table; MLflow UI shows experiment `movie-recsys` with one run, and Models shows `movie-recommender` version 1 with alias `champion`.

## CHECKPOINT 2

- [ ] `make test` passes
- [ ] `make train-local` finishes in about a minute or less; running it again prints "champion already exists" and exits (because of `--if-missing`)
- [ ] MLflow run contains the metrics listed above and the `bundle` artifact with all six files from Step 0.5 A
- [ ] Model `movie-recommender` has version 1 with alias `champion`
- [ ] `val_rmse` is below `baseline_val_rmse` (global-mean RMSE, about 1.04 on this data). Ranking scores are low on this dataset; report them, do not tune to chase them.

---

# STEP 3: Serving API and web app (with accounts)

**Goal:** a FastAPI service that loads the `champion` bundle from MLflow, answers recommendation requests using only quick maths, handles signup/login, saves ratings, and serves one HTML page.
**Needs:** Steps 1 and 2 (a local MLflow with a champion). Contracts: Step 0.5 B, C, D, E.

## Files added

```
src/common/db.py  src/common/reload.py
src/api/__init__.py  main.py  auth.py  state.py  metrics.py  static/index.html
tests/test_api.py
```

## What each file must do

**src/common/db.py**: SQLAlchemy models for **all** tables in Step 0.5 C, `get_engine(url)`, `get_session` helper, `init_db(engine)` (calls `create_all`). Must work with Postgres and SQLite (tests and local dev use SQLite). `created_at` stored in UTC.

**src/common/reload.py**: `reload_replicas(admin_key) -> list`. Targets come from settings: if `API_HEADLESS_HOST` is set (host:port), resolve **all** its addresses by DNS and call each; otherwise use `API_RELOAD_TARGETS`. Call `POST /reload` with `X-Admin-Key` on each target **one after the other**, with a 10 s timeout; return a list of (target, success, model_version). One failed target must not stop the others.

**src/api/state.py**: holds the loaded model.
- `load_champion(settings) -> LoadedModel`: call `fetch_alias(settings, "champion")` from `src/recsys/registry.py` and return the bundle plus the version number.
- `ModelHolder` with `get()` and `swap(new)`: build the new model first, then replace the reference in one step, so requests never see a half-loaded model.

**src/api/auth.py**
- `hash_password(pw)` and `verify_password(pw, stored)`: Python's `hashlib.scrypt` (n=2**14, r=8, p=1), a random 16-byte salt, stored as `salt_hex$hash_hex`, compared with `hmac.compare_digest`.
- `create_session(db, user_id)`: token from `secrets.token_urlsafe(32)`, store only its SHA-256, expiry 7 days.
- `current_user(...)` dependency: read the Bearer token, look up the hash, check expiry, return the user or raise 401. `optional_user(...)` returns `None` for guests.
- Simulator key check: a helper that returns `source` (`simulator` if the header equals `SIMULATOR_KEY`, 403 if present but wrong, else `web`).
- Username rule: 3 to 30 characters of letters, digits, underscore; stored lowercase; unique. Password: 6 to 128 characters.

**src/api/metrics.py**: the metric objects of Step 0.5 E and a middleware that records `http_requests_total` and `http_request_duration_seconds` using the **route path template** (not the raw URL) as `endpoint` (`unmatched` if no route). `source` label = `simulator` when the request carries a valid simulator key, else `web`. Also helper functions to increment `recommendations_served_total`, `feedback_total`, `feedback_rating_sum`, and to set `model_version`.

**src/api/main.py**
- `create_app(database_url=None, model_loader=None) -> FastAPI`, and a module-level `app = create_app()`. Tests pass an SQLite URL and a loader that returns a small fake bundle.
- On startup: `init_db`, then load the champion. **If no champion exists or loading fails, raise a clear error** ("No champion model in MLflow. Run the trainer first") so the process stops. Set the `model_version` gauge.
- Add the middleware that sets `X-Served-By` from `REPLICA_NAME`.
- Implement every endpoint of Step 0.5 D with the exact request/response shapes and status codes.
  - **History for a user** = bundle history plus that user's `feedback` rows, latest rating per movie wins (Step 0.5 B). `n_ratings` in `/me` counts that merged history.
  - `/recommend` and `/because` insert a `recommendation_logs` row (kind, model version, movie ids, scores, source) and increment `recommendations_served_total` (`/recommend` counts as `popular` or `personalized` by the returned kind; `/because` counts as `because` once per call).
  - `/feedback` validates the rating (0.5 to 5.0, multiples of 0.5, else 422), checks the movie exists in the bundle's movies table (else 404), inserts the row with the server-chosen `source`, updates the feedback metrics.
  - `/preview/{user_id}`: only IDs that exist in the model and are 1 to 99999; computed from bundle history only (no database).
  - `/reload`: requires `X-Admin-Key` equal to `ADMIN_KEY`; load the champion and swap; return the new version.
  - `/health`: 503 until a model is loaded.
  - `GET /`: return `static/index.html`.

**src/api/static/index.html**: one file, plain JavaScript, plain CSS, **no framework, no CDN, no build step**.
- On load: call `/me` if a token is in `localStorage` (clear it on 401); call `/popular`. Logged in users call `/recommend` and `/because` as well.
- Header: when logged out, signup/login form (username, password, two buttons, error text from the API); when logged in, `Hello <username>` and Log out.
- Rows rendered as horizontal lists of cards: **Popular**, **Recommended for you** (only when `/recommend` returns kind `personalized`), and one **Because you liked <title>** row per entry in `/because`. Hide a row with no movies.
- Card: title, genre chips, five stars. Click a star: logged in sends `POST /feedback` then reloads all rows; guest opens the signup form. In preview mode stars are disabled.
- "Preview a MovieLens user" dropdown (numbers 1 to 610) calls `/preview/{id}` and shows a banner "Read-only preview of MovieLens user N" with an Exit button. A 404 shows "That user is not in the model".
- Works at phone width. Errors from the API are shown as a short message, never a blank page.

## Tests (tests/test_api.py, FastAPI `TestClient`, SQLite, fake bundle; no MLflow, no Docker)

signup then `/me`; duplicate username returns 409; wrong password returns 401; weak password returns 422; `/popular` works for a guest; `/feedback` without a token returns 401; a new user gets `kind=popular` from `/recommend`; after 3 ratings `/recommend` returns `personalized` and never contains rated movies; after one 4-star rating `/because` has a row; rating 0.7 returns 422; unknown movie returns 404; simulator key plus `user_id` stores `source=simulator` while a normal call stores `web`; simulator key with `user_id` 100001 returns 403; a wrong simulator key returns 403; `/reload` without the admin key returns 403; `/health` shows the replica name and version; response has `X-Served-By`.

## Commands

```bash
make mlflow-local            # terminal 1 (leave running; Step 2 left a champion in mlflow.db)
make api-local               # terminal 2; open http://127.0.0.1:8000
make test
```
Manual walk-through in the browser: popular row appears as a guest, sign up, rate three movies (at least one 4 or 5 stars), watch "Because you liked ..." and "Recommended for you" appear, reload the page (still logged in), log out and in again (same history), try the preview dropdown.

## CHECKPOINT 3

- [ ] `make test` passes (all API tests included)
- [ ] The manual walk-through behaves exactly as listed above
- [ ] Stopping MLflow and starting the API prints the clear "No champion / cannot load" error and the process exits
- [ ] `curl -s localhost:8000/metrics | grep http_requests_total` shows counters after some clicks
- [ ] Replica name `local` appears in `/health` and in the `X-Served-By` header

---

# STEP 4: Docker and Compose

**Goal:** one command set starts Postgres, MLflow and the API in containers, the same on every laptop.
**Needs:** Steps 1 to 3, Docker Desktop. `.env` exists.

## Files added

```
docker/api.Dockerfile  trainer.Dockerfile  mlflow.Dockerfile
requirements/api.txt  trainer.txt  mlflow.txt
infra/postgres/init.sql
docker-compose.yml
.dockerignore
src/train/__init__.py  src/train/admin.py   (only the check-champion command for now)
```
(`simulator` and `drift` Dockerfiles and requirements are added in Steps 6 and 8.)

## Per-image requirements (same pins as the root `requirements.txt`)

| File | Packages |
|---|---|
| `requirements/api.txt` | numpy, pandas, pyarrow, pyyaml, mlflow, sqlalchemy, psycopg2-binary, fastapi, uvicorn, prometheus-client |
| `requirements/trainer.txt` | numpy, pandas, pyarrow, pyyaml, requests, mlflow, sqlalchemy, psycopg2-binary |
| `requirements/simulator.txt` | pandas, pyarrow, pyyaml, requests |
| `requirements/drift.txt` | numpy, pandas, pyarrow, pyyaml, mlflow, sqlalchemy, psycopg2-binary, evidently, prometheus-client |
| `requirements/mlflow.txt` | mlflow, psycopg2-binary |

**Rule:** the `mlflow` version is **identical** in `api`, `trainer`, `drift` and `mlflow` requirements (2.14.1). A client/server version mismatch causes strange errors.

## Dockerfiles (must do)

All start from `python:3.11-slim`, set `WORKDIR /app`, `PYTHONPATH=/app`, `PYTHONUNBUFFERED=1`; copy the image's requirements file first and `pip install --no-cache-dir -r` it (so the layer is cached), then copy only the code folders the image needs. Build context is the project root.

| Image tag | Copies | Start command |
|---|---|---|
| `movie-recsys-api:dev` | `src/__init__.py`, `src/common`, `src/recsys`, `src/api` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` |
| `movie-recsys-trainer:dev` | `src/__init__.py`, `src/common`, `src/recsys`, `src/train` | `python -m src.train.train --if-missing` |
| `movie-recsys-mlflow:dev` | nothing from `src` | set by compose (see below) |

`.dockerignore`: `.venv`, `.git`, `data`, `reports`, `mlflow.db`, `mlartifacts`, `local.db`, `__pycache__`, `.pytest_cache`.

## infra/postgres/init.sql
One statement: create the database `mlflow`. (The database `app` is created by the Postgres image through `POSTGRES_DB`.)

## src/train/admin.py (partial)
`python -m src.train.admin check-champion`: exit 0 if the registry has the `champion` alias for the model, otherwise print "No champion model. Run: make train" and exit 1.

## docker-compose.yml (must do)

All Python services use `env_file: .env`, mount `./params.yaml:/app/params.yaml:ro` (so editing it needs only a restart), and set `restart: unless-stopped` (except one-off jobs). Pin third-party images: `postgres:16`.

| Service | Image or build | Details |
|---|---|---|
| `postgres` | `postgres:16` | env `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` from `.env`; volume `pgdata:/var/lib/postgresql/data`; mount `./infra/postgres/init.sql` into `/docker-entrypoint-initdb.d/init.sql` read-only; healthcheck `pg_isready -U $POSTGRES_USER -d app` |
| `mlflow` | build `docker/mlflow.Dockerfile`, image `movie-recsys-mlflow:dev` | command: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://<user>:<password>@postgres:5432/mlflow --artifacts-destination /mlflow-artifacts --serve-artifacts` (user and password from `.env`); volume `mlflow-artifacts:/mlflow-artifacts`; port `5000:5000`; depends on `postgres` healthy; healthcheck: Python one-liner calling `http://localhost:5000/health` |
| `trainer` | build `docker/trainer.Dockerfile`, image `movie-recsys-trainer:dev` | **profile `tools`** (not started by `up`); mount `./data/processed:/app/data/processed:ro`; depends on `mlflow` healthy |
| `api1` | build `docker/api.Dockerfile`, image `movie-recsys-api:dev` | env `REPLICA_NAME=api1`; depends on `postgres` and `mlflow` healthy; healthcheck: Python one-liner calling `http://localhost:8000/health` (start period 30 s); **for now** publish `8000:8000` (removed in Step 5) |
| `api2` | image `movie-recsys-api:dev` (no build) | same as `api1` with `REPLICA_NAME=api2` and no published port |

Named volumes: `pgdata`, `mlflow-artifacts`.

## Makefile additions
`build`, `infra`, `train`, `train-new`, `up`, `down`, `clean`, `logs` as in Step 0.6. `train` runs `docker compose run --rm trainer` (default command already has `--if-missing`). `train-new` runs the trainer with command `python -m src.train.train` (no `--if-missing`). `up` first runs `docker compose run --rm trainer python -m src.train.admin check-champion` and stops with its message when it fails.

## Commands

```bash
make data            # data must exist on the host
make build
make infra           # waits until postgres and mlflow are healthy
make train           # registers version 1 as champion in the Dockerized MLflow
make up
```
Open http://localhost:5000 (MLflow shows `movie-recommender` v1 with alias `champion`) and http://localhost:8000 (the web app).

## CHECKPOINT 4

- [ ] `make build` builds three images without errors
- [ ] `make infra` ends with both services healthy; `make train` finishes and MLflow shows version 1 with alias `champion`
- [ ] `make up` starts `api1` and `api2`; the web app works at http://localhost:8000 (signup, rate, rows appear)
- [ ] `make train` run again exits quickly with "champion already exists"
- [ ] `make down` then `make infra && make up` keeps the user account and ratings (volumes survive); `make clean` removes them
- [ ] Running `make up` before `make train` on a clean volume stops with "No champion model. Run: make train"

---

# STEP 5: Nginx gateway and load balancing

**Goal:** users and the simulator talk only to Nginx on port 8080, which alternates between `api1` and `api2` and survives one of them going down.
**Needs:** Step 4.

## Files added
`infra/nginx/nginx.conf` (and later `nginx.k8s.conf`), `nginx` service in compose.

## infra/nginx/nginx.conf (must do)
- One `upstream` named `api` with servers `api1:8000` and `api2:8000`, round robin, each with `max_fails=2 fail_timeout=5s`.
- `proxy_next_upstream error timeout http_502 http_503 http_504`, `proxy_connect_timeout 2s`, `proxy_read_timeout 30s`.
- `location = /metrics` and `location = /reload` both `return 404;`.
- `location /` proxies to `http://api` and passes `Host`, `X-Real-IP`, `X-Forwarded-For`, `X-Forwarded-Proto`.
- Access log format with: client address, request line, status, `$request_time`, `$upstream_addr`, `$upstream_http_x_served_by`.
- Listens on port 80 inside the container. No HTTPS, no caching, no compression.

`infra/nginx/nginx.k8s.conf` is identical except the upstream has the single server `api:8000` (the Kubernetes Service). Create it in Step 11.

## Compose changes
- Add service `nginx`: image `nginx:1.27-alpine`, port `8080:80`, mounts `./infra/nginx/nginx.conf` to `/etc/nginx/nginx.conf` read-only, depends on `api1` and `api2` healthy (Nginx fails to start if an upstream host does not exist yet), healthcheck `wget -qO- http://localhost/health`.
- **Remove** the published port `8000:8000` from `api1`. Neither API copy is reachable from the host any more.
- Add `nginx` to the `make up` service list.

## Verify

```bash
make up
for i in 1 2 3 4; do curl -si localhost:8080/health | grep -i x-served-by; done   # alternates api1, api2
curl -si localhost:8080/metrics | head -1                                        # 404
curl -si -X POST localhost:8080/reload | head -1                                 # 404
docker compose stop api1
for i in 1 2 3 4; do curl -s -o /dev/null -w "%{http_code}\n" localhost:8080/health; done   # all 200, served by api2
docker compose start api1
```

## CHECKPOINT 5

- [ ] The web app works at http://localhost:8080 (and no longer at 8000)
- [ ] `X-Served-By` alternates between `api1` and `api2`
- [ ] `/metrics` and `/reload` return 404 through Nginx
- [ ] With `api1` stopped every request still succeeds; after restart both serve again
- [ ] Nginx access log (`docker compose logs nginx`) shows the upstream address and response time

---

# STEP 6: Traffic simulator

**Goal:** a script that plays the newest 10% of ratings (`stream.parquet`) against the app as if real users were rating today. It fills the database, gives Prometheus real traffic, and produces the data for drift detection and retraining.
**Needs:** Step 5.

## Files added

```
src/simulator/__init__.py  src/simulator/simulate.py
docker/simulator.Dockerfile  requirements/simulator.txt (list is in Step 4)
simulator service in docker-compose.yml
```
`src/train/admin.py` gains the command `reset-feedback`.

## src/simulator/simulate.py (`python -m src.simulator.simulate`)

Options (defaults from `params.yaml` section `simulator` unless noted): `--mode normal|drift` (default `normal`), `--rate` (events per second), `--limit` (default: all events), `--genre` (`drift_genre`), `--drift-fraction`.
Environment: `API_URL` (inside Docker `http://nginx:80`, local run `http://localhost:8080`), `SIMULATOR_KEY`.

Behavior:
1. Read `data/processed/stream.parquet` (sorted by `timestamp`), `movies.parquet` and `train.parquet`. Keep the first `--limit` events.
2. Before starting, poll `GET /health` for up to 60 s; stop with a clear message if the API never answers.
3. **Normal mode:** for each event, in order: `GET /recommend?n=10&user_id=<userId>` with header `X-Simulator-Key`, then `POST /feedback` with `{user_id, movie_id, rating}` and the same header.
4. **Drift mode:** build a pool of the `drift_pool_size` most-rated movies (by count in `train.parquet`) of the chosen genre. For each event, with probability `--drift-fraction` (random generator seeded with 42) replace the movie with a random one from the pool and the rating with a random choice of 4.0, 4.5, 5.0; otherwise send the event unchanged. The result is a clear shift toward one genre with high ratings.
5. Hold the requested rate (use a monotonic clock schedule, not a fixed sleep).
6. Each request is retried up to 3 times (0.5 s apart). A request that still fails counts as an error and the run continues.
7. At the end print: events sent, recommend calls, feedback calls, errors, elapsed seconds, achieved events per second, and how many responses each `X-Served-By` value gave. Exit code 1 if more than 5% of requests failed, otherwise 0.

## Docker and Makefile
- Service `simulator`: build `docker/simulator.Dockerfile` (copies `src/__init__.py`, `src/common`, `src/simulator`; start command `python -m src.simulator.simulate`), image `movie-recsys-simulator:dev`, **profile `tools`**, `env_file: .env`, `API_URL=http://nginx:80`, mounts `./data/processed:/app/data/processed:ro` and `./params.yaml:/app/params.yaml:ro`, depends on `nginx` healthy.
- `make simulate` runs `docker compose run --rm simulator python -m src.simulator.simulate --mode normal` and adds `--limit $(LIMIT)` when `LIMIT` is set. `make simulate-drift` is the same with `--mode drift`.
- `make reset-feedback` runs `docker compose run --rm trainer python -m src.train.admin reset-feedback`. The command deletes rows with `source='simulator'` from `feedback` and `recommendation_logs` and **all** rows of `drift_reports` and `retrain_runs`. Real (`web`) rows and accounts are kept.
- Build the simulator image with `make build`.

## Commands and expected output

```bash
make simulate LIMIT=300      # about 15 seconds at the default 20 events per second
```
Summary shows 300 events, 300 recommend calls, 300 feedback calls, 0 errors, and both `api1` and `api2` in the `X-Served-By` counts.
Check the database: `docker compose exec postgres psql -U recsys -d app -c "select source, count(*) from feedback group by source;"` shows 300 rows with source `simulator`. Rate a movie in the browser, rerun the query: one `web` row appears. `make reset-feedback` removes only the simulator rows.

## CHECKPOINT 6

- [ ] `make simulate LIMIT=300` finishes with 0 errors
- [ ] `feedback` and `recommendation_logs` have rows with `source=simulator`; a rating made in the browser has `source=web`
- [ ] `make simulate-drift LIMIT=300` finishes; most new feedback rows are for movies of the drift genre with ratings of 4.0 or higher
- [ ] `make reset-feedback` deletes only the simulator rows (the browser account still logs in)

---

# STEP 7: Monitoring (Prometheus and Grafana)

**Goal:** graphs that show system health and model behavior while the simulator runs.
**Needs:** Step 6.

## Files added

```
infra/prometheus/prometheus.yml  infra/prometheus/alerts.yml
infra/grafana/provisioning/datasources/datasource.yml
infra/grafana/provisioning/dashboards/dashboards.yml
infra/grafana/dashboards/movie-recsys.json
prometheus and grafana services in docker-compose.yml
```

## Configuration (must do)

**prometheus.yml:** `scrape_interval: 5s`, `evaluation_interval: 5s`, `rule_files` pointing to `/etc/prometheus/alerts.yml`, one job `api` with static targets `api1:8000` and `api2:8000` (each API copy scraped directly, because Nginx hides `/metrics`). The `drift` job is added in Step 8.

**alerts.yml:** group `movie-recsys` with these rules:

| Alert | Condition | For |
|---|---|---|
| `ApiDown` | `up{job="api"} == 0` | 15 s |
| `HighErrorRate` | share of `http_requests_total` with `status` starting with 5 over the last minute above 0.05 | 1 m |
| `HighLatencyP95` | `histogram_quantile(0.95, ...)` of `http_request_duration_seconds` over 1 m above 0.5 | 1 m |

Alerts are only visible on Prometheus's Alerts page. No Alertmanager.

**Grafana provisioning:** datasource named `Prometheus`, type `prometheus`, URL `http://prometheus:9090`, `uid: prometheus`, default. Dashboard provider reads JSON files from `/var/lib/grafana/dashboards`.

**Compose services:**
- `prometheus`: image `prom/prometheus:v2.53.0`, port `9090:9090`, mounts `prometheus.yml` and `alerts.yml` into `/etc/prometheus/` read-only.
- `grafana`: image `grafana/grafana:10.4.2`, port `3000:3000`, mounts the `provisioning` folder to `/etc/grafana/provisioning` and the `dashboards` folder to `/var/lib/grafana/dashboards` (read-only), volume `grafana-data:/var/lib/grafana`. Environment: `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}`, `GF_AUTH_ANONYMOUS_ENABLED=true`, `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` (anyone can view without login; admin login still needs the password).
- Add both to the `make up` service list.

## The dashboard (movie-recsys.json): `uid: movie-recsys`, refresh 5 s, default range last 15 minutes

| # | Panel | Query idea |
|---|---|---|
| 1 | Requests per second per replica | `sum by (instance) (rate(http_requests_total[30s]))` |
| 2 | p95 latency, and error rate | p95 from the duration histogram; `sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m]))` |
| 3 | Traffic by source (real vs simulator) | `sum by (source) (rate(http_requests_total[30s]))` |
| 4 | Recommendation kinds (stacked) | `sum by (kind) (rate(recommendations_served_total[1m]))` |
| 5 | Feedback per minute by source, and average rating | `sum by (source) (increase(feedback_total[1m]))`; `sum(rate(feedback_rating_sum_total[5m])) / sum(rate(feedback_total[5m]))` |
| 6 | Model version per replica | `model_version` |
| 7 | Drift (empty until Step 8) | `drift_share`, `drift_detected`, `drift_window_rows`, `live_hit_rate` |

Every panel uses the datasource uid `prometheus`. The dashboard must load without any click in the Grafana UI.

## Verify

```bash
make up
make simulate LIMIT=600      # in a second terminal, while watching Grafana
```
Open http://localhost:3000 (dashboard "movie-recsys"), http://localhost:9090/targets and http://localhost:9090/alerts.

## CHECKPOINT 7

- [ ] Prometheus targets page shows `api1` and `api2` **UP**
- [ ] Grafana shows the dashboard without login and without manual setup; panels 1 to 6 show live data during a simulation; panel 1 shows two lines (the replicas alternating)
- [ ] Traffic by source shows `simulator`; a click in the browser adds a `web` line
- [ ] `docker compose stop api1` makes `ApiDown` go pending then firing on the Alerts page; starting it clears the alert
- [ ] Panel 6 shows model version 1 for both replicas

---

# STEP 8: Drift detection

**Goal:** notice when incoming ratings no longer look like the ones the model learned from, save a report, and raise a flag that retraining reads.
**Needs:** Step 7.

## Files added

```
src/drift/__init__.py  src/drift/drift_check.py
docker/drift.Dockerfile  (requirements/drift.txt from Step 4)
tests/test_drift_logic.py
drift service in docker-compose.yml, drift job and DriftDetected alert in Prometheus
```

## What is compared

| Column | Meaning |
|---|---|
| `rating` | rating value (numeric) |
| `main_genre` | first genre of the rated movie, `none` if unknown (categorical) |
| `item_popularity` | the movie's `n_ratings` in the champion's training data, 0 if unknown (numeric) |

- **Reference** = `reference.parquet` from the champion bundle, sampled to at most `drift.reference_sample` rows (fixed seed). It follows the champion, so a promotion switches the reference automatically.
- **Current** = rows in `feedback` with `created_at` later than the champion bundle's `meta.json created_at` (all sources), mapped to the same three columns using the bundle's movies and popularity tables.
- **Decision:** `drift_detected = (rows >= drift.min_rows) and (share of drifted columns >= drift.share_threshold)`. With three columns the share can only be 0, 0.33, 0.67 or 1.

## src/drift/drift_check.py (`python -m src.drift.drift_check [--once]`)

Functions (pure ones are unit-tested):

| Function | Behavior |
|---|---|
| `build_current(window, bundle)` | maps a DataFrame of feedback rows to the three drift columns |
| `decide(share, n_rows, params)` | the decision rule above |
| `live_hit_rate(window, logs)` | among window feedback rows with `rating >= serving.liked_threshold` that have an earlier `/recommend` log row (`kind` `popular` or `personalized`) for the same user, the share whose `movie_id` was in the most recent such list; `None` if there are no such rows |
| `run_check(...)` | one full check (below) |
| `main()` | `--once`: one check, print the result, exit. Otherwise start the Prometheus metrics server on port 8001 and run `run_check` every `drift.interval_seconds`, never crashing on a failed check (log and continue) |

`run_check` steps:
1. Find the champion version. If none, log it and return.
2. Load that bundle (cache it until the champion version changes).
3. Select the window; count rows per source.
4. If rows `< drift.min_rows`: insert a `drift_reports` row with `drift_share` empty, `drift_detected` false and `note` "insufficient rows"; update gauges; return.
5. Run **Evidently** (legacy 0.4.x API: `Report` with `DataDriftPreset`, `ColumnMapping` with the numeric and categorical columns above, library default statistical tests). Read the number of drifted columns from the report's `as_dict()` result and compute `share = drifted / 3`. Verify the dictionary keys against the installed version before relying on them.
6. Save `reports/drift/drift_<UTC timestamp>_v<version>.html` and a `.json` summary, copy the HTML to `reports/drift/latest.html`, and keep only the newest `drift.keep_reports` reports.
7. Compute `live_hit_rate`, insert a `drift_reports` row, and update the gauges `drift_share`, `drift_detected`, `drift_window_rows`, `live_hit_rate`, `drift_rows_by_source{source}`, `drift_last_check_timestamp`.

## Docker and Prometheus
- Dockerfile copies `src/__init__.py`, `src/common`, `src/recsys`, `src/drift`; image `movie-recsys-drift:dev`; default command `python -m src.drift.drift_check`.
- Compose service `drift`: `env_file: .env`, mounts `./params.yaml` and `./reports:/app/reports`, depends on `postgres` and `mlflow` healthy, internal port 8001 only. Add it to `make up`. `make drift-check` runs `docker compose run --rm drift python -m src.drift.drift_check --once`.
- `prometheus.yml`: add job `drift` with target `drift:8001`. `alerts.yml`: add `DriftDetected` (`drift_detected == 1`, for 0 s).

## Tests (tests/test_drift_logic.py)
Decision table (share and rows around both thresholds); `build_current` takes the first genre and uses `none` and 0 for unknown movies; rows older than the model's `created_at` are excluded from the window; `live_hit_rate` on a hand-made example (a hit, a miss, and a user with no earlier log).

## Calibration (part of this step, tune `params.yaml`, not the code)

```bash
make reset-feedback
make simulate LIMIT=1500           # normal mode
make drift-check                   # note drift_share; it should stay below the threshold
make reset-feedback
make simulate-drift LIMIT=1500     # drift mode
make drift-check                   # must report drift_detected true
```
If normal mode already reports drift (the newest 10% of MovieLens is naturally a bit different), raise `drift.share_threshold` to 0.67 and repeat. If drift mode does not reach the threshold, raise `simulator.drift_fraction`.

## CHECKPOINT 8

- [ ] `make test` passes
- [ ] Normal-mode run: no drift flagged. Drift-mode run: drift flagged within about two minutes (one loop) and the Grafana drift panel shows `drift_detected = 1`
- [ ] `reports/drift/latest.html` opens in a browser and shows the three columns
- [ ] `drift_reports` has rows with `rows_by_source`; `DriftDetected` fires on the Prometheus Alerts page
- [ ] Prometheus targets page shows the `drift` job UP

---

# STEP 9: Retraining, promotion and rollback

**Goal:** when drift is flagged, train a new model (v2) on old data plus feedback, compare it with v1, promote it only if it is better, and make both API copies switch to it with no downtime.
**Needs:** Step 8.

## Files added

```
src/train/retrain.py
src/train/admin.py            (extended: reset-feedback, rollback, set-champion, reload)
tests/test_retrain_logic.py
retrainer service in docker-compose.yml
```
The retrainer **reuses the trainer image** (`movie-recsys-trainer:dev`).

## src/train/admin.py (`python -m src.train.admin <command>`)

| Command | Behavior |
|---|---|
| `check-champion` | (Step 4) exit 1 with a message if there is no champion |
| `reset-feedback` | (Step 6) delete simulator rows, clear `drift_reports` and `retrain_runs` |
| `rollback` | exit 1 with a message if there is no `previous` alias. Otherwise **swap** the aliases (`champion` goes to the old `previous` version, `previous` goes to the old champion), then call `reload_replicas` and print each result |
| `set-champion --version N` | point `champion` at version N, then reload replicas |
| `reload` | call `reload_replicas` and print each result |

`reload_replicas` (from `src/common/reload.py`, Step 3) needs `ADMIN_KEY` and the targets. In Compose, set `API_RELOAD_TARGETS=http://api1:8000,http://api2:8000` as environment on the `trainer` and `retrainer` services.

## src/train/retrain.py (`python -m src.train.retrain [--loop | --auto | --once | --force]`)

| Mode | Meaning |
|---|---|
| `--loop` | forever: run `--auto` logic every `retrain.interval_seconds`. Never crash on one failed run (log, continue). Compose default. |
| `--auto` | one pass with the **trigger rules**; used by the Kubernetes CronJob |
| `--once` | one retrain now, skipping the trigger rules, quality gate still applies. Trigger recorded as `manual`. |
| `--force` | like `--once` but also skips the quality gate. Recorded as `forced`. |

**Pure functions (unit-tested):**

| Function | Behavior |
|---|---|
| `should_trigger(latest_drift_row, champion_version, new_rows, params)` | true only if the latest `drift_reports` row belongs to this champion version, has `drift_detected` true, and `new_rows >= retrain.min_new_rows`. `new_rows` = feedback rows created after the end of the last finished retrain run (any decision), or after the champion's `created_at` if there is none. This stops endless retraining after a rejection. |
| `merge_training_data(train_df, feedback_df)` | combine both (columns `userId`, `movieId`, `rating`, `timestamp`); per (user, movie) the latest rating wins |
| `split_holdout(feedback_df, fraction)` | by time: the oldest `1 - fraction` of feedback rows, and the newest `fraction` as holdout |
| `decide_promotion(v1_holdout, v2_holdout, v1_test, v2_test, params)` | returns (promote, reason). Promote only if `(v2_holdout - v1_holdout) / max(v1_holdout, 1e-9) >= retrain.min_improvement` **and** `(v1_test - v2_test) / max(v1_test, 1e-9) <= retrain.max_test_drop`. All four numbers are NDCG@K. |

**One retrain, step by step (`run_retrain(trigger)`):**
1. Insert a `retrain_runs` row with status `running`. If another `running` row started less than 30 minutes ago exists, print "already running" and stop.
2. Load the champion bundle (v1) and its version number.
3. Read **all** `feedback` rows (both sources) into a DataFrame (`timestamp` = created_at in unix seconds); the latest rating per (user, movie) wins.
4. `split_holdout`. If the holdout has no rating of `eval.relevant_threshold` or higher, finish with decision `rejected` and note "not enough feedback".
5. `training_df = merge_training_data(train.parquet, older feedback)`. Train v2 with the same model settings as v1 and build its bundle (`feedback_rows` = number of older feedback rows, `parent_version` = v1).
6. Evaluate **v1 and v2** with the same `evaluate` function (Step 2): on the holdout with `seen_df = training_df`; and on `test.parquet` with `seen_df = train.parquet`.
7. `decide_promotion` (skipped when `--force`).
8. Log one MLflow run (experiment `movie-recsys`, tags `stage=retrain` and the trigger): params, metrics `v1_holdout_ndcg_at_10`, `v2_holdout_ndcg_at_10`, `v1_test_ndcg_at_10`, `v2_test_ndcg_at_10` and RMSE values, the v2 bundle, `feedback_snapshot.parquet` (all feedback used, including the holdout; this is the data versioning for runtime data), and `params.yaml`. Register v2 as a new model version.
9. **Promote** (or forced): set alias `previous` to v1's version, then `champion` to v2's version, then call `reload_replicas`. **Reject:** set the model version tag `decision=rejected` and leave aliases alone.
10. Update the `retrain_runs` row (status, scores, decision, versions, note). On any exception set status `failed` with the error in `note`.

After a promotion, Step 8 automatically uses v2's reference data and starts its new-data window from v2's `created_at`.

**Reload with no downtime:** each API copy builds the new model first and swaps it in one step (Step 3). Replicas are reloaded one after the other, so the other copy answers while one reloads, and Nginx failover covers any gap.

## Compose and Makefile
- Service `retrainer`: image `movie-recsys-trainer:dev` (no build), command `python -m src.train.retrain --loop`, `env_file: .env`, `API_RELOAD_TARGETS` as above, mounts `./data/processed:/app/data/processed:ro` and `./params.yaml:/app/params.yaml:ro`, depends on `postgres` and `mlflow` healthy. It must also wait quietly (log and retry) if there is no champion yet. Add it to `make up`.
- `make retrain` = `docker compose run --rm retrainer python -m src.train.retrain --once`; `make retrain-force` uses `--force`; `make rollback` = `docker compose run --rm trainer python -m src.train.admin rollback`; `make demo-reset` = `make reset-feedback` then `docker compose run --rm trainer python -m src.train.admin set-champion --version 1`.

## Tests (tests/test_retrain_logic.py)
`should_trigger` truth table (drift off, drift on but too few rows, drift on and enough rows, drift row from an older champion); `split_holdout` puts the newest 20% in the holdout; `decide_promotion` at the exact thresholds (1% gain passes, 0.9% fails; 10% test drop passes, 11% fails); `merge_training_data` keeps the latest rating; `reload_replicas` still calls the second target when the first one fails (fake HTTP).

## Verify

```bash
make reset-feedback
make simulate-drift LIMIT=2000                     # wait until Grafana shows drift_detected = 1
( while true; do curl -s localhost:8080/health; echo; sleep 0.3; done ) &   # watch model_version and status
make retrain-force                                  # guaranteed swap for the demo
```
Then try the honest path: `make demo-reset`, `make simulate-drift LIMIT=2000`, wait for the automatic retrainer (no command), and read the decision in `retrain_runs` or MLflow.

## CHECKPOINT 9

- [ ] `make test` passes
- [ ] `make retrain-force`: a new MLflow model version appears, `champion` moves to it, `previous` points at version 1, the health loop shows `model_version` change from 1 to 2 with **no non-200 answers**, and Grafana panel 6 shows both replicas on 2
- [ ] `retrain_runs` has a row with decision `forced` and the four NDCG scores
- [ ] `make rollback` returns both replicas to version 1; `make demo-reset` clears the demo state
- [ ] Automatic path: after drift-mode traffic the retrainer starts by itself and records `promoted` or `rejected` with the scores. A rejection is a valid outcome (tune `retrain.min_improvement` or `simulator.drift_fraction` in `params.yaml` only if you want the demo to promote).
- [ ] After a rejection the retrainer does **not** retrain again until 500 more feedback rows arrive

---

# STEP 10: CI/CD with GitHub Actions

**Goal:** every push is checked automatically; code on `main` gets its images built and published; a broken change cannot reach `main`.
**Needs:** Steps 1 to 9. A GitHub repository (assumed **public**; on a private repo make the smoke test lighter and check the free CI minutes).

## Files added

```
.github/workflows/ci.yml  .github/workflows/cd.yml
scripts/smoke_test.sh
README.md
```

## scripts/smoke_test.sh (`bash scripts/smoke_test.sh [BASE_URL]`, default `http://localhost:8080`; needs `curl` and `jq`)
Stop at the first failure with a clear message and a non-zero exit code; print a PASS line per check:
1. Wait up to 120 s for `GET /health` to return 200.
2. `GET /metrics` and `POST /reload` through the gateway return 404.
3. `GET /popular` returns at least 10 movies.
4. Sign up a random user (`smoke_<random>`, password `smoke-pass-1`); `GET /me` returns a `user_id` of 100000 or more.
5. `GET /recommend` returns kind `popular`.
6. Rate three movies from the popular list (5, 4, 4 stars).
7. `GET /recommend` now returns kind `personalized` and none of the rated movie ids.
8. `GET /because` returns at least one row.
9. Log out, log in again with the same credentials, `GET /me` works.
10. Responses carry the `X-Served-By` header.

## .github/workflows/ci.yml (triggers: `pull_request`, and `push` to `main`; permissions `contents: read`)

| Job | Steps |
|---|---|
| `lint-test` | `actions/checkout@v4`; `actions/setup-python@v5` (3.11, pip cache); `pip install -r requirements.txt`; `ruff check .`; `pytest -q`; `dvc repro`; fail if `git diff --exit-code dvc.lock metrics/preprocess.json` shows a change (tell the author to run `dvc repro` and commit). If the hashes differ only because of the environment, report it to the team instead of weakening the check. |
| `build` | `docker compose build` (all images) |
| `smoke` | needs `lint-test`; `cp .env.example .env`; install requirements; `make data`; `make build`; `make infra`; `make train`; `make up`; `bash scripts/smoke_test.sh`; on failure print `docker compose logs`; always run `make down`. Timeout 30 minutes. |

## .github/workflows/cd.yml (trigger: `push` to `main`; permissions `contents: read`, `packages: write`)
A matrix over the services `api`, `trainer`, `simulator`, `drift`, `mlflow`. Each: checkout, `docker/setup-buildx-action@v3`, `docker/login-action@v3` to `ghcr.io` with `github.actor` and `GITHUB_TOKEN`, then `docker/build-push-action@v5` with the matching Dockerfile and tags `ghcr.io/<owner lowercase>/movie-recsys-<service>:<commit sha>` and `:latest`. No other secrets are needed.

## README.md
Short: what the project is, the demo in 6 commands (`make data`, `make build`, `make infra`, `make train`, `make up`, then open http://localhost:8080), the links of Step 12, and the team split.

## One-time GitHub settings (cannot be done from code; write them in the README)
1. Settings, Branches, add a rule for `main`: require a pull request before merging, require status checks `lint-test`, `build` and `smoke`, block direct pushes.
2. After the first publish, Packages: the images are private by default; make them public if the repo is public.
3. Team workflow: branch `feature/<step>-<name>`, open a pull request, wait for green checks, review, merge.

## CHECKPOINT 10

- [ ] `pre-commit run --all-files` passes
- [ ] A pull request with a trivial change shows `lint-test`, `build` and `smoke` all green
- [ ] A throwaway branch with a deliberately failing test shows a red check and cannot be merged
- [ ] After merge to `main`, `cd.yml` publishes the five images (visible under the repository's Packages) with a commit-sha tag and `latest`
- [ ] `bash scripts/smoke_test.sh` passes against a locally running stack

---

# STEP 11: Kubernetes (kind)

**Goal:** run the same system on a local Kubernetes cluster, with the same URLs and the same demo as Compose.
**Needs:** Steps 1 to 10; `kubectl` and `kind` installed. **Run `make down` first** (Compose and Kubernetes share the host ports 8080, 5000, 9090, 3000). Docker Desktop memory of at least 6 GB is an estimate; raise it if pods are killed.

## Files added

```
k8s/kind-config.yaml
k8s/infra/  kustomization.yaml  namespace.yaml  postgres.yaml  mlflow.yaml
k8s/app/    kustomization.yaml  api.yaml  nginx.yaml  drift.yaml  retrainer-cronjob.yaml  prometheus.yaml  grafana.yaml
k8s/jobs/   job-trainer-image.yaml.tpl  job-simulator.yaml.tpl
infra/nginx/nginx.k8s.conf  infra/prometheus/prometheus.k8s.yml
.github/workflows/k8s-deploy.yml
```
Plain YAML only (no Helm). Everything lives in the namespace `movie-recsys`; every manifest sets it. Images use the tags built by Compose (`movie-recsys-<service>:dev`) with `imagePullPolicy: IfNotPresent`.

## Cluster config (k8s/kind-config.yaml, one control-plane node, cluster name `movie-recsys`)
- `extraPortMappings` (TCP): container 30080 to host 8080 (Nginx), 30500 to 5000 (MLflow), 30090 to 9090 (Prometheus), 30300 to 3000 (Grafana).
- `extraMounts`: `__PROJECT_DIR__/data/processed` to `/mnt/data/processed` (read-only) and `__PROJECT_DIR__/reports` to `/mnt/reports`. The Makefile replaces `__PROJECT_DIR__` with the absolute project path using `sed` and creates `reports/drift` first. This mount only works on a local machine; say so in the report.

## Manifests (must do)

| File | Objects and details |
|---|---|
| `infra/postgres.yaml` | PVC (1 Gi), Deployment (1 replica, strategy Recreate, image `postgres:16`, env from Secret `app-secrets`, `PGDATA=/var/lib/postgresql/data/pgdata`, init ConfigMap `postgres-init` mounted at `/docker-entrypoint-initdb.d`, readiness `pg_isready`), Service `postgres` (5432) |
| `infra/mlflow.yaml` | PVC (2 Gi) for artifacts, Deployment (image `movie-recsys-mlflow:dev`, same command as Compose with host `postgres`, env from `app-secrets`, readiness HTTP `/health`), Service `mlflow` type NodePort (5000, nodePort 30500) |
| `app/api.yaml` | Deployment `api` with **2 replicas**; env from `app-secrets`; `REPLICA_NAME` from the pod name (downward API); ConfigMap `params` mounted at `/app/params.yaml`; readiness and liveness HTTP `/health` (liveness start delay 60 s); rolling update with `maxUnavailable: 0`, `maxSurge: 1`. Service `api` (ClusterIP, 8000). Service `api-headless` (`clusterIP: None`, 8000) listing every pod address. |
| `app/nginx.yaml` | Deployment (`nginx:1.27-alpine`) with ConfigMap `nginx-conf` mounted at `/etc/nginx/nginx.conf`; Service `nginx` NodePort (80, nodePort 30080) |
| `app/drift.yaml` | Deployment (image `movie-recsys-drift:dev`, env from `app-secrets`, `params` ConfigMap, hostPath `/mnt/reports` mounted at `/app/reports`), Service `drift` (8001) |
| `app/retrainer-cronjob.yaml` | CronJob `retrainer`, schedule `* * * * *`, `concurrencyPolicy: Forbid`, history limit 1, `activeDeadlineSeconds: 900`, `restartPolicy: Never`; image `movie-recsys-trainer:dev`, command `python -m src.train.retrain --auto`, env from `app-secrets` plus `API_HEADLESS_HOST=api-headless:8000`, hostPath `/mnt/data/processed` (type Directory) at `/app/data/processed`, `params` ConfigMap |
| `app/prometheus.yaml` | Deployment (`prom/prometheus:v2.53.0`) with ConfigMap `prometheus-config` (keys `prometheus.yml`, `alerts.yml`); Service `prometheus` NodePort (9090, nodePort 30090) |
| `app/grafana.yaml` | Deployment (`grafana/grafana:10.4.2`), env `GF_SECURITY_ADMIN_PASSWORD` from the Secret key `GRAFANA_ADMIN_PASSWORD` plus the two anonymous-viewer variables; ConfigMaps `grafana-datasource`, `grafana-dashboard-provider`, `grafana-dashboards` mounted where Compose mounts them; Service `grafana` NodePort (3000, nodePort 30300) |
| `jobs/job-trainer-image.yaml.tpl` | Job template with placeholders `__NAME__` and `__COMMAND__` (run with `sh -c`); trainer image; env from `app-secrets` plus `API_HEADLESS_HOST`; data hostPath and `params`; `backoffLimit: 0` |
| `jobs/job-simulator.yaml.tpl` | Job template with `__NAME__`, `__MODE__`, `__LIMIT_ARG__`; simulator image; `API_URL=http://nginx:80`; `SIMULATOR_KEY` from the Secret; data hostPath and `params`; `backoffLimit: 0` |

`app/kustomization.yaml` and `infra/kustomization.yaml` list their files. The two job templates are **not** in a kustomization (they are filled in by the Makefile).

**Config differences from Compose:**
- `infra/nginx/nginx.k8s.conf`: identical to `nginx.conf` except the upstream has the single server `api:8000`.
- `infra/prometheus/prometheus.k8s.yml`: job `api` uses `dns_sd_configs` (type `A`, names `api-headless`, port 8000) so every API pod is scraped; job `drift` has the static target `drift:8001`; same `rule_files`.
- No code change is needed: `X-Served-By` already comes from `REPLICA_NAME`, and the retrainer finds every API pod through `API_HEADLESS_HOST`.

## Makefile targets

| Target | Behavior |
|---|---|
| `make k8s-up` | 1) check `data/processed/train.parquet` and `.env`; 2) `docker compose build`; 3) create the kind cluster `movie-recsys` if missing; 4) `kind load docker-image` for the five images; 5) apply `k8s/infra/namespace.yaml`; 6) create or update the Secret `app-secrets` from `.env` (`kubectl create secret generic --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -`); 7) create or update the ConfigMaps from the repo files the same way: `params` (params.yaml), `postgres-init`, `nginx-conf` (from `nginx.k8s.conf`), `prometheus-config` (from `prometheus.k8s.yml` and `alerts.yml`), `grafana-datasource`, `grafana-dashboard-provider`, `grafana-dashboards`; 8) `kubectl apply -k k8s/infra` and wait for `postgres` and `mlflow` (timeout 300 s); 9) delete any old `trainer` Job, create it from the template with command `python -m src.train.train --if-missing`, wait for completion (timeout 600 s); 10) `kubectl apply -k k8s/app`, wait for rollout of `api`, `nginx`, `drift`, `prometheus`, `grafana`; 11) print the four URLs |
| `make k8s-down` | `kind delete cluster --name movie-recsys` |
| `make k8s-status` | `kubectl -n movie-recsys get pods,svc,jobs,cronjobs` |
| `make k8s-simulate` / `make k8s-simulate-drift` | delete Job `simulator` if present, fill the simulator template (mode, and `--limit $(LIMIT)` when set), apply, follow its logs until it finishes |
| `make k8s-retrain` | Job `retrain-manual` from the trainer template with command `python -m src.train.retrain --once` (`--force` when `FORCE=1`); delete the old one first; follow logs |
| `make k8s-rollback` | Job `rollback` with command `python -m src.train.admin rollback` |
| `make k8s-demo-reset` | Job running `reset-feedback`, then a Job running `set-champion --version 1` |
| `make k8s-rollout` | `docker compose build api1`, `kind load docker-image movie-recsys-api:dev --name movie-recsys`, `kubectl rollout restart deployment/api`, `kubectl rollout status deployment/api` |

## .github/workflows/k8s-deploy.yml (trigger: `push` to `main`; timeout 40 minutes)
Checkout; Python 3.11; `pip install -r requirements.txt`; `cp .env.example .env`; `make data`; make sure `kind` is available (`helm/kind-action@v1` with `install_only: true`); `make k8s-up`; `bash scripts/smoke_test.sh http://localhost:8080`; on failure print `kubectl -n movie-recsys get pods`, `describe pods` and logs; always `make k8s-down`.

## Verify

```bash
make down && make k8s-up
make k8s-status          # postgres, mlflow, 2 api pods, nginx, drift, prometheus, grafana Running; trainer Job Completed; CronJob retrainer present
bash scripts/smoke_test.sh
make k8s-simulate-drift LIMIT=2000
make k8s-retrain FORCE=1
( while true; do curl -s -o /dev/null -w "%{http_code} " localhost:8080/health; sleep 0.3; done ) &
make k8s-rollout          # only 200s must print
```

## CHECKPOINT 11

- [ ] `make k8s-up` finishes without manual steps; the app works at http://localhost:8080 with the same behavior as Compose
- [ ] `scripts/smoke_test.sh` passes against the cluster
- [ ] Prometheus (http://localhost:9090/targets) shows **both API pods** and `drift` UP; Grafana shows live data; `X-Served-By` alternates between the two pod names
- [ ] Drift-mode traffic raises the drift flag; `make k8s-retrain FORCE=1` promotes v2 and both pods report version 2 with no failed requests
- [ ] `make k8s-rollout` replaces the API pods one at a time with no failed requests
- [ ] The CronJob `retrainer` runs every minute (`kubectl -n movie-recsys get jobs`) and records "no trigger" runs without errors
- [ ] `make k8s-down` removes the cluster; `make up` works again afterwards

---

# STEP 12: Demo script, troubleshooting, ownership, report notes

## 12.1 Demo script (Compose first, then Kubernetes)

Preparation: `make down`, then `make data build infra train up` and `make demo-reset`. Open tabs: http://localhost:8080 (app), http://localhost:3000 (Grafana dashboard), http://localhost:5000 (MLflow), http://localhost:9090/alerts (Prometheus).

| # | Do | Point at |
|---|---|---|
| 1 | Open the app as a guest | Cold start: popular movies only |
| 2 | Sign up, rate 3 movies (one with 5 stars) | "Because you liked X" and "Recommended for you" appear; history is stored on the server |
| 3 | Run `curl -si localhost:8080/health \| grep -i x-served-by` a few times, then `docker compose stop api1`, refresh the app, `docker compose start api1` | Nginx load balancing and failover |
| 4 | `make simulate LIMIT=1500` | Grafana traffic panels: replicas alternate, real vs simulator source, popular vs personalized share |
| 5 | `make simulate-drift LIMIT=2000` | Drift panel flips to detected; open `reports/drift/latest.html`; `DriftDetected` alert on Prometheus |
| 6 | Watch the retrainer decide (or run `make retrain-force`) while a health loop runs (`while true; do curl -s localhost:8080/health; echo; sleep 0.3; done`) | New version in MLflow with scores and the feedback snapshot; `model_version` changes 1 to 2 with no failed request; Grafana panel 6 |
| 7 | `make rollback` | Back to version 1 in one command |
| 8 | Show a pull request with green checks and the published images under Packages | CI/CD |
| 9 | `make down && make k8s-up`, `make k8s-status`, repeat steps 2, 5 and 6 with the `k8s-` targets, then `make k8s-rollout` | Same system on Kubernetes, zero-downtime release |

Reset between runs: `make demo-reset` (Compose) or `make k8s-demo-reset` (Kubernetes). Full wipe: `make clean`, or `make k8s-down`.

## 12.2 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "port is already allocated" (8080, 5000, 3000, 9090) | Compose and Kubernetes running together, or another program | `make down` and `make k8s-down`; find the other program |
| Containers exit with code 137, pods OOMKilled | Docker memory too low | Raise Docker Desktop memory (6 to 8 GB) |
| Windows: mounted folders empty or very slow | Project is under `/mnt/c` | Move it to `~/movie-recsys` inside WSL |
| `make up` stops with "No champion model" | Trainer not run | `make train` |
| API will not start, MLflow errors | MLflow not ready, or client and server versions differ | Same MLflow pin in every requirements file; `docker compose logs mlflow` |
| Nginx: "host not found in upstream" | API containers not up yet | `make up` (waits for healthy); then `docker compose restart nginx` |
| `dvc repro` download fails | GroupLens unreachable or URL changed | Update `data.url` in `params.yaml` from the GroupLens page |
| CI: `dvc.lock` differs | Change not committed | Run `dvc repro`, commit `dvc.lock` and `metrics/preprocess.json` |
| Drift never fires | Fewer than 500 new rows, or threshold too high | Simulate more (`LIMIT`), tune `drift.share_threshold` and `simulator.drift_fraction` |
| Drift fires in normal mode | Threshold too low | Raise `drift.share_threshold` to 0.67 |
| Retrain always rejects | v2 not better on the holdout | Valid outcome; use `make retrain-force` for the demo, or tune `retrain.min_improvement` |
| Grafana panels empty | Prometheus targets down, or wrong datasource uid | Check http://localhost:9090/targets; datasource uid must be `prometheus` |
| `reports/` files owned by root (Linux) | Container wrote them | `sudo chown -R $USER reports` |
| `ErrImagePull` in Kubernetes | Image not loaded into kind | `make k8s-up` loads them, or run `kind load docker-image <image> --name movie-recsys` |
| Evidently import errors | Wrong version installed | Keep Evidently on the 0.4.x line (see Step 0.3) |
| `pip install` conflict | Pins incompatible on this machine | Fix per rule 3 of section 0 |

## 12.3 Team ownership and handoffs

| Member | Steps | Area |
|---|---|---|
| 1 | 1, 2 | Data and ML: ingestion, validation, DVC, model, MLflow |
| 2 | 3, 4, 5, 6 | Serving and infra: FastAPI and web page, Docker, Compose, Nginx, simulator |
| 3 | 10, 11 | CI/CD and Kubernetes |
| 4 | 7, 8, 9 | Monitoring, drift, retraining |

Step 0 is done together. Handoff contracts: member 1 to 2 and 4 is the **model bundle** (Step 0.5 A); member 2 to 4 is the **database tables and metric names** (Step 0.5 C and E); members 2 and 4 to 3 are the **images and `scripts/smoke_test.sh`**. A step is handed over only after its checkpoint passes.

## 12.4 Notes for the report

**Where each course topic is shown:** GitHub and pre-commit (Steps 0 and 10), Docker (4), pipelines and data versioning (1), MLflow and model versioning (2, 9), serving (3), Nginx (5), Prometheus and monitoring (7), drift detection (8), retraining (9), CI/CD (10), Kubernetes (11).

**Decisions with reasons:** matrix factorization because it learns parameters and gives real metrics and retraining behavior to track; time split to copy "train on the past, serve the future"; small dataset so CI and laptops stay fast; plain NumPy ALS because `scikit-surprise` often fails to build on new Python versions; two named API copies so the retrainer can reload each one; database-backed sessions so any replica can verify a login; data mounted from the project folder (not baked into images) so retraining needs no rebuild.

**Limits we accepted (state them honestly):**
- Some test and stream users have no training history (cold start), which is intended.
- Login is demo-grade: no email, no reset, no rate limit, no HTTPS.
- The v2 holdout rows are not used to train v2; the next retrain includes them.
- Drift in the demo is generated by the simulator; a real deployment would need real tuning of the thresholds.
- Mounting the project's data folder works only on one local machine.
- The Kubernetes cluster is a single local node without autoscaling or Ingress.

**Dataset citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.

## 12.5 Command cheat sheet

```
make data build infra train up          # first start (Compose)
make simulate | simulate-drift          # traffic (LIMIT=2000 for short runs)
make drift-check | retrain | retrain-force | rollback | demo-reset
make down | clean                       # stop | stop and delete volumes
make k8s-up | k8s-status | k8s-simulate | k8s-simulate-drift | k8s-retrain | k8s-rollback | k8s-demo-reset | k8s-rollout | k8s-down
make test | lint
```

## FINAL CHECKPOINT

- [ ] Steps 0 to 11 checkpoints all pass on a **fresh clone** by a second teammate, following only this file
- [ ] The demo script in 12.1 runs end to end on Compose and on Kubernetes
- [ ] `docs/DEVIATIONS.md` lists every change made to this spec
