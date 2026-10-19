SHELL := /bin/bash
export PATH := $(PWD)/.venv/bin:$(PATH)
export VIRTUAL_ENV := $(PWD)/.venv

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)
RUFF := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
DVC := $(if $(wildcard .venv/bin/dvc),.venv/bin/dvc,dvc)
MLFLOW := $(if $(wildcard .venv/bin/mlflow),.venv/bin/mlflow,mlflow)
UVICORN := $(if $(wildcard .venv/bin/uvicorn),.venv/bin/uvicorn,uvicorn)

.PHONY: all data test lint mlflow-local train-local api-local build infra train train-new up down clean logs simulate simulate-drift reset-feedback drift-check retrain retrain-force rollback demo-reset k8s-up k8s-down k8s-status k8s-simulate k8s-simulate-drift k8s-retrain k8s-rollback k8s-demo-reset k8s-rollout check-data

all: test lint

check-data:
	@if [ ! -f data/processed/train.parquet ]; then \
		echo "run make data first"; \
		exit 1; \
	fi

data:
	$(DVC) repro

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

mlflow-local:
	$(MLFLOW) server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts

train-local: check-data
	DATABASE_URL=sqlite:///./local.db MLFLOW_TRACKING_URI=http://127.0.0.1:5000 $(PYTHON) -m src.train.train --if-missing

api-local:
	DATABASE_URL=sqlite:///./local.db \
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
	REPLICA_NAME=local \
	SIMULATOR_KEY=sim-secret-change-me \
	ADMIN_KEY=admin-secret-change-me \
	$(UVICORN) src.api.main:app --port 8000 --reload

build:
	docker compose build

infra:
	docker compose up -d --wait postgres mlflow

train: check-data
	docker compose run --rm trainer

train-new: check-data
	docker compose run --rm trainer python -m src.train.train

up: check-data
	docker compose run --rm trainer python -m src.train.admin check-champion
	docker compose up -d --wait api1 api2 nginx prometheus grafana drift retrainer

down:
	docker compose down

clean:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

simulate: check-data
	docker compose run --rm simulator python -m src.simulator.simulate --mode normal $(if $(LIMIT),--limit $(LIMIT),)

simulate-drift: check-data
	docker compose run --rm simulator python -m src.simulator.simulate --mode drift $(if $(LIMIT),--limit $(LIMIT),)

reset-feedback:
	docker compose run --rm trainer python -m src.train.admin reset-feedback

drift-check:
	docker compose run --rm drift python -m src.drift.drift_check --once

retrain: check-data
	docker compose run --rm retrainer python -m src.train.retrain --once

retrain-force: check-data
	docker compose run --rm retrainer python -m src.train.retrain --force

rollback:
	docker compose run --rm trainer python -m src.train.admin rollback

demo-reset: reset-feedback
	docker compose run --rm trainer python -m src.train.admin set-champion --version 1

# Kubernetes targets
k8s-up: check-data
	@if [ ! -f .env ]; then echo ".env file missing"; exit 1; fi
	docker compose build
	mkdir -p reports/drift
	PROJECT_PATH=$$(pwd); \
	sed "s|__PROJECT_DIR__|$$PROJECT_PATH|g" k8s/kind-config.yaml > /tmp/kind-config-rendered.yaml
	@if ! kind get clusters | grep -q "movie-recsys"; then \
		kind create cluster --config /tmp/kind-config-rendered.yaml; \
	fi
	kind load docker-image movie-recsys-api:dev --name movie-recsys
	kind load docker-image movie-recsys-trainer:dev --name movie-recsys
	kind load docker-image movie-recsys-simulator:dev --name movie-recsys
	kind load docker-image movie-recsys-drift:dev --name movie-recsys
	kind load docker-image movie-recsys-mlflow:dev --name movie-recsys
	kubectl apply -f k8s/infra/namespace.yaml
	kubectl create secret generic app-secrets --from-env-file=.env --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap params --from-file=params.yaml=params.yaml --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap postgres-init --from-file=init.sql=infra/postgres/init.sql --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap nginx-conf --from-file=nginx.conf=infra/nginx/nginx.k8s.conf --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap prometheus-config --from-file=prometheus.yml=infra/prometheus/prometheus.k8s.yml --from-file=alerts.yml=infra/prometheus/alerts.yml --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap grafana-datasource --from-file=datasource.yml=infra/grafana/provisioning/datasources/datasource.yml --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap grafana-dashboard-provider --from-file=dashboards.yml=infra/grafana/provisioning/dashboards/dashboards.yml --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap grafana-dashboards --from-file=movie-recsys.json=infra/grafana/dashboards/movie-recsys.json --namespace=movie-recsys --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -k k8s/infra
	kubectl -n movie-recsys wait --for=condition=ready pod -l app=postgres --timeout=300s
	kubectl -n movie-recsys wait --for=condition=ready pod -l app=mlflow --timeout=300s
	kubectl -n movie-recsys delete job trainer --ignore-not-found=true
	sed -e 's|__NAME__|trainer|g' -e 's|__COMMAND__|python -m src.train.train --if-missing|g' k8s/jobs/job-trainer-image.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys wait --for=condition=complete job/trainer --timeout=600s
	kubectl apply -k k8s/app
	kubectl -n movie-recsys rollout status deployment/api --timeout=300s
	kubectl -n movie-recsys rollout status deployment/nginx --timeout=300s
	kubectl -n movie-recsys rollout status deployment/drift --timeout=300s
	kubectl -n movie-recsys rollout status deployment/prometheus --timeout=300s
	kubectl -n movie-recsys rollout status deployment/grafana --timeout=300s
	@echo "All services up on Kubernetes!"
	@echo "Web App:    http://localhost:8080"
	@echo "MLflow:     http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000"

k8s-down:
	kind delete cluster --name movie-recsys

k8s-status:
	kubectl -n movie-recsys get pods,svc,jobs,cronjobs

k8s-simulate: check-data
	kubectl -n movie-recsys delete job simulator --ignore-not-found=true
	sed -e 's|__NAME__|simulator|g' -e 's|__MODE__|normal|g' -e 's|__LIMIT_ARG__|$(if $(LIMIT),--limit $(LIMIT),)|g' k8s/jobs/job-simulator.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys logs -f job/simulator

k8s-simulate-drift: check-data
	kubectl -n movie-recsys delete job simulator --ignore-not-found=true
	sed -e 's|__NAME__|simulator|g' -e 's|__MODE__|drift|g' -e 's|__LIMIT_ARG__|$(if $(LIMIT),--limit $(LIMIT),)|g' k8s/jobs/job-simulator.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys logs -f job/simulator

k8s-retrain: check-data
	kubectl -n movie-recsys delete job retrain-manual --ignore-not-found=true
	sed -e 's|__NAME__|retrain-manual|g' -e 's|__COMMAND__|python -m src.train.retrain $(if $(FORCE),--force,--once)|g' k8s/jobs/job-trainer-image.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys logs -f job/retrain-manual

k8s-rollback:
	kubectl -n movie-recsys delete job rollback --ignore-not-found=true
	sed -e 's|__NAME__|rollback|g' -e 's|__COMMAND__|python -m src.train.admin rollback|g' k8s/jobs/job-trainer-image.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys logs -f job/rollback

k8s-demo-reset:
	kubectl -n movie-recsys delete job reset-feedback --ignore-not-found=true
	sed -e 's|__NAME__|reset-feedback|g' -e 's|__COMMAND__|python -m src.train.admin reset-feedback|g' k8s/jobs/job-trainer-image.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys wait --for=condition=complete job/reset-feedback --timeout=120s
	kubectl -n movie-recsys delete job set-champion-v1 --ignore-not-found=true
	sed -e 's|__NAME__|set-champion-v1|g' -e 's|__COMMAND__|python -m src.train.admin set-champion --version 1|g' k8s/jobs/job-trainer-image.yaml.tpl | kubectl apply -f -
	kubectl -n movie-recsys wait --for=condition=complete job/set-champion-v1 --timeout=120s

k8s-rollout:
	docker compose build api1
	kind load docker-image movie-recsys-api:dev --name movie-recsys
	kubectl -n movie-recsys rollout restart deployment/api
	kubectl -n movie-recsys rollout status deployment/api
