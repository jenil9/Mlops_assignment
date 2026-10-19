FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements/mlflow.txt /app/requirements/mlflow.txt
RUN pip install --no-cache-dir -r /app/requirements/mlflow.txt
