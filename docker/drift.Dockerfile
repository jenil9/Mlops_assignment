FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements/drift.txt /app/requirements/drift.txt
RUN pip install --no-cache-dir -r /app/requirements/drift.txt

COPY src/__init__.py /app/src/
COPY src/common /app/src/common
COPY src/recsys /app/src/recsys
COPY src/drift /app/src/drift

CMD ["python", "-m", "src.drift.drift_check"]
