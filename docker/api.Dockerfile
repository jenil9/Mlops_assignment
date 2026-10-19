FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements/api.txt /app/requirements/api.txt
RUN pip install --no-cache-dir -r /app/requirements/api.txt

COPY src/__init__.py /app/src/
COPY src/common /app/src/common
COPY src/recsys /app/src/recsys
COPY src/api /app/src/api

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
