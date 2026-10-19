FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements/trainer.txt /app/requirements/trainer.txt
RUN pip install --no-cache-dir -r /app/requirements/trainer.txt

COPY src/__init__.py /app/src/
COPY src/common /app/src/common
COPY src/recsys /app/src/recsys
COPY src/train /app/src/train

CMD ["python", "-m", "src.train.train", "--if-missing"]
