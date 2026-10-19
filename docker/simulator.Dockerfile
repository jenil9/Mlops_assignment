FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements/simulator.txt /app/requirements/simulator.txt
RUN pip install --no-cache-dir -r /app/requirements/simulator.txt

COPY src/__init__.py /app/src/
COPY src/common /app/src/common
COPY src/simulator /app/src/simulator

CMD ["python", "-m", "src.simulator.simulate"]
