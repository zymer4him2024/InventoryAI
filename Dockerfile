FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libzbar0 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup --gid=1001 \
    && useradd -r -g appgroup --uid=1001 --home-dir=/app --shell=/sbin/nologin appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000 8002 8004
