FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py transcriber.py ./

ENV OUTPUT_DIR=/data/output
ENV HF_HOME=/data/hf-cache

RUN mkdir -p /data/output /data/hf-cache \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8140

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8140"]
