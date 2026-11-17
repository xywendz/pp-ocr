FROM python:3.10-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-build.txt ./
COPY scripts/ ./scripts/

RUN pip install -i ${PIP_INDEX_URL} -r requirements.txt \
    && python scripts/download_models.py --models-dir /app/models

COPY app/ ./app/
COPY .env.example ./.env.example

VOLUME /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
