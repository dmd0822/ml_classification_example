FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app

WORKDIR /app

# Minimal OS deps for common Python wheels (TF / sklearn) on slim images.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.serve.txt ./
RUN pip install --no-cache-dir -r requirements.serve.txt \
	&& pip install --no-cache-dir --no-deps keras_hub==0.25.0

# App code
COPY src/ ./src/
COPY entrypoints/ ./entrypoints/
COPY config/ ./config/

# Bake best-model artifacts (from the report: distilbert_unfrozen/20251223T010101Z)
RUN mkdir -p /app/artifacts/best
COPY data/04-predictions/huffpost/distilbert_unfrozen/20251223T010101Z/keras_model.keras /app/artifacts/best/keras_model.keras
COPY data/04-predictions/huffpost/distilbert_unfrozen/20251223T010101Z/tokenizer/ /app/artifacts/best/tokenizer/
COPY data/03-features/huffpost/tfidf_v1/label_encoder.joblib /app/artifacts/best/label_encoder.joblib

ENV KERAS_MODEL_PATH=/app/artifacts/best/keras_model.keras \
	TOKENIZER_DIR=/app/artifacts/best/tokenizer \
	LABEL_ENCODER_PATH=/app/artifacts/best/label_encoder.joblib \
	TEXT_CONFIG_PATH=/app/config/huffpost_category_text.json \
	MAX_LENGTH=64

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.pipelines.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]
