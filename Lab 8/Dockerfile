# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY artifacts/models/best_model.pkl artifacts/models/best_model.pkl

EXPOSE 8000

ENV MODEL_PATH=/app/artifacts/models/best_model.pkl

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
