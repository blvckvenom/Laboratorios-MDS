FROM python:3.11-slim

WORKDIR /app

# copiar requirements primero para aprovechar cache de docker
COPY requirements.txt .

# instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiar codigo de la aplicacion
COPY main.py models.py inference.py config.py ./

# copiar artifacts (modelo y metricas)
COPY artifacts/ ./artifacts/

# exponer puerto
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# comando para ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
