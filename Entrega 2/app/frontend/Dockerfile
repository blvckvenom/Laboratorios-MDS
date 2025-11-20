FROM python:3.11-slim

WORKDIR /app

# copiar requirements primero para aprovechar cache de docker
COPY requirements.txt .

# instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiar codigo de la aplicacion
COPY app.py client.py ./

# exponer puerto
EXPOSE 7860

# comando para ejecutar
CMD ["python", "app.py"]
