FROM python:3.11-slim

WORKDIR /app

# Copiar primero requirements.txt para aprovechar el caché de capas de Docker:
# si solo cambia el código, esta capa no se reconstruye.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación y modelo ya optimizado (generado en proyecto_01)
COPY app/ ./app
COPY model/ ./model

EXPOSE 8000

# Endpoint de salud usado por Docker/orquestadores para saber si el contenedor está listo
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
