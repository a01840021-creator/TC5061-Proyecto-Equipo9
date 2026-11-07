# Imagen del servicio de predicción de MiBici: API de FastAPI que sirve el modelo champion
# de P1 recuperado del Model Registry de MLflow.
#
# Construir:  docker build -t mibici-api .
# Ejecutar:   ver el README (servidor de MLflow, variables de entorno y montaje
#             de mlartifacts/).

FROM python:3.10.20-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    LOGURU_COLORIZE=false

WORKDIR /srv

# Las dependencias van en su propia capa: si sólo cambia el código no se reinstalan.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Usuario sin privilegios. El código queda como propiedad de root, de sólo lectura para él.
RUN useradd --create-home --uid 10001 api

COPY mibici/ mibici/
COPY app/ app/

# Valores por defecto; se sobrescriben con -e al ejecutar el contenedor.
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_URI=models:/mibici-equipo09-p1-demanda@champion

USER api

EXPOSE 8000

# El contenedor se reporta sano sólo cuando el modelo está cargado y puede predecir.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
