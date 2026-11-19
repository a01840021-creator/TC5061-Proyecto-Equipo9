# Imagen del servicio de predicción de MiBici: API de FastAPI que sirve el champion de P1.
# El modelo viene empaquetado en models/p1/, exportado del Model Registry con
# `make modelo-servicio`, así que el contenedor no necesita un servidor de MLflow.
#
# Construir:  docker build -t mibici-api .
# Ejecutar:   docker run -d --name mibici-api -p 8000:8000 mibici-api

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

# El modelo cambia menos que el código, así que va en una capa anterior.
COPY models/p1/ models/p1/
COPY mibici/ mibici/
COPY app/ app/
# Lectura para todos sin importar los permisos del clon que construye la imagen; root sigue
# siendo el dueño, así que el usuario api puede leer pero no modificar.
RUN chmod -R a+rX models mibici app

# La API carga la carpeta empaquetada. Con -e MODEL_URI=models:/<nombre>@<alias> y
# -e MLFLOW_TRACKING_URI=... cargaría el modelo del Model Registry (ver el README).
ENV MODEL_URI=/srv/models/p1

USER api

EXPOSE 8000

# El contenedor se reporta sano sólo cuando el modelo está cargado y puede predecir.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
