"""API REST para servir el modelo P1 de demanda horaria de MiBici."""

from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pandas as pd

from app.model_service import ModelService
from app.schemas import (
    DemandaEntrada,
    DemandaSalida,
    FEATURES_P1,
    HealthResponse,
    ModelInfoResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa el servicio y conserva disponible el diagnóstico de carga."""
    service = ModelService()
    app.state.model_service = service
    try:
        service.load()
    except RuntimeError:
        logger.exception("No fue posible cargar el modelo durante el inicio de la API.")
    yield
    app.state.model_service = None


app = FastAPI(
    title="MiBici Model Serving API",
    description=(
        "API para estimar la demanda horaria P1 con el modelo champion "
        "registrado en MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def obtener_servicio(request: Request, *, exigir_modelo: bool = True) -> ModelService:
    """Recupera el servicio del estado de FastAPI y valida su disponibilidad."""
    service = getattr(request.app.state, "model_service", None)
    if service is None or (exigir_modelo and not service.ready):
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return service


@app.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    """Confirma que el proceso de la API está activo, aun si MLflow no responde."""
    service = obtener_servicio(request, exigir_modelo=False)
    return HealthResponse(status="ok", model_status=service.metadata["status"])


@app.get("/health/ready", response_model=HealthResponse)
def readiness(request: Request) -> HealthResponse:
    """Confirma que el modelo está cargado y el servicio puede predecir."""
    service = obtener_servicio(request)
    return HealthResponse(status="ok", model_status=service.metadata["status"])


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Mantiene compatibilidad con el endpoint de salud de la primera versión."""
    return readiness(request)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Devuelve nombre, alias, versión y estado del modelo cargado."""
    service = obtener_servicio(request)
    return ModelInfoResponse(**service.metadata)


@app.post("/predict/p1", response_model=DemandaSalida)
def predict_p1(payload: DemandaEntrada, request: Request) -> DemandaSalida:
    """Predice demanda horaria con las ocho variables de la firma de P1."""
    service = obtener_servicio(request)
    inicio = time.perf_counter()

    fila = payload.model_dump()
    frame = pd.DataFrame([[fila[columna] for columna in FEATURES_P1]], columns=FEATURES_P1)
    frame = frame.astype("float64")

    try:
        predicciones = service.predict(frame)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falló la inferencia del modelo P1.")
        raise HTTPException(status_code=500, detail="No fue posible generar la predicción.") from exc

    if predicciones.size != 1 or not np.isfinite(predicciones[0]):
        raise HTTPException(status_code=500, detail="El modelo produjo una salida inválida.")

    latencia_ms = (time.perf_counter() - inicio) * 1_000
    return DemandaSalida(
        prediction=float(predicciones[0]),
        model_name=service.metadata["model_name"],
        alias=service.metadata["alias"],
        version=service.metadata["version"],
        model_uri=service.metadata["model_uri"],
        latency_ms=round(latencia_ms, 3),
    )
