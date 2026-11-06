"""Aplicación FastAPI para servir modelos registrados en MLflow."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la aplicación."""
    service = ModelService()

    try:
        service.load()
        app.state.model_service = service
        yield
    except Exception:
        raise
    finally:
        app.state.model_service = None


app = FastAPI(
    title="MiBici Model Serving API",
    description="API para consultar el estado y los metadatos del modelo de MiBici.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    """Consulta el estado del servicio."""
    service = getattr(request.app.state, "model_service", None)

    if service is None or service.model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible.",
        )

    return {"status": "ok", "model_status": service.metadata["status"]}


@app.get("/model-info")
def model_info(request: Request):
    """Devuelve los metadatos del modelo cargado."""
    service = getattr(request.app.state, "model_service", None)

    if service is None or service.model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible.",
        )

    return service.metadata
