"""Pruebas unitarias de la API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeModelService:
    """Simula el servicio de carga de modelos."""

    def __init__(self):
        self.model = None
        self.metadata = {
            "model_name": "mibici-equipo09-p1-demanda",
            "alias": "champion",
            "version": "1",
            "model_uri": "models:/mibici-equipo09-p1-demanda@champion",
            "status": "not_loaded",
        }

    def load(self):
        self.model = object()
        self.metadata["status"] = "loaded"
        return self.model


def test_health_devuelve_estado_correcto(monkeypatch):
    """Comprueba que /health responde cuando el modelo está cargado."""

    monkeypatch.setattr("app.main.ModelService", FakeModelService)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_status": "loaded",
    }


def test_model_info_devuelve_metadatos(monkeypatch):
    """Comprueba que /model-info devuelve los datos del modelo."""

    monkeypatch.setattr("app.main.ModelService", FakeModelService)

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "mibici-equipo09-p1-demanda"
    assert response.json()["alias"] == "champion"
    assert response.json()["version"] == "1"
    assert response.json()["status"] == "loaded"


def test_api_falla_si_no_puede_cargar_el_modelo(monkeypatch):
    """Comprueba que el inicio falla si MLflow no puede cargar el modelo."""

    class FailingModelService:
        def __init__(self):
            self.model = None

        def load(self):
            raise RuntimeError("Modelo no disponible en MLflow")

    monkeypatch.setattr("app.main.ModelService", FailingModelService)

    with pytest.raises(RuntimeError, match="Modelo no disponible"):
        with TestClient(app):
            pass


def test_health_devuelve_503_si_no_hay_modelo(monkeypatch):
    """Comprueba que /health devuelve 503 si el modelo no está disponible."""

    class UnavailableModelService:
        def __init__(self):
            self.model = None
            self.metadata = {"status": "not_loaded"}

        def load(self):
            return None

    monkeypatch.setattr(
        "app.main.ModelService",
        UnavailableModelService,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "El modelo no está disponible."
