"""Pruebas de contrato, validación y manejo de errores de la API FastAPI."""

from app.main import app
from fastapi.testclient import TestClient
import numpy as np

ENTRADA_VALIDA = {
    "estacion_id": 44,
    "dpcapacity": 24.0,
    "hora": 8,
    "dia_semana": 1,
    "es_fin_de_semana": 0,
    "rezago_24h": 3.0,
    "rezago_48h": 2.0,
    "rezago_168h": 4.0,
}


class FakeModelService:
    """Servicio determinista para probar la API sin depender de un Registry local."""

    def __init__(self):
        self.model = None
        self.metadata = {
            "model_name": "mibici-equipo09-p1-demanda",
            "alias": "champion",
            "version": "1",
            "model_uri": "models:/mibici-equipo09-p1-demanda@champion",
            "status": "not_loaded",
        }

    @property
    def ready(self):
        return self.model is not None and self.metadata["status"] == "loaded"

    def load(self):
        self.model = object()
        self.metadata["status"] = "loaded"
        return self.model

    def predict(self, frame):
        assert list(frame.columns) == [
            "estacion_id",
            "dpcapacity",
            "hora",
            "dia_semana",
            "es_fin_de_semana",
            "rezago_24h",
            "rezago_48h",
            "rezago_168h",
        ]
        assert all(dtype == "float64" for dtype in frame.dtypes.astype(str))
        return np.array([3.75])


class FailingModelService:
    """Simula un Registry inaccesible durante el arranque."""

    def __init__(self):
        self.model = None
        self.metadata = {
            "model_name": None,
            "alias": None,
            "version": None,
            "model_uri": "models:/mibici-equipo09-p1-demanda@champion",
            "status": "not_loaded",
        }

    @property
    def ready(self):
        return False

    def load(self):
        self.metadata["status"] = "error"
        raise RuntimeError("Modelo no disponible en MLflow")


def test_health_y_model_info_devuelven_estado_cargado(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        info = client.get("/model-info")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "model_status": "loaded"}
    assert info.status_code == 200
    assert info.json()["model_name"] == "mibici-equipo09-p1-demanda"
    assert info.json()["alias"] == "champion"


def test_predict_p1_devuelve_prediccion_y_linaje(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=ENTRADA_VALIDA)

    assert response.status_code == 200
    salida = response.json()
    assert salida["prediction"] == 3.75
    assert salida["model_name"] == "mibici-equipo09-p1-demanda"
    assert salida["alias"] == "champion"
    assert salida["version"] == "1"
    assert salida["latency_ms"] >= 0


def test_predict_p1_rechaza_hora_fuera_de_rango(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)
    entrada = {**ENTRADA_VALIDA, "hora": 24}

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=entrada)

    assert response.status_code == 422


def test_predict_p1_rechaza_rezago_negativo(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)
    entrada = {**ENTRADA_VALIDA, "rezago_24h": -1.0}

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=entrada)

    assert response.status_code == 422


def test_predict_p1_rechaza_indicador_de_fin_de_semana_inconsistente(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)
    entrada = {**ENTRADA_VALIDA, "dia_semana": 6, "es_fin_de_semana": 0}

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=entrada)

    assert response.status_code == 422


def test_predict_p1_rechaza_campos_faltantes(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)
    entrada = {clave: valor for clave, valor in ENTRADA_VALIDA.items() if clave != "dpcapacity"}

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=entrada)

    assert response.status_code == 422


def test_predict_p1_rechaza_campos_desconocidos(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FakeModelService)
    entrada = {**ENTRADA_VALIDA, "campo_no_permitido": 1}

    with TestClient(app) as client:
        response = client.post("/predict/p1", json=entrada)

    assert response.status_code == 422


def test_liveness_funciona_y_readiness_falla_sin_modelo(monkeypatch):
    monkeypatch.setattr("app.main.ModelService", FailingModelService)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        prediction = client.post("/predict/p1", json=ENTRADA_VALIDA)

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "model_status": "error"}
    assert ready.status_code == 503
    assert prediction.status_code == 503
