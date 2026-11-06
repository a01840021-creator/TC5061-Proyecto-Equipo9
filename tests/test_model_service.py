"""Pruebas unitarias del servicio de carga de modelos."""

from types import SimpleNamespace

import pytest

from app.model_service import ModelService


def test_carga_modelo_y_obtiene_metadatos(monkeypatch):
    """Comprueba que el modelo y sus metadatos se cargan correctamente."""

    modelo_falso = object()
    llamadas = {"carga": 0, "cliente": 0, "tracking": 0}

    class ClienteFalso:
        def __init__(self, tracking_uri):
            llamadas["cliente"] += 1
            assert tracking_uri == "http://localhost:5001"

        def get_model_version_by_alias(self, name, alias):
            assert name == "mibici-equipo09-p1-demanda"
            assert alias == "champion"
            return SimpleNamespace(version="3")

    def cargar_modelo(uri):
        llamadas["carga"] += 1
        assert uri == "models:/mibici-equipo09-p1-demanda@champion"
        return modelo_falso

    def configurar_tracking(uri):
        llamadas["tracking"] += 1
        assert uri == "http://localhost:5001"

    monkeypatch.setattr(
        "app.model_service.MlflowClient",
        ClienteFalso,
    )
    monkeypatch.setattr(
        "app.model_service.mlflow.sklearn.load_model",
        cargar_modelo,
    )
    monkeypatch.setattr(
        "app.model_service.mlflow.set_tracking_uri",
        configurar_tracking,
    )

    servicio = ModelService(
        tracking_uri="http://localhost:5001",
        model_uri="models:/mibici-equipo09-p1-demanda@champion",
    )

    modelo = servicio.load()

    assert modelo is modelo_falso
    assert servicio.metadata == {
        "model_name": "mibici-equipo09-p1-demanda",
        "alias": "champion",
        "version": "3",
        "model_uri": "models:/mibici-equipo09-p1-demanda@champion",
        "status": "loaded",
    }

    assert llamadas == {"carga": 1, "cliente": 1, "tracking": 1}


def test_no_carga_dos_veces_el_mismo_modelo(monkeypatch):
    """Comprueba que una segunda llamada reutiliza el modelo cargado."""

    modelo_falso = object()
    llamadas = {"carga": 0}

    class ClienteFalso:
        def __init__(self, tracking_uri):
            pass

        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="1")

    def cargar_modelo(uri):
        llamadas["carga"] += 1
        return modelo_falso

    monkeypatch.setattr(
        "app.model_service.MlflowClient",
        ClienteFalso,
    )
    monkeypatch.setattr(
        "app.model_service.mlflow.sklearn.load_model",
        cargar_modelo,
    )
    monkeypatch.setattr(
        "app.model_service.mlflow.set_tracking_uri",
        lambda uri: None,
    )

    servicio = ModelService(
        tracking_uri="http://localhost:5001",
        model_uri="models:/modelo-prueba@champion",
    )

    primera_carga = servicio.load()
    segunda_carga = servicio.load()

    assert primera_carga is segunda_carga
    assert llamadas["carga"] == 1


def test_rechaza_model_uri_incorrecta(monkeypatch):
    """Comprueba que se rechaza una URI que no corresponde a Model Registry."""

    monkeypatch.setattr(
        "app.model_service.mlflow.set_tracking_uri",
        lambda uri: None,
    )

    servicio = ModelService(
        tracking_uri="http://localhost:5001",
        model_uri="runs:/123/model",
    )

    with pytest.raises(RuntimeError, match="MODEL_URI debe utilizar"):
        servicio.load()


def test_maneja_error_si_mlflow_no_encuentra_el_modelo(monkeypatch):
    """Comprueba que se informa claramente si falla la carga desde MLflow."""

    class ClienteFalso:
        def __init__(self, tracking_uri):
            pass

        def get_model_version_by_alias(self, name, alias):
            raise Exception("Modelo no encontrado")

    monkeypatch.setattr(
        "app.model_service.MlflowClient",
        ClienteFalso,
    )
    monkeypatch.setattr(
        "app.model_service.mlflow.set_tracking_uri",
        lambda uri: None,
    )

    servicio = ModelService(
        tracking_uri="http://localhost:5001",
        model_uri="models:/modelo-prueba@champion",
    )

    with pytest.raises(RuntimeError, match="No fue posible cargar el modelo"):
        servicio.load()
