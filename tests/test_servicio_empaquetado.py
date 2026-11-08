"""Modelo de la API empaquetado en models/p1/ y su exportación desde el Model Registry."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app.model_service import ModelService
from mibici.config import METADATOS_SERVICIO, MODELO_SERVICIO_DIR
from mibici.modeling.serving import exportar_modelo

FEATURES_P1 = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
]

# Martes a las 8:00 en la estación 2. El champion v1 de P1 predice 3.6143534553217758
# viajes, el mismo valor que devuelve la API dentro y fuera del contenedor.
ENTRADA = [2.0, 15.0, 8.0, 1.0, 0.0, 3.0, 2.0, 4.0]
PREDICCION_CHAMPION_V1 = 3.6143534553217758


def test_por_omision_la_api_usa_la_carpeta_empaquetada(monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)

    assert ModelService().model_uri == str(MODELO_SERVICIO_DIR)


def test_carpeta_empaquetada_es_el_champion_v1_de_p1_y_predice_igual(monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    servicio = ModelService()

    servicio.load()
    prediccion = servicio.predict(pd.DataFrame([ENTRADA], columns=FEATURES_P1))

    assert servicio.metadata == {
        "model_name": "mibici-equipo09-p1-demanda",
        "alias": "champion",
        "version": "1",
        "model_uri": "models:/mibici-equipo09-p1-demanda@champion",
        "status": "loaded",
    }
    assert prediccion[0] == pytest.approx(PREDICCION_CHAMPION_V1, abs=1e-9)


def test_carpeta_inexistente_da_error_claro(tmp_path):
    servicio = ModelService(model_uri=str(tmp_path / "no_existe"))

    with pytest.raises(RuntimeError, match="MODEL_URI debe utilizar"):
        servicio.load()

    assert servicio.metadata["status"] == "error"


def test_carpeta_sin_metadatos_da_error_claro(tmp_path):
    (tmp_path / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    servicio = ModelService(model_uri=str(tmp_path))

    with pytest.raises(RuntimeError, match=METADATOS_SERVICIO):
        servicio.load()

    assert servicio.metadata["status"] == "error"


def test_exporta_del_registry_una_carpeta_que_la_api_carga(mlflow_registro_temporal, tmp_path):
    client = mlflow_registro_temporal
    X = pd.DataFrame(np.arange(24, dtype="float64").reshape(8, 3), columns=["a", "b", "c"])
    estimador = LinearRegression().fit(X, X["a"] * 2 + 1)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            estimador, name="model", serialization_format="cloudpickle", pip_requirements=[]
        )
    version = mlflow.register_model(info.model_uri, "modelo-prueba").version
    client.set_registered_model_alias("modelo-prueba", "champion", version)
    destino = tmp_path / "p1"

    metadatos = exportar_modelo("modelo-prueba", "champion", destino, client)

    assert (destino / "MLmodel").is_file()
    assert json.loads((destino / METADATOS_SERVICIO).read_text(encoding="utf-8")) == metadatos
    assert metadatos["version"] == "1"
    assert metadatos["model_uri"] == "models:/modelo-prueba@champion"
    servicio = ModelService(model_uri=str(destino))
    servicio.load()
    assert servicio.metadata["version"] == "1"
    np.testing.assert_allclose(servicio.predict(X), estimador.predict(X))

    reexportado = exportar_modelo("modelo-prueba", "champion", destino, client, reemplazar=True)

    assert reexportado["version"] == "1"
    assert (destino / "MLmodel").is_file()


def test_exportar_no_sustituye_un_modelo_sin_reemplazar(mlflow_registro_temporal, tmp_path):
    destino = tmp_path / "p1"
    destino.mkdir()
    (destino / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--reemplazar"):
        exportar_modelo("modelo-prueba", "champion", destino, mlflow_registro_temporal)

    assert (destino / "MLmodel").read_text(encoding="utf-8") == "flavors: {}\n"
