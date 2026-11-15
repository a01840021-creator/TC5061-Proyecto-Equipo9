"""Challenger reentrenado de P1 exportado en models/p1_challenger/ para el A/B."""

import json

import pandas as pd
import pytest

from app.model_service import ModelService
from mibici.config import (
    ALIAS_CHALLENGER,
    METADATOS_SERVICIO,
    MODELO_CHALLENGER_DIR,
    MODELO_SERVICIO_DIR,
    MODELOS_REGISTRO,
)
from mibici.monitoring import ab_test

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

# La misma entrada que tests/test_servicio_empaquetado.py: martes a las 8:00 en la estación 2.
# El challenger v3 predice 3.6396361815061535 viajes; el champion v1, 3.6143534553217758.
ENTRADA = [2.0, 15.0, 8.0, 1.0, 0.0, 3.0, 2.0, 4.0]
PREDICCION_CHALLENGER_V3 = 3.6396361815061535


def test_carpeta_es_el_challenger_v3_de_p1():
    metadatos = json.loads((MODELO_CHALLENGER_DIR / METADATOS_SERVICIO).read_text("utf-8"))

    assert metadatos["model_name"] == MODELOS_REGISTRO["P1"]
    assert metadatos["alias"] == ALIAS_CHALLENGER
    assert metadatos["version"] == "3"
    assert metadatos["model_uri"] == f"models:/{MODELOS_REGISTRO['P1']}@{ALIAS_CHALLENGER}"
    assert (MODELO_CHALLENGER_DIR / "MLmodel").is_file()


def test_challenger_empaquetado_carga_y_predice():
    servicio = ModelService(model_uri=str(MODELO_CHALLENGER_DIR))

    servicio.load()
    prediccion = servicio.predict(pd.DataFrame([ENTRADA], columns=FEATURES_P1))

    assert servicio.metadata == {
        "model_name": MODELOS_REGISTRO["P1"],
        "alias": ALIAS_CHALLENGER,
        "version": "3",
        "model_uri": f"models:/{MODELOS_REGISTRO['P1']}@{ALIAS_CHALLENGER}",
        "status": "loaded",
    }
    assert prediccion[0] == pytest.approx(PREDICCION_CHALLENGER_V3, abs=1e-9)


def test_la_api_sigue_cargando_el_champion_por_omision(monkeypatch):
    """Exportar el challenger no cambia el modelo que sirve la API."""
    monkeypatch.delenv("MODEL_URI", raising=False)

    assert ModelService().model_uri == str(MODELO_SERVICIO_DIR)


def test_el_ab_compara_por_omision_contra_el_challenger_empaquetado():
    descripcion = ab_test.describir_modelo(ab_test.MODELO_CHALLENGER_DEFAULT, "challenger")

    assert descripcion["origen"] == "models/p1_challenger"
    assert descripcion["etiqueta"] == "Challenger (v3)"
    assert descripcion["version"] == "3"
    modelo = ab_test.cargar_modelo(ab_test.MODELO_CHALLENGER_DEFAULT)
    prediccion = modelo.predict(pd.DataFrame([ENTRADA], columns=FEATURES_P1))
    assert prediccion[0] == pytest.approx(PREDICCION_CHALLENGER_V3, abs=1e-9)
