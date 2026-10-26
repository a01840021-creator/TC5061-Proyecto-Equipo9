"""Pruebas unitarias ligeras para la configuración de candidatos."""

import pandas as pd
import pytest

from mibici.config import (
    ESTACIONES_ACTIVAS,
    ESTACIONES_TODAS,
    FECHA_CORTE,
    FECHA_VALIDACION,
    HORIZONTE_P2_HORAS,
)
from mibici.modeling.candidates import (
    FEATURES_P1,
    HIPERPARAMETROS,
    MODELOS,
    configuraciones_busqueda,
    particionar_p2,
)


@pytest.mark.parametrize(
    ("pregunta", "familias"),
    [
        ("P1", ["hist_gradient_boosting"] * 3 + ["ridge"]),
        ("P2", ["hist_gradient_boosting"] * 3 + ["logistic_regression"]),
        ("P3", ["kmeans", "kmeans", "minibatch_kmeans", "ward", "kmeans"]),
    ],
)
def test_busqueda_contiene_todos_los_candidatos(pregunta: str, familias: list[str]) -> None:
    configuraciones = configuraciones_busqueda(pregunta)
    assert [config["modelo"] for config in configuraciones] == familias
    assert all(config["modelo"] in MODELOS[pregunta] for config in configuraciones)
    nombres = [config["run_name"] for config in configuraciones]
    assert len(set(nombres)) == len(nombres)


def test_cada_candidato_trae_sus_hiperparametros() -> None:
    for pregunta in MODELOS:
        for config in configuraciones_busqueda(pregunta):
            assert set(HIPERPARAMETROS[config["modelo"]]) <= set(config)


def test_p3_tiene_una_corrida_sin_inactivas() -> None:
    estaciones = [config["estaciones"] for config in configuraciones_busqueda("P3")]
    assert estaciones.count(ESTACIONES_ACTIVAS) == 1
    assert estaciones.count(ESTACIONES_TODAS) == 4


def test_p1_no_usa_mes() -> None:
    assert "mes" not in FEATURES_P1


def test_pregunta_invalida_falla() -> None:
    with pytest.raises(ValueError):
        configuraciones_busqueda("P4")


def test_p2_purga_entrenamiento_antes_de_validacion() -> None:
    validacion = pd.Timestamp(FECHA_VALIDACION)
    horas = pd.date_range(validacion - pd.Timedelta(hours=6), pd.Timestamp(FECHA_CORTE), freq="h")
    datos = pd.DataFrame({"fecha_hora": horas, "riesgo": 0})
    train, val, test = particionar_p2(datos)

    ultima_valida = validacion - pd.Timedelta(hours=HORIZONTE_P2_HORAS + 1)
    assert train["fecha_hora"].max() == ultima_valida
    assert len(train) == 6 - HORIZONTE_P2_HORAS
    # La validación y la prueba no pierden horas
    assert val["fecha_hora"].min() == validacion
    assert len(val) + len(test) == (horas >= validacion).sum()
