"""Pruebas de las gráficas y tablas que los runs registran en MLflow."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.artifacts import (
    grafica_mapa_p3,
    grafica_p1,
    grafica_p2,
    grafica_perfiles_p3,
    grafica_precision_recall_p2,
    grafica_residuales_p1,
)

FIRMA_PNG = b"\x89PNG\r\n\x1a\n"


def es_png(ruta) -> bool:
    return ruta.exists() and ruta.read_bytes()[:8] == FIRMA_PNG


def test_grafica_p1_suma_real_y_prediccion_por_hora(tmp_path):
    fechas = pd.Series(
        pd.to_datetime(["2026-09-01 08:00", "2026-09-01 08:00", "2026-09-01 09:00"])
    )
    resumen = grafica_p1(
        fechas, np.array([1, 2, 3]), np.array([1.5, 1.0, 2.0]), tmp_path / "p1.png", "Modelo", "T"
    )
    assert es_png(tmp_path / "p1.png")
    assert resumen["real"].tolist() == [3, 3]
    assert resumen["prediccion"].tolist() == [2.5, 2.0]
    assert list(resumen.columns) == ["fecha_hora", "real", "prediccion"]


def test_grafica_p1_con_una_sola_hora(tmp_path):
    fechas = pd.Series(pd.to_datetime(["2026-09-01 08:00"]))
    resumen = grafica_p1(fechas, np.array([0]), np.array([0.0]), tmp_path / "p1.png", "M", "T")
    assert len(resumen) == 1
    assert es_png(tmp_path / "p1.png")


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([0, 1, 1, 0], [0, 1, 0, 0]), ([0, 0, 0], [0, 0, 0])],
    ids=["dos_clases", "una_sola_clase"],
)
def test_grafica_p2_guarda_la_matriz_de_confusion(tmp_path, y_true, y_pred):
    grafica_p2(np.array(y_true), np.array(y_pred), tmp_path / "p2.png")
    assert es_png(tmp_path / "p2.png")


@pytest.mark.parametrize("n", [10, 50_000])
def test_grafica_residuales_p1_con_pocos_y_muchos_puntos(tmp_path, n):
    rng = np.random.default_rng(0)
    y_true = rng.poisson(2, n).astype(float)
    grafica_residuales_p1(y_true, y_true + rng.normal(0, 1, n), tmp_path / "res.png")
    assert es_png(tmp_path / "res.png")


def test_grafica_precision_recall_p2(tmp_path):
    grafica_precision_recall_p2(
        np.array([0, 1, 1, 0, 1]), np.array([0.1, 0.8, 0.6, 0.4, 0.3]), tmp_path / "pr.png"
    )
    assert es_png(tmp_path / "pr.png")


def grupos():
    return pd.DataFrame(
        {
            "latitude": [20.6, 20.7, 20.8, 20.9],
            "longitude": [-103.3, -103.4, -103.5, -103.6],
            "dpcapacity": [10, 0, -5, 20],
            "cluster": [0, 0, 1, 1],
            "prop_hora_00": [0.2, 0.4, 0.0, 0.1],
            "prop_hora_01": [0.8, 0.6, 1.0, 0.9],
        }
    )


def test_grafica_mapa_p3_acepta_capacidades_no_positivas(tmp_path):
    grafica_mapa_p3(grupos(), tmp_path / "mapa.png")
    assert es_png(tmp_path / "mapa.png")


def test_grafica_perfiles_p3_devuelve_el_promedio_por_grupo(tmp_path):
    perfiles = grafica_perfiles_p3(grupos(), ["prop_hora_00", "prop_hora_01"], tmp_path / "p.png")
    assert es_png(tmp_path / "p.png")
    assert perfiles.index.tolist() == [0, 1]
    np.testing.assert_allclose(perfiles["prop_hora_00"], [0.3, 0.05])
    np.testing.assert_allclose(perfiles["prop_hora_01"], [0.7, 0.95])
