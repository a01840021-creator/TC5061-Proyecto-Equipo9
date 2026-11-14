"""Pruebas unitarias para el módulo de A/B testing de P1."""

from __future__ import annotations

import json
from pathlib import Path
import pickle  # nosec B403

import numpy as np
import pytest
from typer.testing import CliRunner

from mibici.monitoring import ab_test


class ModeloDummy:
    """Modelo simple para pruebas unitarias sin dependencia de archivos pesados."""

    def __init__(self, sesgo: float = 0.0):
        self.sesgo = sesgo

    def predict(self, X):
        # Predicción determinista basada en el número de filas
        n = len(X)
        return np.ones(n, dtype=float) * 2.0 + self.sesgo


@pytest.fixture
def datos_sinteticos():
    """Genera datos sintéticos para pruebas deterministas."""
    y_real = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    # Modelo A con predicción perfecta
    y_pred_a = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    # Modelo B con error de 1.0 en cada fila
    y_pred_b = np.array([3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    return y_real, y_pred_a, y_pred_b


def test_evaluar_desempeno_perfecto(datos_sinteticos):
    """Verifica el cálculo de métricas cuando la predicción es exacta."""
    y_real, y_pred_a, _ = datos_sinteticos
    res = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.5, nombre="modelo_perfecto")

    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["wape_pct"] == 0.0
    assert res["r2"] == 1.0
    assert res["latencia_ms_muestra"] == 0.5
    assert len(res["_errores_absolutos"]) == len(y_real)


def test_evaluar_desempeno_con_error(datos_sinteticos):
    """Verifica el cálculo de métricas con errores conocidos."""
    y_real, _, y_pred_b = datos_sinteticos
    res = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=1.2, nombre="modelo_error")

    assert res["mae"] == 1.0
    assert res["rmse"] == 1.0
    # suma errores = 5, suma real = 20 -> WAPE = 25%
    assert res["wape_pct"] == 25.0
    assert res["latencia_ms_muestra"] == 1.2


def test_ejecutar_ab_testing_challenger_gana():
    """Verifica la recomendación cuando el Challenger supera al Champion con significancia."""
    # 100 muestras con error consistente del Champion (2.0) y menor del Challenger (1.0)
    n = 100
    y_real = np.zeros(n, dtype=float)
    y_pred_a = np.ones(n, dtype=float) * 2.0
    y_pred_b = np.ones(n, dtype=float) * 1.0

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.05)

    assert res["decision"]["recomendacion"] == "promover_challenger"
    assert res["decision"]["criterios_cumplidos"]["mejora_mae"] is True
    assert res["decision"]["criterios_cumplidos"]["estadisticamente_significativo"] is True
    assert res["comparativa"]["delta_mae"] == -1.0
    assert res["inferencia_estadistica"]["p_valor_t"] < 0.05


def test_ejecutar_ab_testing_no_significativo():
    """Verifica que no se promueva si la diferencia no es estadísticamente significativa."""
    # Diferencia casi nula y con alta variabilidad
    rng = np.random.default_rng(42)
    n = 30
    y_real = rng.normal(10, 2, n)
    y_pred_a = y_real + rng.normal(0, 1.0, n)
    y_pred_b = y_real + rng.normal(0, 0.99, n)

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.01)

    # Si p > alpha, la recomendación no debe ser promover
    assert res["decision"]["recomendacion"] in {"mantener_champion", "descartar_challenger"}


def test_ejecutar_ab_testing_challenger_peor():
    """Verifica el descarte si el Challenger comete mayor error que el Champion."""
    n = 50
    y_real = np.zeros(n, dtype=float)
    y_pred_a = np.ones(n, dtype=float) * 1.0
    y_pred_b = np.ones(n, dtype=float) * 3.0

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.05)

    assert res["decision"]["recomendacion"] == "descartar_challenger"
    assert res["comparativa"]["delta_mae"] > 0


def test_guardar_resultados_ab(tmp_path: Path):
    """Verifica que el artefacto JSON se guarde con todas las secciones requeridas."""
    datos = {
        "modelo_a": {"mae": 0.8},
        "modelo_b": {"mae": 0.7},
        "comparativa": {"delta_mae": -0.1},
        "inferencia_estadistica": {"t_stat": 2.5, "p_valor_t": 0.01},
        "decision": {"recomendacion": "promover_challenger"},
    }

    ruta = ab_test.guardar_resultados_ab(datos, carpeta_salida=tmp_path)
    assert ruta.exists()

    with open(ruta, "r", encoding="utf-8") as f:
        contenido = json.load(f)

    assert contenido["decision"]["recomendacion"] == "promover_challenger"
    assert "inferencia_estadistica" in contenido


def test_formatear_tabla_markdown():
    """Verifica que la salida en Markdown contenga los encabezados y métricas clave."""
    res = {
        "modelo_a": {
            "mae": 0.83,
            "rmse": 1.60,
            "wape_pct": 61.4,
            "r2": 0.63,
            "latencia_ms_muestra": 0.001,
        },
        "modelo_b": {
            "mae": 0.82,
            "rmse": 1.59,
            "wape_pct": 61.2,
            "r2": 0.64,
            "latencia_ms_muestra": 0.001,
        },
        "comparativa": {
            "delta_mae": -0.01,
            "delta_mae_pct": -1.2,
            "delta_rmse": -0.01,
            "delta_rmse_pct": -0.6,
            "delta_wape_pct": -0.2,
            "delta_r2": 0.01,
            "delta_latencia_ms": 0.0,
        },
        "inferencia_estadistica": {
            "t_stat": 4.5,
            "p_valor_t": 1e-5,
            "alpha": 0.05,
            "ic_95_diferencia": [0.005, 0.015],
        },
        "decision": {
            "recomendacion": "promover_challenger",
            "justificacion": "Mejora demostrada con significancia.",
        },
    }

    tabla = ab_test.formatear_tabla_markdown(res)
    assert "| Métrica | Champion (v1) | Challenger (v2) |" in tabla
    assert "PROMOVER_CHALLENGER" in tabla
    assert "MAE" in tabla
    assert "RMSE" in tabla


def test_cargar_modelo_archivo_y_directorio(tmp_path: Path):
    """Verifica la carga de modelos desde archivo .pkl y desde directorio con model.pkl."""
    modelo_original = ModeloDummy(sesgo=0.5)

    # 1. Archivo pickle directo
    ruta_pkl = tmp_path / "test_model.pkl"
    with open(ruta_pkl, "wb") as f:
        pickle.dump(modelo_original, f)

    m1 = ab_test.cargar_modelo(ruta_pkl)
    assert isinstance(m1, ModeloDummy)

    # 2. Directorio con model.pkl
    dir_modelo = tmp_path / "modelo_dir"
    dir_modelo.mkdir()
    with open(dir_modelo / "model.pkl", "wb") as f:
        pickle.dump(modelo_original, f)

    m2 = ab_test.cargar_modelo(dir_modelo)
    assert isinstance(m2, ModeloDummy)


def test_cargar_modelo_no_existente():
    """Verifica que se lance FileNotFoundError ante una ruta inexistente."""
    with pytest.raises(FileNotFoundError):
        ab_test.cargar_modelo("ruta/inexistente/a_modelo")


def test_predecir_con_latencia():
    """Verifica que la predicción devuelva un array float y latencia no negativa."""
    import pandas as pd

    mod = ModeloDummy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    preds, lat = ab_test.predecir_con_latencia(mod, X)

    assert len(preds) == 3
    assert lat >= 0.0


def test_ab_test_cli_smoke(tmp_path: Path):
    """Verifica que la CLI de ab_test se ejecute correctamente con max-filas."""
    runner = CliRunner()
    modelo_path = Path("models/p1")

    # Si models/p1 existe, probamos la CLI con max-filas
    if modelo_path.exists():
        resultado = runner.invoke(
            ab_test.app,
            [
                "--modelo-a",
                str(modelo_path),
                "--modelo-b",
                "auto",
                "--max-filas",
                "100",
                "--salida",
                str(tmp_path),
            ],
        )
        assert resultado.exit_code == 0
        assert "REPORTE DE A/B TESTING" in resultado.output
        assert (tmp_path / "ab_test_results.json").exists()
