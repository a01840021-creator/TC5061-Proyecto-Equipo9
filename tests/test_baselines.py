import numpy as np
import pandas as pd
import pytest

from mibici.modeling.baselines import (
    baseline_p1_rezago_semanal,
    baseline_p2_persistencia,
    entrenar_baseline_p3,
    metricas_p1,
    metricas_p2,
)


def test_baseline_p1_usa_rezago_semanal():
    datos = pd.DataFrame({"demanda": [2, 4, 6], "rezago_168h": [1, 5, 6]})
    prediccion = baseline_p1_rezago_semanal(datos)
    np.testing.assert_array_equal(prediccion, np.array([1.0, 5.0, 6.0]))
    metricas = metricas_p1(datos["demanda"].to_numpy(), prediccion)
    assert metricas["test_mae"] >= 0
    assert metricas["test_rmse"] >= metricas["test_mae"]


def test_baseline_p2_aplica_umbral_sobre_capacidad():
    datos = pd.DataFrame(
        {
            "saldo_lag1": [1, -4, 3, -2],
            "dpcapacity": [10, 10, 10, 20],
            "riesgo": [0, 1, 1, 0],
        }
    )
    prediccion, score = baseline_p2_persistencia(datos, umbral_capacidad=0.30)
    np.testing.assert_array_equal(prediccion, np.array([0, 1, 1, 0]))
    metricas = metricas_p2(datos["riesgo"].to_numpy(), prediccion, score)
    assert metricas["test_recall"] == 1.0
    assert metricas["test_precision"] == 1.0


def test_baseline_p3_es_reproducible():
    datos = pd.DataFrame({
        **{
            f"prop_hora_{hora}": [
                0.8 if hora < 8 or hora >= 20 else 0.1,
                0.1 if hora < 8 or hora >= 20 else 0.8,
                0.7 if hora < 8 or hora >= 20 else 0.2,
                0.2 if hora < 8 or hora >= 20 else 0.7,
            ]
            for hora in range(24)
        },
        "prop_fin_de_semana": [0.1, 0.8, 0.2, 0.7],
        "ratio_salidas": [1.0, 0.5, 1.1, 0.6],
    })

    _, etiquetas_1, variables_1, metricas_1 = entrenar_baseline_p3(
        datos, n_clusters=2, semilla=17
    )
    _, etiquetas_2, variables_2, metricas_2 = entrenar_baseline_p3(
        datos, n_clusters=2, semilla=17
    )

    np.testing.assert_array_equal(etiquetas_1, etiquetas_2)
    assert variables_1 == variables_2
    assert metricas_1 == pytest.approx(metricas_2)
