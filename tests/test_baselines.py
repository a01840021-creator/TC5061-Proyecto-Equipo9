import numpy as np
import pandas as pd

from mibici.modeling.baselines import (
    baseline_p1_rezago_semanal,
    baseline_p2_persistencia,
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
