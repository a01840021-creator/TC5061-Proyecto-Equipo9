import numpy as np
import pandas as pd
import pytest

from mibici.modeling.baselines import (
    baseline_p1_rezago_semanal,
    baseline_p2_persistencia,
    columnas_p3,
    entrenar_baseline_p3,
    metricas_p1,
    metricas_p2,
    renombrar_metricas,
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
    datos = pd.DataFrame(
        {
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
        }
    )

    _, etiquetas_1, variables_1, metricas_1 = entrenar_baseline_p3(datos, n_clusters=2, semilla=17)
    _, etiquetas_2, variables_2, metricas_2 = entrenar_baseline_p3(datos, n_clusters=2, semilla=17)

    np.testing.assert_array_equal(etiquetas_1, etiquetas_2)
    assert variables_1 == variables_2
    assert metricas_1 == pytest.approx(metricas_2)


def test_renombrar_metricas_cambia_el_prefijo_test():
    assert renombrar_metricas({"test_mae": 1.0, "test_r2": 0.5}, "val") == {
        "val_mae": 1.0,
        "val_r2": 0.5,
    }
    # Una métrica sin prefijo test_ recibe el prefijo nuevo completo
    assert renombrar_metricas({"inercia": 3.0}, "train") == {"train_inercia": 3.0}
    assert renombrar_metricas({}, "val") == {}


def test_metricas_p1_sin_demanda_no_divide_entre_cero():
    metricas = metricas_p1(np.zeros(3), np.array([1.0, 0.0, 2.0]))
    assert metricas["test_wape"] == pytest.approx(3.0)
    assert metricas["test_bias"] == pytest.approx(1.0)


def test_baseline_p1_falla_sin_rezago_semanal():
    with pytest.raises(ValueError, match="rezago_168h"):
        baseline_p1_rezago_semanal(pd.DataFrame({"demanda": [1]}))


def test_baseline_p2_falla_sin_columnas():
    with pytest.raises(ValueError, match="saldo_lag1"):
        baseline_p2_persistencia(pd.DataFrame({"dpcapacity": [10]}))


@pytest.mark.parametrize("umbral", [0, -0.1, 1.01])
def test_baseline_p2_rechaza_umbral_fuera_de_intervalo(umbral):
    datos = pd.DataFrame({"saldo_lag1": [1], "dpcapacity": [10]})
    with pytest.raises(ValueError, match="umbral"):
        baseline_p2_persistencia(datos, umbral_capacidad=umbral)


def test_baseline_p2_con_capacidad_cero_no_alerta():
    datos = pd.DataFrame({"saldo_lag1": [5, 3], "dpcapacity": [0, 10]})
    prediccion, score = baseline_p2_persistencia(datos, umbral_capacidad=1.0)
    np.testing.assert_array_equal(score, [0.0, 0.3])
    np.testing.assert_array_equal(prediccion, [0, 0])


def test_metricas_p2_sin_alertas_ni_positivos():
    metricas = metricas_p2(np.zeros(4), np.zeros(4), np.zeros(4))
    assert metricas["test_precision"] == 0.0
    assert metricas["test_recall"] == 0.0
    assert metricas["test_prevalencia_riesgo"] == 0.0


def test_columnas_p3_exige_las_24_horas():
    perfiles = pd.DataFrame(
        {
            **{f"prop_hora_{h:02d}": [0.0] for h in range(23)},
            "prop_fin_de_semana": [0.0],
            "ratio_salidas": [0.5],
        }
    )
    with pytest.raises(ValueError, match="24 proporciones"):
        columnas_p3(perfiles)


def test_columnas_p3_ignora_identificadores_y_ubicacion(perfiles_p3):
    variables = columnas_p3(perfiles_p3)
    assert len(variables) == 26
    assert not {"id", "name", "latitude", "longitude", "dpcapacity", "n_salidas"} & set(variables)


@pytest.mark.parametrize("n_clusters", [1, 32])
def test_baseline_p3_rechaza_k_fuera_de_rango(perfiles_p3, n_clusters):
    with pytest.raises(ValueError, match="n_clusters"):
        entrenar_baseline_p3(perfiles_p3, n_clusters=n_clusters)
