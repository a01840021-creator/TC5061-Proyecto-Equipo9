"""Reproducibilidad de los candidatos con las mismas funciones que usa la búsqueda.

Cada prueba entrena dos veces con ``ejecutar_p1``, ``ejecutar_p2`` o ``ejecutar_p3`` de
``mibici/modeling/candidates.py``, sobre tablones construidos con ``mibici/features.py``
a partir de viajes sintéticos, y exige las mismas predicciones, probabilidades y métricas.
Las corridas van a una base de MLflow temporal.
"""

import mlflow
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from mibici import config
from mibici.config import ESTACIONES_ACTIVAS, FECHA_CORTE
from mibici.features import particionar_temporalmente
from mibici.modeling import candidates
from mibici.modeling.baselines import columnas_p3
from mibici.modeling.candidates import (
    FEATURES_P1,
    FEATURES_P2,
    configuraciones_busqueda,
    construir_clasificador,
    construir_cluster,
    construir_regresor,
    ejecutar_p1,
    ejecutar_p2,
    ejecutar_p3,
    limpiar_x,
    particionar_p2,
)


def configuracion(pregunta: str, modelo: str) -> dict:
    """Primera configuración de la búsqueda de la pregunta para ese modelo."""
    return next(c for c in configuraciones_busqueda(pregunta) if c["modelo"] == modelo)


def x_prueba_p1(tablon):
    _, test = particionar_temporalmente(tablon, fecha_corte=FECHA_CORTE)
    return limpiar_x(test, FEATURES_P1)


def x_prueba_p2(tablon):
    _, _, test = particionar_p2(tablon)
    return limpiar_x(test, FEATURES_P2)


def entrenar_dos_veces(ejecutar, datos, config_modelo, capturados, client):
    """Dos corridas con la misma configuración; devuelve modelos y métricas de cada una."""
    corridas = [ejecutar(datos, "v-prueba", dict(config_modelo)) for _ in range(2)]
    metricas = [client.get_run(run_id).data.metrics for run_id in corridas]
    return capturados[-2:], metricas


@pytest.mark.parametrize("modelo", ["hist_gradient_boosting", "ridge"])
def test_candidato_p1_es_reproducible(modelo, tablon_p1, mlflow_temporal, modelos_capturados):
    (primero, segundo), (metricas_1, metricas_2) = entrenar_dos_veces(
        ejecutar_p1, tablon_p1, configuracion("P1", modelo), modelos_capturados, mlflow_temporal
    )
    x = x_prueba_p1(tablon_p1)
    np.testing.assert_array_equal(primero.predict(x), segundo.predict(x))
    assert metricas_1 == metricas_2
    assert {"val_mae", "test_mae", "val_r2", "test_wape"} <= set(metricas_1)


@pytest.mark.parametrize("modelo", ["hist_gradient_boosting", "logistic_regression"])
def test_candidato_p2_es_reproducible(modelo, tablon_p2, mlflow_temporal, modelos_capturados):
    (primero, segundo), (metricas_1, metricas_2) = entrenar_dos_veces(
        ejecutar_p2, tablon_p2, configuracion("P2", modelo), modelos_capturados, mlflow_temporal
    )
    x = x_prueba_p2(tablon_p2)
    np.testing.assert_array_equal(primero.predict(x), segundo.predict(x))
    np.testing.assert_array_equal(primero.predict_proba(x), segundo.predict_proba(x))
    assert metricas_1 == metricas_2
    assert {"val_average_precision", "test_average_precision", "test_recall"} <= set(metricas_1)


@pytest.mark.parametrize("nombre", ["P3_kmeans_k5", "P3_minibatch_kmeans_k5", "P3_ward_k5"])
def test_candidato_p3_es_reproducible(nombre, perfiles_p3, mlflow_temporal, modelos_capturados):
    config_modelo = next(c for c in configuraciones_busqueda("P3") if c["run_name"] == nombre)
    (primero, segundo), (metricas_1, metricas_2) = entrenar_dos_veces(
        ejecutar_p3, perfiles_p3, config_modelo, modelos_capturados, mlflow_temporal
    )
    x = limpiar_x(perfiles_p3, columnas_p3(perfiles_p3))
    np.testing.assert_array_equal(primero.predict(x), segundo.predict(x))
    assert metricas_1 == metricas_2
    assert set(metricas_1) == {
        "train_silhouette",
        "train_davies_bouldin",
        "train_calinski_harabasz",
        "train_inercia",
    }


def test_p3_con_estaciones_activas_excluye_las_que_no_tienen_salidas(
    perfiles_p3, mlflow_temporal, modelos_capturados
):
    config_modelo = next(
        c for c in configuraciones_busqueda("P3") if c["estaciones"] == ESTACIONES_ACTIVAS
    )
    run_id = ejecutar_p3(perfiles_p3, "v-prueba", dict(config_modelo))
    parametros = mlflow_temporal.get_run(run_id).data.params
    assert parametros["n_estaciones"] == str(len(perfiles_p3) - 2)
    assert parametros["n_estaciones_excluidas"] == "2"
    assert parametros["estaciones_excluidas"] == "31,32"


def test_p3_rechaza_un_conjunto_de_estaciones_desconocido(perfiles_p3, mlflow_temporal):
    with pytest.raises(ValueError, match="estaciones"):
        ejecutar_p3(perfiles_p3, "v", {"modelo": "kmeans", "n_clusters": 5, "estaciones": "x"})


# Origen de la semilla -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("construir", "pregunta", "modelo", "paso"),
    [
        (construir_regresor, "P1", "hist_gradient_boosting", None),
        (construir_clasificador, "P2", "hist_gradient_boosting", None),
        (construir_clasificador, "P2", "logistic_regression", "logistic"),
    ],
)
def test_semilla_de_los_modelos_viene_de_config(construir, pregunta, modelo, paso, monkeypatch):
    estimador = construir(configuracion(pregunta, modelo))
    if paso:
        estimador = estimador.named_steps[paso]
    assert candidates.SEMILLA == config.SEMILLA
    assert estimador.random_state == config.SEMILLA

    # La semilla se lee de la constante al construir el modelo, no está escrita en el código
    monkeypatch.setattr(candidates, "SEMILLA", config.SEMILLA + 1)
    estimador = construir(configuracion(pregunta, modelo))
    if paso:
        estimador = estimador.named_steps[paso]
    assert estimador.random_state == config.SEMILLA + 1


@pytest.mark.parametrize("modelo", ["kmeans", "minibatch_kmeans"])
def test_semilla_de_kmeans_viene_de_config(modelo):
    estimador = construir_cluster({"modelo": modelo, "n_clusters": 5}).named_steps["cluster"]
    assert estimador.random_state == config.SEMILLA


def test_ridge_no_necesita_semilla_porque_su_solver_es_determinista(tablon_p1):
    """Ridge sólo usa ``random_state`` con los solvers ``sag`` y ``saga``.

    Con datos densos, ``solver="auto"`` elige ``cholesky``, una solución cerrada; por eso
    el proyecto no le pasa semilla y la prueba de reproducibilidad de Ridge pasa igual.
    """
    train, _ = particionar_temporalmente(tablon_p1, fecha_corte=FECHA_CORTE)
    modelo = construir_regresor(configuracion("P1", "ridge"))
    modelo.fit(limpiar_x(train, FEATURES_P1), train["demanda"])
    assert modelo.named_steps["ridge"].solver_ == "cholesky"


@pytest.mark.parametrize(
    ("estimador", "datos", "objetivo", "features"),
    [
        (HistGradientBoostingRegressor, "tablon_p1", "demanda", FEATURES_P1),
        (HistGradientBoostingClassifier, "tablon_p2", "riesgo", FEATURES_P2),
    ],
)
def test_la_semilla_si_cambia_hist_gradient_boosting(
    estimador, datos, objetivo, features, request
):
    """Con otra semilla cambian las predicciones: la prueba de reproducibilidad no es trivial.

    Con más de 10,000 filas HistGradientBoosting activa la parada temprana y separa al azar
    su validación interna con ``random_state``.
    """
    tablon = request.getfixturevalue(datos)
    train, test = particionar_temporalmente(tablon, fecha_corte=FECHA_CORTE)
    x_train, x_test = limpiar_x(train, features), limpiar_x(test, features)
    assert len(x_train) > 10_000

    def predecir(semilla):
        modelo = estimador(max_iter=50, random_state=semilla).fit(x_train, train[objetivo])
        assert modelo.do_early_stopping_
        return modelo.predict(x_test) if objetivo == "demanda" else modelo.predict_proba(x_test)

    assert not np.array_equal(predecir(config.SEMILLA), predecir(config.SEMILLA + 1))


# Registro completo del modelo --------------------------------------------------------


def test_ejecutar_p1_registra_un_modelo_que_se_carga_y_predice_igual(tablon_p1, mlflow_temporal):
    """Sin sustituir ``log_model``: el modelo registrado y su recarga dan lo mismo."""
    config_modelo = configuracion("P1", "hist_gradient_boosting")
    run_id = ejecutar_p1(tablon_p1, "v-prueba", dict(config_modelo))

    artefactos = {
        a.path
        for carpeta in ("plots", "config", "data")
        for a in mlflow_temporal.list_artifacts(run_id, carpeta)
    }
    assert {
        "plots/real_vs_predicho_p1.png",
        "plots/residuales_p1.png",
        "config/run_config.json",
        "config/config.py",
        "data/manifiesto_datos.json",
    } <= artefactos

    corrida = mlflow_temporal.get_run(run_id)
    assert corrida.data.params["semilla"] == str(config.SEMILLA)
    assert corrida.data.params["version_datos"] == "v-prueba"
    assert corrida.data.tags["stage_of_work"] == "candidate"

    cargado = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    x = x_prueba_p1(tablon_p1)
    nuevo = construir_regresor(config_modelo)
    train, _ = particionar_temporalmente(tablon_p1, fecha_corte=config.FECHA_VALIDACION)
    nuevo.fit(limpiar_x(train, FEATURES_P1), train["demanda"].to_numpy())
    np.testing.assert_array_equal(cargado.predict(x), nuevo.predict(x))
