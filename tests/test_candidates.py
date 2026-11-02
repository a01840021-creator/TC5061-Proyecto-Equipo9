"""Pruebas de la configuración, la construcción y el registro de los candidatos."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans, MiniBatchKMeans
from typer.testing import CliRunner

from mibici.config import (
    ESTACIONES_ACTIVAS,
    ESTACIONES_TODAS,
    FECHA_CORTE,
    FECHA_VALIDACION,
    HORIZONTE_P2_HORAS,
    PERIODO_TRAIN_CANDIDATOS,
    PERIODO_VALIDACION,
    SEMILLA,
)
from mibici.modeling import candidates
from mibici.modeling.candidates import (
    FEATURES_P1,
    HIPERPARAMETROS,
    MODELOS,
    configuraciones_busqueda,
    construir_clasificador,
    construir_cluster,
    construir_regresor,
    limpiar_x,
    parametros_comunes,
    particionar_p2,
    preparar_tracking,
    registrar_configuracion,
    registrar_contexto_datos,
    registrar_parametros,
    separar_estaciones_inactivas,
)
from mibici.modeling.clustering import WardCentroides


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


# Utilidades de entrenamiento y registro -----------------------------------------------


def test_limpiar_x_selecciona_y_reemplaza_no_finitos() -> None:
    datos = pd.DataFrame({"a": [1, np.inf, -np.inf], "b": [np.nan, 2, 3], "c": ["x", "y", "z"]})
    x = limpiar_x(datos, ["a", "b"])
    assert list(x.columns) == ["a", "b"]
    assert (x.dtypes == "float64").all()
    assert x.to_numpy().tolist() == [[1.0, 0.0], [0.0, 2.0], [0.0, 3.0]]


def test_limpiar_x_falla_si_falta_una_variable() -> None:
    with pytest.raises(KeyError):
        limpiar_x(pd.DataFrame({"a": [1]}), ["a", "b"])


@pytest.mark.parametrize(
    ("construir", "pregunta"),
    [(construir_regresor, "P1"), (construir_clasificador, "P2")],
)
def test_construir_rechaza_modelos_de_otra_pregunta(construir, pregunta) -> None:
    with pytest.raises(ValueError, match=pregunta):
        construir({"modelo": "kmeans"})


@pytest.mark.parametrize(
    ("modelo", "clase"),
    [("kmeans", KMeans), ("MiniBatch_KMeans", MiniBatchKMeans), ("ward", WardCentroides)],
)
def test_construir_cluster_estandariza_y_usa_k(modelo, clase) -> None:
    pipeline = construir_cluster({"modelo": modelo, "n_clusters": 6, "batch_size": 64})
    assert list(pipeline.named_steps) == ["scaler", "cluster"]
    assert isinstance(pipeline.named_steps["cluster"], clase)
    assert pipeline.named_steps["cluster"].n_clusters == 6


def test_construir_cluster_rechaza_modelo_desconocido() -> None:
    with pytest.raises(ValueError, match="P3"):
        construir_cluster({"modelo": "dbscan", "n_clusters": 3})


def test_separar_estaciones_inactivas() -> None:
    perfiles = pd.DataFrame({"id": [1, 2, 3, 4], "n_salidas": [5, 0, 1, 0]})
    activas, inactivas = separar_estaciones_inactivas(perfiles)
    assert activas["id"].tolist() == [1, 3]
    assert inactivas["id"].tolist() == [2, 4]
    assert activas.index.tolist() == [0, 1]


def test_separar_estaciones_inactivas_sin_inactivas() -> None:
    activas, inactivas = separar_estaciones_inactivas(pd.DataFrame({"id": [1], "n_salidas": [3]}))
    assert len(activas) == 1 and inactivas.empty


def test_parametros_comunes_de_candidatos_entrenan_hasta_julio() -> None:
    parametros = parametros_comunes("v1")
    assert parametros["periodo_train"] == PERIODO_TRAIN_CANDIDATOS
    assert parametros["periodo_validation"] == PERIODO_VALIDACION
    assert parametros["semilla"] == SEMILLA


def test_registrar_parametros_omite_run_name_y_los_ya_registrados(mlflow_temporal) -> None:
    with mlflow.start_run() as corrida:
        mlflow.log_param("alpha", "1")
        registrar_parametros({"alpha": 1.0, "nuevo": 3, "run_name": "x"})
    parametros = mlflow_temporal.get_run(corrida.info.run_id).data.params
    assert parametros == {"alpha": "1", "nuevo": "3"}


def test_registrar_configuracion_guarda_json_y_config(mlflow_temporal, tmp_path) -> None:
    with mlflow.start_run() as corrida:
        registrar_configuracion({"modelo": "ridge", "alpha": 2.0}, tmp_path)
    assert json.loads((tmp_path / "run_config.json").read_text("utf-8")) == {
        "modelo": "ridge",
        "alpha": 2.0,
    }
    nombres = {a.path for a in mlflow_temporal.list_artifacts(corrida.info.run_id, "config")}
    assert nombres == {"config/run_config.json", "config/config.py"}


@pytest.mark.parametrize(("filas", "esperadas"), [(20, 20), (6_000, 5_000)])
def test_registrar_contexto_datos_registra_una_muestra_acotada(
    mlflow_temporal, filas, esperadas
) -> None:
    train = pd.DataFrame({"x": np.arange(filas), "y": np.arange(filas) % 2, "privada": 1})
    with mlflow.start_run() as corrida:
        registrar_contexto_datos(train, ["x"], "y", "P2")
    entradas = mlflow_temporal.get_run(corrida.info.run_id).inputs.dataset_inputs
    assert len(entradas) == 1
    conjunto = entradas[0].dataset
    assert conjunto.name == "mibici_p2_training_sample"
    perfil = json.loads(conjunto.profile)
    assert perfil["num_rows"] == esperadas
    assert "privada" not in conjunto.schema
    artefactos = {a.path for a in mlflow_temporal.list_artifacts(corrida.info.run_id, "data")}
    assert artefactos == {"data/manifiesto_datos.json"}


def test_preparar_tracking_prefiere_la_variable_de_entorno(mlflow_temporal) -> None:
    uri = mlflow.get_tracking_uri()
    preparar_tracking("http://servidor-que-no-existe.invalid:5000", "experimento-prueba")
    assert mlflow.get_tracking_uri() == uri
    assert mlflow_temporal.get_experiment_by_name("experimento-prueba") is not None


@pytest.mark.parametrize(
    "argumentos",
    [
        ["train", "--pregunta", "P4"],
        ["train", "--pregunta", "P1", "--modelo", "kmeans"],
        ["train", "--pregunta", "P3", "--estaciones", "algunas"],
        ["search", "--pregunta", "P9"],
    ],
)
def test_cli_rechaza_opciones_invalidas_antes_de_conectarse(argumentos) -> None:
    resultado = CliRunner().invoke(candidates.app, argumentos)
    assert resultado.exit_code == 2
