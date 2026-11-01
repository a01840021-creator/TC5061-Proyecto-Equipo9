"""Pruebas de mibici/modeling/train.py: Git, etiquetas, parámetros y baselines en MLflow."""

import json

import pytest
from typer.testing import CliRunner

from mibici import config
from mibici.modeling import train


@pytest.fixture
def salida_git(monkeypatch):
    """Sustituye ``subprocess.check_output`` y registra con qué argumentos se llamó."""
    llamadas = []

    def preparar(resultado):
        def check_output(argumentos, **kwargs):
            llamadas.append((argumentos, kwargs))
            if isinstance(resultado, Exception):
                raise resultado
            return resultado

        monkeypatch.setattr(train.subprocess, "check_output", check_output)
        return llamadas

    return preparar


def test_obtener_commit_git_devuelve_el_sha_sin_espacios(salida_git):
    llamadas = salida_git("0123abcd\n")
    assert train.obtener_commit_git() == "0123abcd"
    argumentos, opciones = llamadas[0]
    assert argumentos == ["git", "rev-parse", "HEAD"]
    assert opciones["cwd"] == config.PROJ_ROOT
    assert "shell" not in opciones


@pytest.mark.parametrize(
    "error",
    [train.subprocess.CalledProcessError(128, ["git"]), FileNotFoundError("git")],
)
def test_obtener_commit_git_sin_repositorio_o_sin_git(salida_git, error):
    salida_git(error)
    assert train.obtener_commit_git() == "desconocido"


def test_obtener_autor_usa_el_nombre_de_git(salida_git):
    llamadas = salida_git("Persona Autora\n")
    assert train.obtener_autor() == "Persona Autora"
    assert llamadas[0][0] == ["git", "config", "user.name"]


@pytest.mark.parametrize(
    "resultado",
    ["", "   \n", train.subprocess.CalledProcessError(1, ["git"]), FileNotFoundError("git")],
)
def test_obtener_autor_cae_al_usuario_del_sistema(salida_git, monkeypatch, resultado):
    salida_git(resultado)
    monkeypatch.setattr(train.getpass, "getuser", lambda: "usuario_del_sistema")
    assert train.obtener_autor() == "usuario_del_sistema"


def test_etiquetas_comunes(monkeypatch):
    monkeypatch.setattr(train, "obtener_commit_git", lambda: "abc123")
    monkeypatch.setattr(train, "obtener_autor", lambda: "autor")
    assert train.etiquetas("p2", "regla", "baseline") == {
        "equipo": config.EQUIPO,
        "question": "P2",
        "model_family": "regla",
        "stage_of_work": "baseline",
        "git_commit": "abc123",
        "autor": "autor",
    }


def test_parametros_comunes_con_y_sin_validacion():
    parametros = train.parametros_comunes("v1")
    assert parametros == {
        "fecha_inicio": config.FECHA_INICIO,
        "fecha_fin": config.FECHA_FIN,
        "periodo_train": config.PERIODO_TRAIN,
        "periodo_validation": config.PERIODO_VALIDACION,
        "periodo_test": config.PERIODO_TEST,
        "version_datos": "v1",
        "semilla": config.SEMILLA,
    }
    sin_validacion = train.parametros_comunes("v1", con_validacion=False)
    assert "periodo_validation" not in sin_validacion
    assert len(sin_validacion) == len(parametros) - 1


def test_guardar_json(tmp_path):
    destino = tmp_path / "metricas.json"
    train.guardar_json({"test_mae": 1.5, "test_r2": -0.25}, destino)
    assert json.loads(destino.read_text(encoding="utf-8")) == {"test_mae": 1.5, "test_r2": -0.25}


# Baselines con MLflow temporal --------------------------------------------------------


def ultima_corrida(client):
    corridas = client.search_runs([client.experimento_id], order_by=["attributes.start_time DESC"])
    return corridas[0]


def artefactos(client, run_id, carpeta):
    return {a.path.split("/")[-1] for a in client.list_artifacts(run_id, carpeta)}


def test_baseline_p1_registra_metricas_de_validacion_y_prueba(datos_sinteticos, mlflow_temporal):
    train.ejecutar_p1(*datos_sinteticos, "v-prueba")
    corrida = ultima_corrida(mlflow_temporal)
    assert corrida.info.run_name == "P1_baseline_rezago_semanal"
    assert corrida.data.tags["stage_of_work"] == "baseline"
    assert corrida.data.params["modelo"] == "naive_rezago_168h"
    assert corrida.data.params["version_datos"] == "v-prueba"
    assert {"val_mae", "test_mae", "val_wape", "test_bias"} <= set(corrida.data.metrics)
    assert artefactos(mlflow_temporal, corrida.info.run_id, "evidencias_p1") == {
        "demanda_p1.png",
        "demanda_horaria_p1.csv",
        "metricas_p1.json",
    }


def test_baseline_p2_registra_umbral_y_artefactos(datos_sinteticos, mlflow_temporal):
    train.ejecutar_p2(*datos_sinteticos, "v-prueba", umbral=0.4)
    corrida = ultima_corrida(mlflow_temporal)
    assert corrida.data.params["umbral_capacidad"] == "0.4"
    assert corrida.data.params["horizonte_horas"] == str(config.HORIZONTE_P2_HORAS)
    assert {"val_average_precision", "test_recall"} <= set(corrida.data.metrics)
    assert artefactos(mlflow_temporal, corrida.info.run_id, "evidencias_p2") == {
        "matriz_confusion_p2.png",
        "riesgo_diario_p2.csv",
        "metricas_p2.json",
    }


def test_baseline_p3_registra_modelo_y_tabla_de_grupos(
    datos_p3, mlflow_temporal, modelos_capturados
):
    train.ejecutar_p3(*datos_p3, "v-prueba", n_clusters=4)
    corrida = ultima_corrida(mlflow_temporal)
    assert corrida.info.run_name == "P3_baseline_kmeans_k4"
    assert "periodo_validation" not in corrida.data.params
    assert corrida.data.params["n_estaciones"] == "32"
    assert set(corrida.data.metrics) == {
        "train_silhouette",
        "train_davies_bouldin",
        "train_calinski_harabasz",
        "train_inercia",
    }
    assert artefactos(mlflow_temporal, corrida.info.run_id, "evidencias_p3") == {
        "mapa_clusters_p3.png",
        "perfiles_clusters_p3.png",
        "estaciones_clusters_p3.csv",
        "perfiles_promedio_p3.csv",
        "metricas_p3.json",
    }
    assert modelos_capturados[0].named_steps["kmeans"].random_state == config.SEMILLA


def test_main_rechaza_pregunta_invalida_antes_de_conectarse():
    resultado = CliRunner().invoke(train.app, ["--pregunta", "P4"])
    assert resultado.exit_code == 2
    assert "P1, P2, P3 o all" in resultado.output
