"""Challenger reentrenado de P1: herencia del champion, registro, aliases y linaje."""

from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404  # solo para simular la salida de git status
from types import SimpleNamespace

import mlflow
from mlflow.models import get_model_info, infer_signature
import mlflow.sklearn
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.config import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    MODELOS_REGISTRO,
    PERIODO_AB,
    PERIODO_TRAIN_CANDIDATOS,
    PERIODO_TRAIN_CHALLENGER,
    PROJ_ROOT,
    SEMILLA,
)
from mibici.modeling import challenger
from mibici.modeling.candidates import FEATURES_P1, construir_regresor, limpiar_x
from mibici.monitoring.drift import cortar

NOMBRE = MODELOS_REGISTRO["P1"]
# Distintos de los valores por defecto, para comprobar que se heredan del champion
PARAMETROS_CHAMPION = {
    "modelo": "hist_gradient_boosting",
    "learning_rate": "0.1",
    "max_iter": "15",
    "max_leaf_nodes": "7",
    "l2_regularization": "0.5",
}
PARAMETROS_SEGUNDA = {**PARAMETROS_CHAMPION, "max_iter": "5", "max_leaf_nodes": "3"}
# Los candidatos registran la semilla global junto con sus hiperparámetros
SEMILLA_CANDIDATOS = "42"


def registrar_candidata(tablon: pd.DataFrame, parametros: dict, nombre_run: str) -> str:
    """Corrida candidata de P1 como las de candidates.py, con su modelo y su firma.

    Entrena con la semilla que registra: la de ``parametros`` si la trae, si no la global.
    """
    parametros = {"semilla": SEMILLA_CANDIDATOS, **parametros}
    train = cortar(tablon, PERIODO_TRAIN_CANDIDATOS)
    x = limpiar_x(train, FEATURES_P1)
    modelo = construir_regresor(parametros, semilla=int(parametros["semilla"]))
    modelo.fit(x, train["demanda"])
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.log_params({**parametros, "version_datos": "v-prueba"})
        mlflow.log_metric("val_mae", 0.5)
        ejemplo = x.head(5)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
            pip_requirements=[],
        )
    return run.info.run_id


def preparar_registro(client, tablon, monkeypatch, parametros_champion=PARAMETROS_CHAMPION):
    """Registra el champion (v1) y el challenger original (v2) de P1 en el registry temporal."""
    corridas = {}
    for alias, nombre_run, parametros in (
        (ALIAS_CHAMPION, "P1_hgb_lr010_iter15_leaf7", parametros_champion),
        (ALIAS_CHALLENGER, "P1_hgb_lr010_iter5_leaf3", PARAMETROS_SEGUNDA),
    ):
        run_id = registrar_candidata(tablon, parametros, nombre_run)
        version = mlflow.register_model(f"runs:/{run_id}/model", NOMBRE).version
        client.set_registered_model_alias(NOMBRE, alias, version)
        corridas[alias] = run_id
    # Sin inferir requisitos de pip en un subproceso, que tarda varios segundos
    monkeypatch.setattr(
        mlflow.sklearn, "log_model", partial(mlflow.sklearn.log_model, pip_requirements=[])
    )
    client.corridas = corridas
    return client


@pytest.fixture
def registro_p1(mlflow_registro_temporal, tablon_p1, monkeypatch):
    """Registry temporal con el champion (v1) y el challenger original (v2) de P1."""
    return preparar_registro(mlflow_registro_temporal, tablon_p1, monkeypatch)


@pytest.fixture
def interpretacion(tmp_path):
    """Interpretación del drift con la misma estructura que la de reports/drift/."""
    detalle = {
        c: {"valor": 0.15, "umbral": 0.1, "drift": True}
        for c in ("demanda", "rezago_168h", "rezago_24h", "rezago_48h")
    }
    informe = {
        "interpretaciones": {
            "test": {
                "periodo_actual": "2026-09-01/2026-09-30",
                "entradas_criticas_con_drift": ["rezago_168h", "rezago_24h", "rezago_48h"],
                "objetivo_con_drift": True,
                "detalle_drift": detalle,
            }
        },
        "continuous_training": {"accion": "evaluar_challenger"},
    }
    ruta = tmp_path / "interpretacion_drift.json"
    ruta.write_text(json.dumps(informe), encoding="utf-8")
    return ruta


def test_hereda_hiperparametros_registra_v3_y_mueve_challenger(
    registro_p1, tablon_p1, interpretacion
):
    client = registro_p1

    resultado = challenger.entrenar_y_registrar(
        tablon_p1, client, "v-prueba", NOMBRE, ruta_interpretacion=interpretacion
    )

    assert resultado["version"] == "3"
    assert str(client.get_model_version_by_alias(NOMBRE, ALIAS_CHALLENGER).version) == "3"
    assert str(client.get_model_version_by_alias(NOMBRE, ALIAS_CHAMPION).version) == "1"
    assert (
        client.get_model_version_by_alias(NOMBRE, ALIAS_CHAMPION).run_id
        == (client.corridas[ALIAS_CHAMPION])
    )

    run = client.get_run(resultado["run_id"])
    for clave, valor in PARAMETROS_CHAMPION.items():
        assert run.data.params[clave] == valor
    assert run.data.params["periodo_train"] == PERIODO_TRAIN_CHALLENGER
    assert run.data.params["periodo_evaluacion"] == PERIODO_AB
    assert run.data.params["version_datos"] == "v-prueba"
    assert run.data.params["semilla"] == "42"
    assert run.info.run_name == "P1_challenger_hgb_lr010_iter15_leaf7_hasta_20260915"

    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}@{ALIAS_CHALLENGER}")
    assert modelo.get_params()["learning_rate"] == 0.1
    assert modelo.get_params()["max_iter"] == 15
    assert modelo.get_params()["max_leaf_nodes"] == 7
    assert modelo.get_params()["l2_regularization"] == 0.5
    assert modelo.get_params()["random_state"] == 42

    assert set(resultado["metricas"]) == {"ab_mae", "ab_rmse", "ab_wape", "ab_r2", "ab_bias"}
    assert run.data.metrics == pytest.approx(resultado["metricas"])
    firma = get_model_info(f"models:/{NOMBRE}/3").signature
    assert firma.inputs.input_names() == FEATURES_P1
    assert {c.type.name for c in firma.inputs.inputs} == {"double"}


def test_escribe_las_etiquetas_de_linaje(registro_p1, tablon_p1, interpretacion):
    client = registro_p1

    resultado = challenger.entrenar_y_registrar(
        tablon_p1, client, "v-prueba", NOMBRE, ruta_interpretacion=interpretacion
    )

    etiquetas_run = client.get_run(resultado["run_id"]).data.tags
    assert etiquetas_run["stage_of_work"] == "challenger"
    assert etiquetas_run["question"] == "P1"
    assert etiquetas_run["model_family"] == "hist_gradient_boosting"
    assert etiquetas_run["parent_model"] == NOMBRE
    assert etiquetas_run["parent_version"] == "1"
    assert etiquetas_run["parent_alias"] == ALIAS_CHAMPION
    assert etiquetas_run["parent_run_id"] == client.corridas[ALIAS_CHAMPION]
    motivo = etiquetas_run["retrain_reason"]
    assert "rezago_24h, rezago_48h, rezago_168h y demanda" in motivo
    assert "interpretacion_drift.json" in motivo

    version = client.get_model_version(NOMBRE, "3")
    assert version.tags["source_run_id"] == resultado["run_id"]
    assert version.tags["parent_version"] == "1"
    assert version.tags["parent_model"] == NOMBRE
    assert version.tags["train_period"] == PERIODO_TRAIN_CHALLENGER
    assert version.tags["evaluation_period"] == PERIODO_AB
    assert version.tags["data_version"] == "v-prueba"
    assert version.tags["selection_metric"] == "ab_mae"
    assert {"question", "git_commit", "experiment", "selection_metric_value"} <= set(version.tags)
    assert "Challenger reentrenado" in version.description
    assert client.corridas[ALIAS_CHAMPION] in version.description


def test_hereda_la_semilla_del_champion_aunque_difiera_de_la_global(
    mlflow_registro_temporal, tablon_p1, monkeypatch, interpretacion
):
    """Con un champion entrenado con semilla 7, el challenger usa y registra 7, no 42."""
    client = preparar_registro(
        mlflow_registro_temporal,
        tablon_p1,
        monkeypatch,
        parametros_champion={**PARAMETROS_CHAMPION, "semilla": "7"},
    )
    assert client.get_run(client.corridas[ALIAS_CHAMPION]).data.params["semilla"] == "7"
    padre = mlflow.sklearn.load_model(f"models:/{NOMBRE}@{ALIAS_CHAMPION}")
    assert padre.get_params()["random_state"] == 7

    resultado = challenger.entrenar_y_registrar(
        tablon_p1, client, "v-prueba", NOMBRE, ruta_interpretacion=interpretacion
    )

    assert client.get_run(resultado["run_id"]).data.params["semilla"] == "7"
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{resultado['version']}")
    assert modelo.get_params()["random_state"] == 7


@pytest.mark.parametrize(
    ("parametros", "mensaje"),
    [
        ({}, "no registró semilla"),
        ({"semilla": "siete"}, "no es un entero"),
        ({"semilla": "-1"}, "fuera de 0"),
        ({"semilla": str(2**32)}, "fuera de 0"),
    ],
)
def test_falla_si_la_semilla_del_champion_no_es_valida(parametros, mensaje):
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc"),
        data=SimpleNamespace(params={**PARAMETROS_CHAMPION, **parametros}),
    )

    with pytest.raises(ValueError, match=mensaje):
        challenger.configuracion_heredada(run)


def test_configuracion_heredada_incluye_la_semilla_como_entero():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc"),
        data=SimpleNamespace(params={**PARAMETROS_CHAMPION, "semilla": "123"}),
    )

    configuracion = challenger.configuracion_heredada(run)

    assert configuracion == {**PARAMETROS_CHAMPION, "semilla": 123}


def test_construir_regresor_conserva_la_semilla_global_por_omision():
    """Los candidatos no pasan semilla y siguen usando la global."""
    assert construir_regresor(PARAMETROS_CHAMPION).get_params()["random_state"] == SEMILLA
    assert construir_regresor(PARAMETROS_CHAMPION, semilla=7).get_params()["random_state"] == 7


def test_adjunta_la_instantanea_del_codigo_con_su_hash(registro_p1, tablon_p1, interpretacion):
    client = registro_p1

    resultado = challenger.entrenar_y_registrar(
        tablon_p1, client, "v-prueba", NOMBRE, ruta_interpretacion=interpretacion
    )

    esperado = hashlib.sha256(Path(challenger.__file__).read_bytes()).hexdigest()
    etiquetas_run = client.get_run(resultado["run_id"]).data.tags
    version = client.get_model_version(NOMBRE, resultado["version"])
    for etiquetas in (etiquetas_run, version.tags):
        assert etiquetas["code_snapshot_path"] == "codigo/challenger.py"
        assert etiquetas["code_snapshot_sha256"] == esperado
        assert etiquetas["git_dirty"] in {"true", "false", "desconocido"}
    artefactos = [a.path for a in client.list_artifacts(resultado["run_id"], "codigo")]
    assert artefactos == ["codigo/challenger.py"]
    local = mlflow.artifacts.download_artifacts(
        run_id=resultado["run_id"], artifact_path="codigo/challenger.py"
    )
    assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == esperado


@pytest.mark.parametrize(
    ("salida", "esperado"),
    [(" M mibici/config.py\n", "true"), ("?? nuevo.py\n", "true"), ("", "false")],
)
def test_arbol_git_con_cambios_lee_git_status(monkeypatch, salida, esperado):
    llamadas = []

    def falso_check_output(argumentos, **kwargs):
        llamadas.append(argumentos)
        return salida

    monkeypatch.setattr(challenger.subprocess, "check_output", falso_check_output)

    assert challenger.arbol_git_con_cambios() == esperado
    assert llamadas == [["git", "--no-optional-locks", "status", "--porcelain"]]


def test_arbol_git_con_cambios_sin_git(monkeypatch):
    def sin_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(challenger.subprocess, "check_output", sin_git)
    assert challenger.arbol_git_con_cambios() == "desconocido"

    def fuera_de_repositorio(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(challenger.subprocess, "check_output", fuera_de_repositorio)
    assert challenger.arbol_git_con_cambios() == "desconocido"


def test_falla_si_el_champion_no_es_hist_gradient_boosting():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc"),
        data=SimpleNamespace(params={"modelo": "ridge", "alpha": "1.0"}),
    )

    with pytest.raises(ValueError, match="sólo admite 'hist_gradient_boosting'"):
        challenger.configuracion_heredada(run)


def test_falla_si_faltan_hiperparametros_del_champion():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc"),
        data=SimpleNamespace(params={"modelo": "hist_gradient_boosting"}),
    )

    with pytest.raises(ValueError, match="no registró learning_rate"):
        challenger.configuracion_heredada(run)


def test_entrenamiento_y_evaluacion_no_se_traslapan(tablon_p1):
    train, evaluacion = challenger.separar_periodos(tablon_p1)

    assert train["fecha_hora"].min() >= pd.Timestamp("2026-04-01")
    assert train["fecha_hora"].max() == pd.Timestamp("2026-09-15 23:00")
    assert evaluacion["fecha_hora"].min() == pd.Timestamp("2026-09-16 00:00")
    assert evaluacion["fecha_hora"].max() == pd.Timestamp("2026-09-30 23:00")


def test_motivo_sale_de_la_interpretacion_del_repositorio():
    motivo = challenger.motivo_reentrenamiento()

    informe = json.loads(challenger.RUTA_INTERPRETACION.read_text(encoding="utf-8"))
    assert "rezago_24h, rezago_48h, rezago_168h y demanda" in motivo
    assert "Wasserstein normalizada > 0.1" in motivo
    assert informe["continuous_training"]["accion"] in motivo
    assert "reports/drift/interpretacion_drift.json" in motivo
    assert str(PROJ_ROOT) not in motivo


def test_motivo_por_omision_sin_interpretacion(tmp_path):
    motivo = challenger.motivo_reentrenamiento(tmp_path / "no_existe.json")

    assert motivo.startswith(challenger.MOTIVO_POR_OMISION)


def test_cli_imprime_version_corrida_y_metricas(registro_p1, datos_sinteticos, monkeypatch):
    client = registro_p1
    monkeypatch.setattr(challenger, "cargar_datos_procesados", lambda: datos_sinteticos)
    monkeypatch.setattr(challenger, "leer_version_datos", lambda: "v-prueba")
    experimento = client.get_experiment(client.experimento_id).name

    resultado = CliRunner().invoke(challenger.app, ["--experiment-name", experimento])

    assert resultado.exit_code == 0, resultado.output
    assert f"{NOMBRE}@{ALIAS_CHALLENGER}: versión 3, corrida " in resultado.output
    assert "ab_mae = " in resultado.output
    assert str(client.get_model_version_by_alias(NOMBRE, ALIAS_CHAMPION).version) == "1"
