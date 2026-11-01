"""Pruebas de la selección de candidatas y del registro en el Model Registry."""

from types import SimpleNamespace

import mlflow
import mlflow.sklearn
import pytest
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
import typer
from typer.testing import CliRunner

from mibici.config import ALIAS_CHALLENGER, ALIAS_CHAMPION, ESTACIONES_TODAS, MODELOS_REGISTRO
from mibici.modeling import registry
from mibici.modeling.registry import filtro_candidatas, ordenar_candidatas, promover_pregunta


def corrida(run_id: str, **metricas: float):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(metrics=metricas)
    )


def test_menor_es_mejor_para_mae():
    runs = [corrida("a", val_mae=0.9), corrida("b", val_mae=0.7), corrida("c", val_mae=0.8)]
    ordenadas = ordenar_candidatas(runs, "val_mae", mayor_es_mejor=False)
    assert [r.info.run_id for r in ordenadas] == ["b", "c", "a"]


def test_mayor_es_mejor_para_average_precision():
    runs = [
        corrida("a", val_average_precision=0.2),
        corrida("b", val_average_precision=0.4),
        corrida("c", val_average_precision=0.3),
    ]
    ordenadas = ordenar_candidatas(runs, "val_average_precision", mayor_es_mejor=True)
    assert [r.info.run_id for r in ordenadas[:2]] == ["b", "c"]


def test_ignora_corridas_sin_la_metrica():
    runs = [corrida("a", test_mae=0.1), corrida("b", val_mae=0.7)]
    assert [r.info.run_id for r in ordenar_candidatas(runs, "val_mae", False)] == ["b"]


def test_filtro_exige_candidata_terminada_del_commit():
    filtro = filtro_candidatas("P1", "abc123")
    assert "tags.stage_of_work = 'candidate'" in filtro
    assert "tags.question = 'P1'" in filtro
    assert "tags.git_commit = 'abc123'" in filtro
    assert "attributes.status = 'FINISHED'" in filtro
    assert "params.estaciones" not in filtro


def test_p3_solo_compite_con_todas_las_estaciones():
    assert f"params.estaciones = '{ESTACIONES_TODAS}'" in filtro_candidatas("P3", "abc123")


# Registro con una base de MLflow temporal ---------------------------------------------

MODELOS_PRUEBA = {
    "P1": lambda: DummyRegressor().fit([[0.0], [1.0]], [0.0, 1.0]),
    "P2": lambda: DummyClassifier().fit([[0.0], [1.0]], [0, 1]),
    "P3": lambda: KMeans(n_clusters=2, n_init=1, random_state=0).fit([[0.0], [1.0], [5.0]]),
}


def candidata(client, pregunta, metrica, valor, commit="abc", **opciones):
    """Crea una corrida candidata terminada con un modelo pequeño registrado como artefacto."""
    with mlflow.start_run(run_name=opciones.get("nombre", f"{pregunta}_{valor}")) as corrida:
        mlflow.set_tags(
            {
                "stage_of_work": opciones.get("etapa", "candidate"),
                "question": pregunta,
                "git_commit": commit,
                "model_family": "dummy",
            }
        )
        mlflow.log_params({"version_datos": "v1", **opciones.get("parametros", {})})
        mlflow.log_metric(metrica, valor)
        mlflow.sklearn.log_model(
            sk_model=MODELOS_PRUEBA[pregunta](), name="model", pip_requirements=[]
        )
    if opciones.get("fallida"):
        client.set_terminated(corrida.info.run_id, status="FAILED")
    return corrida.info.run_id


def alias(client, pregunta, nombre_alias):
    return client.get_model_version_by_alias(MODELOS_REGISTRO[pregunta], nombre_alias)


def test_promover_asigna_champion_y_challenger_con_linaje(mlflow_registro_temporal):
    client = mlflow_registro_temporal
    peor = candidata(client, "P1", "val_mae", 0.9)
    mejor = candidata(client, "P1", "val_mae", 0.7)
    segunda = candidata(client, "P1", "val_mae", 0.8)
    # Quedan fuera: otro commit, corrida fallida y baseline, aunque tengan mejor métrica
    candidata(client, "P1", "val_mae", 0.1, commit="otro")
    candidata(client, "P1", "val_mae", 0.2, fallida=True)
    candidata(client, "P1", "val_mae", 0.3, etapa="baseline")

    promover_pregunta(client, client.experimento_id, "P1", "abc")

    champion = alias(client, "P1", ALIAS_CHAMPION)
    challenger = alias(client, "P1", ALIAS_CHALLENGER)
    assert champion.run_id == mejor
    assert challenger.run_id == segunda
    assert peor not in {
        v.run_id for v in client.search_model_versions("name='" + MODELOS_REGISTRO["P1"] + "'")
    }
    assert champion.tags == {
        "question": "P1",
        "git_commit": "abc",
        "data_version": "v1",
        "source_run_id": mejor,
        "experiment": client.get_experiment(client.experimento_id).name,
        "selection_metric": "val_mae",
        "selection_metric_value": "0.700000",
    }
    assert "val_mae = 0.700000" in champion.description
    modelo = client.get_registered_model(MODELOS_REGISTRO["P1"])
    assert modelo.description == registry.DESCRIPCIONES["P1"]


def test_promover_es_idempotente(mlflow_registro_temporal):
    client = mlflow_registro_temporal
    candidata(client, "P2", "val_average_precision", 0.4)
    candidata(client, "P2", "val_average_precision", 0.6)
    for _ in range(2):
        promover_pregunta(client, client.experimento_id, "P2", "abc")
    versiones = client.search_model_versions(f"name='{MODELOS_REGISTRO['P2']}'")
    assert len(versiones) == 2
    assert alias(client, "P2", ALIAS_CHAMPION).tags["selection_metric_value"] == "0.600000"


def test_p3_solo_compite_con_corridas_de_todas_las_estaciones(mlflow_registro_temporal):
    client = mlflow_registro_temporal
    todas = candidata(
        client, "P3", "train_silhouette", 0.3, parametros={"estaciones": ESTACIONES_TODAS}
    )
    candidata(client, "P3", "train_silhouette", 0.9, parametros={"estaciones": "activas"})
    promover_pregunta(client, client.experimento_id, "P3", "abc")
    assert alias(client, "P3", ALIAS_CHAMPION).run_id == todas


def test_con_una_sola_candidata_no_hay_challenger(mlflow_registro_temporal, capsys):
    client = mlflow_registro_temporal
    candidata(client, "P1", "val_mae", 0.5)
    promover_pregunta(client, client.experimento_id, "P1", "abc")
    assert "no se asigna challenger" in capsys.readouterr().out
    with pytest.raises(mlflow.exceptions.MlflowException):
        alias(client, "P1", ALIAS_CHALLENGER)


def test_sin_candidatas_del_commit_falla(mlflow_registro_temporal):
    client = mlflow_registro_temporal
    candidata(client, "P1", "val_mae", 0.5, commit="otro")
    with pytest.raises(typer.BadParameter, match="No hay candidatas"):
        promover_pregunta(client, client.experimento_id, "P1", "abc")


def test_main_registra_las_tres_preguntas_y_carga_los_champion(
    mlflow_registro_temporal, monkeypatch
):
    client = mlflow_registro_temporal
    for pregunta, (metrica, _) in registry.METRICAS_SELECCION.items():
        parametros = {"estaciones": ESTACIONES_TODAS} if pregunta == "P3" else {}
        candidata(client, pregunta, metrica, 0.5, parametros=parametros)
    monkeypatch.setattr(registry, "obtener_commit_git", lambda: "abc")
    nombre = client.get_experiment(client.experimento_id).name
    resultado = CliRunner().invoke(registry.app, ["--experiment-name", nombre])
    assert resultado.exit_code == 0, resultado.output
    for modelo in MODELOS_REGISTRO.values():
        assert f"OK: models:/{modelo}@champion" in resultado.output


def test_main_falla_si_no_existe_el_experimento(mlflow_registro_temporal):
    resultado = CliRunner().invoke(registry.app, ["--experiment-name", "no-existe"])
    assert resultado.exit_code == 2
