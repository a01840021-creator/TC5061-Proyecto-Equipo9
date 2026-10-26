"""Elige el mejor candidato de cada pregunta, lo registra y le asigna sus aliases.

La selección no recibe run IDs: busca en el experimento las corridas candidatas
terminadas del commit actual y las ordena por la métrica de validación definida en
``METRICAS_SELECCION``. La mejor recibe ``champion`` y la segunda ``challenger``.

En P3 sólo compiten por silhouette las corridas con todas las estaciones. Las corridas
con ``estaciones=activas`` se calculan sobre otro conjunto de estaciones y su
silhouette no es comparable.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.entities import Run
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import typer

from mibici.config import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ESTACIONES_TODAS,
    METRICAS_SELECCION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRO,
)
from mibici.modeling.train import obtener_commit_git

app = typer.Typer(add_completion=False)

DESCRIPCIONES = {
    "P1": (
        "Pronóstico de los viajes que inician en cada estación y hora del día siguiente, "
        "para planear la flota. Se elige la versión con menor MAE de validación."
    ),
    "P2": (
        "Probabilidad de que una estación se vacíe o se llene en las próximas dos horas, "
        "para priorizar el rebalanceo. Se elige la versión con mayor average precision "
        "de validación."
    ),
    "P3": (
        "Grupos de estaciones con patrones de uso parecidos, para planeación urbana. Se "
        "elige la versión con mayor silhouette entre las que agrupan todas las estaciones."
    ),
}


def filtro_candidatas(pregunta: str, commit: str) -> str:
    """Filtro de MLflow para las candidatas terminadas de una pregunta y un commit."""
    filtro = (
        f"tags.stage_of_work = 'candidate' and tags.question = '{pregunta}' "
        f"and tags.git_commit = '{commit}' and attributes.status = 'FINISHED'"
    )
    if pregunta == "P3":
        filtro += f" and params.estaciones = '{ESTACIONES_TODAS}'"
    return filtro


def ordenar_candidatas(runs: list[Run], metrica: str, mayor_es_mejor: bool) -> list[Run]:
    """Ordena de mejor a peor las corridas que registraron la métrica de selección."""
    con_metrica = [run for run in runs if metrica in run.data.metrics]
    return sorted(con_metrica, key=lambda run: run.data.metrics[metrica], reverse=mayor_es_mejor)


def registrar_version(
    client: MlflowClient, nombre: str, run: Run, pregunta: str, metrica: str
) -> str:
    """Registra el modelo de una corrida, de forma idempotente, con su linaje."""
    run_id = run.info.run_id
    existentes = [
        v for v in client.search_model_versions(f"name='{nombre}'") if v.run_id == run_id
    ]
    if existentes:
        version = existentes[0].version
    else:
        version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=nombre).version

    valor = run.data.metrics[metrica]
    commit = run.data.tags.get("git_commit", "desconocido")
    datos = run.data.params.get("version_datos", "desconocida")
    experimento = client.get_experiment(run.info.experiment_id).name
    client.update_model_version(
        name=nombre,
        version=version,
        description=(
            f"Corrida {run.info.run_name} ({run_id}) del experimento {experimento}; "
            f"modelo {run.data.tags.get('model_family', 'desconocido')}, "
            f"{metrica} = {valor:.6f}. Commit {commit}, versión de datos {datos}."
        ),
    )
    etiquetas = {
        "question": pregunta,
        "git_commit": commit,
        "data_version": datos,
        "source_run_id": run_id,
        "experiment": experimento,
        "selection_metric": metrica,
        "selection_metric_value": f"{valor:.6f}",
    }
    for clave, valor_etiqueta in etiquetas.items():
        client.set_model_version_tag(nombre, version, clave, valor_etiqueta)
    return str(version)


def promover_pregunta(
    client: MlflowClient, experiment_id: str, pregunta: str, commit: str
) -> None:
    """Elige champion y challenger de una pregunta y les asigna sus aliases."""
    nombre = MODELOS_REGISTRO[pregunta]
    metrica, mayor_es_mejor = METRICAS_SELECCION[pregunta]
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filtro_candidatas(pregunta, commit),
        max_results=1000,
    )
    ordenadas = ordenar_candidatas(runs, metrica, mayor_es_mejor)
    if not ordenadas:
        raise typer.BadParameter(
            f"No hay candidatas terminadas de {pregunta} con {metrica} en el commit {commit}. "
            f"Ejecuta primero la búsqueda de {pregunta.lower()}."
        )

    for alias, run in zip((ALIAS_CHAMPION, ALIAS_CHALLENGER), ordenadas):
        version = registrar_version(client, nombre, run, pregunta, metrica)
        client.set_registered_model_alias(name=nombre, alias=alias, version=version)
        typer.echo(
            f"{nombre}@{alias}: versión {version}, corrida {run.info.run_name} "
            f"({run.info.run_id}), {metrica} = {run.data.metrics[metrica]:.6f}"
        )
    # register_model crea el modelo registrado la primera vez
    client.update_registered_model(name=nombre, description=DESCRIPCIONES[pregunta])
    if len(ordenadas) < 2:
        typer.echo(f"{nombre}: sólo hay una candidata; no se asigna {ALIAS_CHALLENGER}.")


@app.command()
def main(
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Promueve el mejor candidato de P1, P2 y P3 y verifica su carga por alias."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", tracking_uri))
    client = MlflowClient()
    experimento = client.get_experiment_by_name(experiment_name)
    if experimento is None:
        raise typer.BadParameter(f"No existe el experimento {experiment_name}.")
    commit = obtener_commit_git()
    typer.echo(f"Seleccionando candidatas del commit {commit} en {experiment_name}")

    for pregunta in MODELOS_REGISTRO:
        promover_pregunta(client, experimento.experiment_id, pregunta, commit)

    for nombre in MODELOS_REGISTRO.values():
        uri = f"models:/{nombre}@{ALIAS_CHAMPION}"
        modelo = mlflow.sklearn.load_model(uri)
        typer.echo(f"OK: {uri} -> {modelo}")


if __name__ == "__main__":
    app()
