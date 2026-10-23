"""Registra los modelos seleccionados y asigna el alias ``champion``."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import typer

from mibici.config import MLFLOW_TRACKING_URI

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Seleccion:
    """Metadatos necesarios para promover un modelo de una corrida."""

    nombre: str
    run_id: str
    pregunta: str
    metrica_seleccion: str
    descripcion: str


def registrar_champion(client: MlflowClient, seleccion: Seleccion) -> str:
    """Registra una versión de forma idempotente y le asigna ``champion``."""
    run = client.get_run(seleccion.run_id)
    existentes = [
        version
        for version in client.search_model_versions(f"name='{seleccion.nombre}'")
        if version.run_id == seleccion.run_id
    ]
    if existentes:
        version = existentes[0]
    else:
        version = mlflow.register_model(
            model_uri=f"runs:/{seleccion.run_id}/model",
            name=seleccion.nombre,
        )

    client.update_registered_model(
        name=seleccion.nombre,
        description=seleccion.descripcion,
    )
    etiquetas = {
        "question": seleccion.pregunta,
        "git_commit": run.data.tags.get("git_commit", "desconocido"),
        "data_version": run.data.params.get("version_datos", "desconocida"),
        "selection_metric": seleccion.metrica_seleccion,
    }
    for clave, valor in etiquetas.items():
        client.set_model_version_tag(
            seleccion.nombre,
            version.version,
            clave,
            str(valor),
        )
    client.set_registered_model_alias(
        name=seleccion.nombre,
        alias="champion",
        version=version.version,
    )
    return str(version.version)


@app.command()
def main(
    p1_run_id: str = typer.Option(..., help="Run ganador de demanda."),
    p2_run_id: str = typer.Option(..., help="Run ganador de riesgo."),
    p3_run_id: str = typer.Option(..., help="Run ganador de clustering."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
) -> None:
    """Promueve los tres ganadores y verifica su carga por alias."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    selecciones = [
        Seleccion(
            "mibici-equipo09-p1-demanda",
            p1_run_id,
            "P1",
            "val_mae",
            "Predicción de demanda horaria por estación; seleccionado por menor MAE.",
        ),
        Seleccion(
            "mibici-equipo09-p2-riesgo",
            p2_run_id,
            "P2",
            "val_average_precision",
            "Clasificación de riesgo operativo; seleccionado por AP, recall y F1.",
        ),
        Seleccion(
            "mibici-equipo09-p3-grupos",
            p3_run_id,
            "P3",
            "train_silhouette",
            "Agrupación de estaciones; K-Means seleccionado principalmente por silhouette.",
        ),
    ]

    for seleccion in selecciones:
        version = registrar_champion(client, seleccion)
        typer.echo(f"{seleccion.nombre}: versión {version}, alias champion")

    for seleccion in selecciones:
        uri = f"models:/{seleccion.nombre}@champion"
        modelo = mlflow.sklearn.load_model(uri)
        typer.echo(f"OK: {uri} -> {type(modelo).__name__}")


if __name__ == "__main__":
    app()
