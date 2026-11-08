"""Exporta del Model Registry el modelo que empaqueta la imagen del servicio de predicción.

Resuelve ``models:/<nombre>@<alias>`` en el servidor de MLflow, descarga esa versión a
``models/p1/`` y guarda en ``metadata.json`` el nombre, el alias y la versión resueltos.
La API (``app/model_service.py``) carga esa carpeta por omisión, así que la imagen de
Docker responde sin un servidor de MLflow.

Con el servidor de tracking encendido, desde la raíz del repositorio::

    make modelo-servicio

Se vuelve a ejecutar cada vez que ``champion`` apunte a otra versión.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import typer

from mibici.config import (
    ALIAS_CHAMPION,
    METADATOS_SERVICIO,
    MLFLOW_TRACKING_URI,
    MODELO_SERVICIO_DIR,
    MODELOS_REGISTRO,
)

app = typer.Typer(add_completion=False)


def exportar_modelo(
    nombre: str,
    alias: str,
    destino: Path,
    client: MlflowClient,
    reemplazar: bool = False,
) -> dict:
    """Descarga ``models:/<nombre>@<alias>`` a ``destino`` y escribe ``metadata.json``.

    El alias se resuelve una sola vez y se descarga esa versión exacta, para que los
    metadatos correspondan al modelo descargado aunque el alias cambie a la mitad. La
    descarga se hace en una carpeta temporal junto a ``destino``: si falla, el modelo
    anterior queda intacto. Si ``destino`` ya tiene archivos, sólo se sustituyen con
    ``reemplazar=True``.
    """
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()) and not reemplazar:
        raise FileExistsError(
            f"{destino} ya contiene un modelo; usa --reemplazar para sustituirlo."
        )

    version = client.get_model_version_by_alias(name=nombre, alias=alias)
    metadatos = {
        "model_name": nombre,
        "alias": alias,
        "version": str(version.version),
        "model_uri": f"models:/{nombre}@{alias}",
        "run_id": version.run_id,
        "exportado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destino.parent) as temporal:
        descarga = Path(temporal) / destino.name
        descarga.mkdir()
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{nombre}/{version.version}", dst_path=str(descarga)
        )
        if not (descarga / "MLmodel").is_file():
            raise FileNotFoundError(f"La descarga de {nombre} v{version.version} no trae MLmodel.")
        (descarga / METADATOS_SERVICIO).write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destino.exists():
            shutil.rmtree(destino)
        shutil.move(str(descarga), str(destino))

    return metadatos


@app.command()
def main(
    modelo: str = typer.Option(MODELOS_REGISTRO["P1"], help="Nombre en el Model Registry."),
    alias: str = typer.Option(ALIAS_CHAMPION),
    destino: Path = typer.Option(MODELO_SERVICIO_DIR),  # noqa: B008
    reemplazar: bool = typer.Option(False, help="Sustituye el modelo que ya esté en destino."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
) -> None:
    """Exporta el modelo registrado que empaqueta la imagen del servicio de predicción."""
    mlflow.set_tracking_uri(tracking_uri)
    metadatos = exportar_modelo(modelo, alias, destino, MlflowClient(tracking_uri), reemplazar)
    typer.echo(f"OK: {metadatos['model_uri']} (versión {metadatos['version']}) -> {destino}")


if __name__ == "__main__":
    app()
