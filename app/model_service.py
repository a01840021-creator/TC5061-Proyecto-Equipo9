"""Servicio para cargar el modelo de la API.

Por omisión carga la carpeta ``models/p1/``, que ``make modelo-servicio`` exporta del Model
Registry y la imagen de Docker empaqueta. ``MODEL_URI`` también acepta una URI
``models:/<nombre>@<alias>`` para cargar el modelo directamente del Model Registry.
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

from mibici.config import (
    METADATOS_SERVICIO,
    MLFLOW_TRACKING_URI,
    MODELO_SERVICIO_DIR,
)


class ModelService:
    """Carga y administra el modelo exportado del Model Registry de MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_uri: str | None = None,
    ):
        self.tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or MLFLOW_TRACKING_URI
        )

        self.model_uri = (
            model_uri
            or os.getenv("MODEL_URI")
            or str(MODELO_SERVICIO_DIR)
        )

        self.model = None
        self.metadata = {
            "model_name": None,
            "alias": None,
            "version": None,
            "model_uri": self.model_uri,
            "status": "not_loaded",
        }

    @property
    def ready(self) -> bool:
        """Indica si existe un modelo cargado y listo para inferencia."""
        return self.model is not None and self.metadata["status"] == "loaded"

    def load(self):
        """Carga el modelo y obtiene sus metadatos.

        Si ``model_uri`` empieza con ``models:/`` los metadatos vienen del Model Registry;
        si es una carpeta exportada, de su ``metadata.json``. Si ya se cargó
        correctamente, reutiliza la instancia.
        """
        if self.model is not None:
            return self.model

        try:
            if self.model_uri.startswith("models:/"):
                metadatos = self._metadatos_registry()
            else:
                metadatos = self._metadatos_carpeta()

            modelo = load_model(self.model_uri)

            self.model = modelo
            self.metadata = {**metadatos, "status": "loaded"}

            return self.model

        except Exception as exc:
            self.metadata["status"] = "error"
            raise RuntimeError(
                f"No fue posible cargar el modelo ({self.model_uri}): {exc}"
            ) from exc

    def _metadatos_registry(self) -> dict:
        """Resuelve el alias de ``models:/<nombre>@<alias>`` en el Model Registry."""
        mlflow.set_tracking_uri(self.tracking_uri)

        referencia = self.model_uri[len("models:/") :]
        nombre, separador, alias = referencia.partition("@")

        if not separador or not nombre or not alias:
            raise ValueError(
                "MODEL_URI debe especificar un alias, "
                "por ejemplo models:/nombre-modelo@champion"
            )

        client = MlflowClient(tracking_uri=self.tracking_uri)

        version = client.get_model_version_by_alias(
            name=nombre,
            alias=alias,
        )

        return {
            "model_name": nombre,
            "alias": alias,
            "version": str(version.version),
            "model_uri": self.model_uri,
        }

    def _metadatos_carpeta(self) -> dict:
        """Lee nombre, alias y versión del ``metadata.json`` de la carpeta exportada.

        ``model_uri`` es la referencia del Model Registry de la que se exportó la carpeta.
        """
        carpeta = Path(self.model_uri)
        if not carpeta.is_dir():
            raise ValueError(
                "MODEL_URI debe utilizar el formato models:/... o ser la carpeta de un "
                "modelo exportado con make modelo-servicio"
            )

        archivo = carpeta / METADATOS_SERVICIO
        if not archivo.is_file():
            raise FileNotFoundError(
                f"Falta {archivo}; regenera la carpeta con make modelo-servicio"
            )

        datos = json.loads(archivo.read_text(encoding="utf-8"))
        return {
            "model_name": datos["model_name"],
            "alias": datos["alias"],
            "version": str(datos["version"]),
            "model_uri": datos["model_uri"],
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Ejecuta inferencia con el modelo cargado y normaliza la salida numérica."""
        if not self.ready:
            raise RuntimeError("El modelo no está disponible para realizar inferencias.")
        return np.asarray(self.model.predict(frame), dtype="float64")
