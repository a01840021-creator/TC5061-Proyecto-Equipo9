"""Servicio para cargar modelos desde MLflow Model Registry."""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mibici.config import (
    ALIAS_CHAMPION,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRO,
)


class ModelService:
    """Carga y administra un modelo registrado en MLflow."""

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
            or f"models:/{MODELOS_REGISTRO['P1']}@{ALIAS_CHAMPION}"
        )

        self.model = None
        self.metadata = {
            "model_name": None,
            "alias": None,
            "version": None,
            "model_uri": self.model_uri,
            "status": "not_loaded",
        }

    def load(self):
        """Carga el modelo registrado y obtiene sus metadatos.

        Si ya se cargó correctamente, reutiliza la instancia.
        """
        if self.model is not None:
            return self.model

        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            if not self.model_uri.startswith("models:/"):
                raise ValueError(
                    "MODEL_URI debe utilizar el formato models:/..."
                )

            referencia = self.model_uri[len("models:/"):]
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

            modelo = mlflow.sklearn.load_model(self.model_uri)

            self.model = modelo
            self.metadata = {
                "model_name": nombre,
                "alias": alias,
                "version": str(version.version),
                "model_uri": self.model_uri,
                "status": "loaded",
            }

            return self.model

        except Exception as exc:
            raise RuntimeError(
                f"No fue posible cargar el modelo desde MLflow "
                f"({self.model_uri}): {exc}"
            ) from exc
