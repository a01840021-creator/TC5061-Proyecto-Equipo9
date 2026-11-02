"""Datos sintéticos, tracking temporal de MLflow y bloqueo de red para todas las pruebas.

Ninguna prueba lee data/raw/ ni data/processed/, usa la red o un servidor de MLflow:

- los viajes y las estaciones se generan aquí con una semilla fija;
- cada prueba que usa MLflow recibe una base SQLite y una carpeta de artefactos dentro
  de su ``tmp_path``;
- ``sin_red`` hace fallar cualquier conexión de red durante una prueba.
"""

import os

# Antes de importar MLflow: sin telemetría ni mensajes de ayuda en las pruebas
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "true"

import socket
import uuid

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici.config import FECHA_FIN, FECHA_INICIO
from mibici.features import (
    generar_dataset_p1,
    generar_dataset_p2,
    generar_dataset_p3,
)

INICIO = pd.Timestamp(FECHA_INICIO)
SEGUNDOS_PERIODO = int((pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1) - INICIO).total_seconds())


class RedBloqueada(RuntimeError):
    """Una prueba intentó abrir una conexión de red."""


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    """Permite sockets locales de Unix y rechaza cualquier otra conexión."""
    conectar = socket.socket.connect

    def conectar_local(self, direccion):
        if self.family == getattr(socket, "AF_UNIX", None):
            return conectar(self, direccion)
        raise RedBloqueada(f"Conexión de red bloqueada en una prueba: {direccion!r}")

    def bloquear(*args, **kwargs):
        raise RedBloqueada("Conexión de red bloqueada en una prueba.")

    resolver = socket.getaddrinfo

    def resolver_local(host, *args, **kwargs):
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return resolver(host, *args, **kwargs)
        raise RedBloqueada(f"Resolución de nombres bloqueada en una prueba: {host!r}")

    monkeypatch.setattr(socket.socket, "connect", conectar_local)
    monkeypatch.setattr(socket.socket, "connect_ex", conectar_local)
    monkeypatch.setattr(socket, "create_connection", bloquear)
    monkeypatch.setattr(socket, "getaddrinfo", resolver_local)


def generar_viajes(
    n_estaciones: int = 4, n_viajes: int = 30_000, semilla: int = 0, sin_viajes: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes y estaciones sintéticos con el esquema de ``data/processed/``.

    Los viajes caen al azar entre abril y septiembre, duran de 2 a 60 minutos y unen
    estaciones al azar. Las últimas ``sin_viajes`` estaciones de la nomenclatura no
    tienen ningún viaje.
    """
    rng = np.random.default_rng(semilla)
    inicio = INICIO + pd.to_timedelta(np.sort(rng.integers(0, SEGUNDOS_PERIODO, n_viajes)), "s")
    duracion = pd.to_timedelta(rng.integers(120, 3_600, n_viajes), "s")
    viajes = pd.DataFrame(
        {
            "viaje_id": np.arange(1, n_viajes + 1, dtype="int64"),
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + duracion,
            "origen_id": rng.integers(1, n_estaciones + 1, n_viajes).astype("int64"),
            "destino_id": rng.integers(1, n_estaciones + 1, n_viajes).astype("int64"),
        }
    )
    total = n_estaciones + sin_viajes
    estaciones = pd.DataFrame(
        {
            "id": np.arange(1, total + 1, dtype="int64"),
            "name": [f"Estación {i}" for i in range(1, total + 1)],
            "latitude": 20.65 + 0.002 * np.arange(total),
            "longitude": -103.40 + 0.003 * np.arange(total),
            "dpcapacity": (6 + 2 * (np.arange(total) % 5)).astype("int64"),
        }
    )
    return viajes, estaciones


@pytest.fixture(scope="session")
def datos_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuatro estaciones con 30,000 viajes.

    Con cuatro estaciones, el entrenamiento de los candidatos (abril a julio) tiene más de
    10,000 filas y HistGradientBoosting activa la parada temprana, que separa una
    validación interna al azar. Así la semilla sí cambia el modelo y la prueba de
    reproducibilidad no pasa por casualidad.
    """
    return generar_viajes()


@pytest.fixture(scope="session")
def tablon_p1(datos_sinteticos) -> pd.DataFrame:
    return generar_dataset_p1(*datos_sinteticos)


@pytest.fixture(scope="session")
def tablon_p2(datos_sinteticos) -> pd.DataFrame:
    return generar_dataset_p2(*datos_sinteticos)


@pytest.fixture(scope="session")
def datos_p3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """30 estaciones con viajes y 2 sin ninguno, para agrupar con k de 5 o 6."""
    return generar_viajes(n_estaciones=30, n_viajes=6_000, semilla=1, sin_viajes=2)


@pytest.fixture(scope="session")
def perfiles_p3(datos_p3) -> pd.DataFrame:
    return generar_dataset_p3(*datos_p3, solo_train=True)


def uri_sqlite(carpeta) -> str:
    return f"sqlite:///{(carpeta / 'mlflow.db').as_posix()}"


@pytest.fixture(scope="session")
def uri_mlflow_sesion(tmp_path_factory) -> str:
    """Una base SQLite temporal para toda la sesión: crearla tarda alrededor de un segundo."""
    return uri_sqlite(tmp_path_factory.mktemp("mlflow"))


@pytest.fixture
def mlflow_temporal(uri_mlflow_sesion, tmp_path, monkeypatch):
    """Tracking de MLflow en una base SQLite temporal, con un experimento propio por prueba.

    Los artefactos quedan en ``tmp_path``. Al terminar se cierra cualquier corrida abierta
    y se restablece la URI anterior.
    """
    yield from preparar_mlflow(uri_mlflow_sesion, tmp_path, monkeypatch)


@pytest.fixture
def mlflow_registro_temporal(tmp_path, monkeypatch):
    """Como ``mlflow_temporal``, pero con una base propia de la prueba.

    Los modelos registrados son globales en la base, así que cada prueba del Model
    Registry necesita la suya para empezar sin versiones.
    """
    yield from preparar_mlflow(uri_sqlite(tmp_path), tmp_path, monkeypatch)


def preparar_mlflow(uri, tmp_path, monkeypatch):
    anterior = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    experimento = mlflow.create_experiment(
        f"pruebas-{uuid.uuid4().hex}", artifact_location=(tmp_path / "artefactos").as_uri()
    )
    mlflow.set_experiment(experiment_id=experimento)
    client = MlflowClient(uri)
    # Para que cada prueba busque sólo sus corridas
    client.experimento_id = experimento
    yield client
    while mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(anterior)


@pytest.fixture
def modelos_capturados(monkeypatch) -> list:
    """Sustituye ``mlflow.sklearn.log_model`` y guarda el estimador que recibe.

    ``log_model`` infiere los requisitos de pip en un subproceso y tarda varios segundos.
    Las pruebas de reproducibilidad sólo necesitan el estimador ajustado; la prueba que
    registra y carga el modelo completo no usa este fixture.
    """
    capturados = []

    def capturar(*args, **kwargs):
        capturados.append(kwargs["sk_model"])

    monkeypatch.setattr(mlflow.sklearn, "log_model", capturar)
    return capturados
