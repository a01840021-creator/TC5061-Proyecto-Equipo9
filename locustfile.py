"""Escenario reproducible de carga para la API de predicción P1 de MiBici."""

from itertools import cycle
import os

from locust import HttpUser, LoadTestShape, between, task

CASOS_P1 = (
    {
        "estacion_id": 1,
        "dpcapacity": 20.0,
        "hora": 8,
        "dia_semana": 0,
        "es_fin_de_semana": 0,
        "rezago_24h": 5.0,
        "rezago_48h": 4.0,
        "rezago_168h": 6.0,
    },
    {
        "estacion_id": 51,
        "dpcapacity": 30.0,
        "hora": 18,
        "dia_semana": 4,
        "es_fin_de_semana": 0,
        "rezago_24h": 12.0,
        "rezago_48h": 9.0,
        "rezago_168h": 11.0,
    },
    {
        "estacion_id": 123,
        "dpcapacity": 25.0,
        "hora": 13,
        "dia_semana": 5,
        "es_fin_de_semana": 1,
        "rezago_24h": 7.0,
        "rezago_48h": 8.0,
        "rezago_168h": 5.0,
    },
)


class UsuarioMiBici(HttpUser):
    """Simula consumidores que consultan salud y predicen demanda."""

    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Asigna a cada usuario una secuencia local de entradas válidas."""
        self._casos = cycle(CASOS_P1)

    @task(10)
    def predecir_demanda(self):
        """Ejecuta la operación de negocio principal y valida su contrato."""
        with self.client.post(
            "/predict/p1",
            json=next(self._casos),
            name="POST /predict/p1",
            catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}: {respuesta.text[:200]}")
                return

            try:
                cuerpo = respuesta.json()
            except ValueError:
                respuesta.failure("La API no devolvió JSON válido")
                return

            requeridos = {"prediction", "model_name", "version", "latency_ms"}
            faltantes = requeridos.difference(cuerpo)
            if faltantes:
                respuesta.failure(f"Respuesta incompleta: {sorted(faltantes)}")
            elif cuerpo["latency_ms"] < 0:
                respuesta.failure("La latencia reportada por la API es inválida")

    @task(1)
    def consultar_salud(self):
        """Incluye una proporción pequeña de verificaciones de disponibilidad."""
        self.client.get("/health/ready", name="GET /health/ready")


class EtapasCarga(LoadTestShape):
    """Incrementa la concurrencia para localizar el punto de saturación."""

    PERFILES = {
        "smoke": (
            {"duration": 10, "users": 2, "spawn_rate": 1},
            {"duration": 20, "users": 5, "spawn_rate": 2},
        ),
        "standard": (
            {"duration": 60, "users": 5, "spawn_rate": 1},
            {"duration": 180, "users": 25, "spawn_rate": 5},
            {"duration": 300, "users": 50, "spawn_rate": 5},
            {"duration": 360, "users": 100, "spawn_rate": 10},
        ),
    }

    def __init__(self):
        super().__init__()
        perfil = os.getenv("MIBICI_LOAD_PROFILE", "standard").lower()
        if perfil not in self.PERFILES:
            permitidos = ", ".join(sorted(self.PERFILES))
            raise ValueError(
                f"MIBICI_LOAD_PROFILE={perfil!r} no es válido; use uno de: {permitidos}"
            )
        self.stages = self.PERFILES[perfil]

    @property
    def duration(self) -> int:
        """Duración total en segundos del perfil seleccionado."""
        return int(self.stages[-1]["duration"])

    def tick(self):
        """Devuelve usuarios y tasa de aparición de la etapa actual."""
        tiempo = self.get_run_time()
        for etapa in self.stages:
            if tiempo < etapa["duration"]:
                return etapa["users"], etapa["spawn_rate"]
        return None
