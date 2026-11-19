"""Pruebas del escenario de carga y del resumen operativo."""

import json
import os
from pathlib import Path

# PyTest ya cargó librerías de red; evita que importar Locust las modifique durante
# la colección. La ejecución real de ``python -m locust`` conserva su parche normal.
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

from app.schemas import DemandaEntrada
from locustfile import CASOS_P1, EtapasCarga
import pandas as pd

from mibici.load_testing.report import (
    graficar_locust,
    graficar_recursos,
    resumir_ejecucion,
)


def test_casos_locust_cumplen_contrato_pydantic():
    """Las solicitudes de carga representan entradas aceptadas por la API."""
    assert len(CASOS_P1) >= 3
    for caso in CASOS_P1:
        validado = DemandaEntrada.model_validate(caso)
        assert validado.estacion_id >= 1


def test_perfil_smoke_es_creciente_y_termina(monkeypatch):
    """El perfil rápido aumenta concurrencia y tiene duración acotada."""
    monkeypatch.setenv("MIBICI_LOAD_PROFILE", "smoke")
    forma = EtapasCarga()
    assert forma.duration == 20
    assert [etapa["users"] for etapa in forma.stages] == [2, 5]


def test_resumen_locust_y_recursos(tmp_path: Path):
    """El resumen entrega métricas reproducibles para análisis y FinOps."""
    stats = tmp_path / "locust_stats.csv"
    history = tmp_path / "locust_stats_history.csv"
    resources = tmp_path / "resources.csv"

    pd.DataFrame(
        [
            {
                "Name": "POST /predict/p1",
                "Request Count": 100,
                "Failure Count": 2,
                "Requests/s": 10.5,
                "Average Response Time": 25.0,
                "50%": 20,
                "95%": 45,
                "99%": 60,
            }
        ]
    ).to_csv(stats, index=False)
    pd.DataFrame(
        [
            {
                "Timestamp": 1000,
                "Name": "POST /predict/p1",
                "Requests/s": 8.0,
                "Failures/s": 0.0,
                "50%": 20,
                "95%": 45,
                "99%": 60,
                "User Count": 5,
            },
            {
                "Timestamp": 1010,
                "Name": "POST /predict/p1",
                "Requests/s": 10.5,
                "Failures/s": 0.2,
                "50%": 22,
                "95%": 48,
                "99%": 65,
                "User Count": 10,
            },
        ]
    ).to_csv(history, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-11-19T00:00:00Z",
                "container": "mibici-api",
                "cpu_percent": "10%",
                "memory_usage": "100MiB / 1GiB",
                "memory_percent": "9.8%",
            },
            {
                "timestamp": "2026-11-19T00:00:02Z",
                "container": "mibici-api",
                "cpu_percent": "20%",
                "memory_usage": "110MiB / 1GiB",
                "memory_percent": "10.7%",
            },
        ]
    ).to_csv(resources, index=False)

    resultado = resumir_ejecucion(stats, history, resources)

    assert resultado["solicitudes_exitosas"] == 98
    assert resultado["tasa_error_pct"] == 2.0
    assert resultado["duracion_segundos"] == 10.0
    assert resultado["recursos_por_contenedor"]["mibici-api"]["cpu_promedio_pct"] == 15.0
    json.dumps(resultado)

    grafica_locust = graficar_locust(history, tmp_path / "locust.png")
    grafica_recursos = graficar_recursos(resources, tmp_path / "resources.png")
    assert grafica_locust.stat().st_size > 0
    assert grafica_recursos.stat().st_size > 0


def test_compose_permite_host_interno_de_mlflow():
    """La política de hosts permite comunicación desde la red de Compose."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "--allowed-hosts" in compose
    assert "mlflow:5000" in compose
