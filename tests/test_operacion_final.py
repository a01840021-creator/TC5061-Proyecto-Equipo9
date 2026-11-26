"""Pruebas unitarias para la verificación operativa y evidencias de despliegue final."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mibici.config import PROJ_ROOT
from mibici.operacion_final import (
    PNG_SIGNATURE,
    app,
    construir_reporte_operacion,
    verificar_docker_compose,
    verificar_dockerfile,
    verificar_evidencias_carga_y_recursos,
    verificar_fastapi_contrato,
    verificar_scripts_y_locust,
)


def test_fastapi_contrato_responde_salud_prediccion_y_422():
    """La API carga el champion empaquetado, predice y rechaza entradas inválidas."""
    verif, resumen = verificar_fastapi_contrato()

    assert verif.aprobado is True
    assert resumen["model_name"] == "mibici-equipo09-p1-demanda"
    assert resumen["alias"] == "champion"
    assert resumen["version"] == "1"
    assert resumen["rechazo_invalido_http"] == 422
    assert resumen["prediccion_caso_demo"] > 0


def test_dockerfile_y_compose_del_repositorio_cumplen_contratos():
    """Dockerfile y docker-compose.yml mantienen hardening, red, volumen y límites."""
    verif_docker = verificar_dockerfile(PROJ_ROOT)
    verif_compose, resumen_compose = verificar_docker_compose(PROJ_ROOT)

    assert verif_docker.aprobado is True
    assert verif_compose.aprobado is True
    assert set(resumen_compose["servicios"]) >= {"api", "mlflow", "mlflow-init"}
    assert resumen_compose["limites_api"]["cpus"] == 1.0
    assert resumen_compose["limites_api"]["mem_limit"] == "1g"


def test_dockerfile_detecta_falta_de_usuario_no_root(tmp_path: Path):
    """Sin USER api ni HEALTHCHECK la verificación debe marcar fallo."""
    (tmp_path / "Dockerfile").write_text("FROM python:3.10.20-slim\n", encoding="utf-8")

    verif = verificar_dockerfile(tmp_path)

    assert verif.aprobado is False
    assert "USER api" in verif.detalle


def test_scripts_y_locust_validan_casos_contra_pydantic():
    """Existen Locust y scripts operativos, y todos los casos cumplen DemandaEntrada."""
    verif = verificar_scripts_y_locust(PROJ_ROOT)

    assert verif.aprobado is True
    assert "DemandaEntrada" in verif.detalle


def test_evidencias_carga_y_recursos_preservadas_son_validas():
    """Los artefactos oficiales de carga y docker stats existen y cumplen KPIs."""
    resultados, resumen = verificar_evidencias_carga_y_recursos(PROJ_ROOT)

    assert len(resultados) == 2
    assert all(r.aprobado for r in resultados)
    assert resumen["muestras_csv"] > 0
    assert resumen["resumen_carga"]["tasa_error_pct"] == 0.0
    assert resumen["resumen_carga"]["solicitudes_totales"] > 0


def test_evidencias_carga_detectan_png_corrupto(tmp_path: Path):
    """Un PNG sin firma válida debe marcarse como evidencia inconsistente."""
    carpeta = tmp_path / "reports/load_testing"
    carpeta.mkdir(parents=True)
    (carpeta / "load_test_summary.json").write_text("{}", encoding="utf-8")
    (carpeta / "resources.csv").write_text("timestamp,container\n", encoding="utf-8")
    (carpeta / "locust_performance.png").write_bytes(b"no-es-png")
    (carpeta / "resource_usage.png").write_bytes(PNG_SIGNATURE)

    resultados, _ = verificar_evidencias_carga_y_recursos(tmp_path)

    assert resultados[0].aprobado is False
    assert "locust_performance.png" in resultados[0].detalle


def test_cli_operacion_final_escribe_json_estricto(tmp_path: Path):
    """El CLI escribe el JSON de verificación operativa con estado aprobado."""
    destino = tmp_path / "operacion_evidencia.json"
    runner = CliRunner()

    resultado = runner.invoke(app, ["--salida", str(destino)])

    assert resultado.exit_code == 0, resultado.output
    assert destino.is_file()
    contenido = json.loads(destino.read_text(encoding="utf-8"))
    assert contenido["estado"] == "aprobado"
    assert contenido["bloque"] == "despliegue_operacion_y_demo"
    assert all(v["aprobado"] for v in contenido["verificaciones"])


def test_reporte_versionado_coincide_con_contratos_actuales():
    """El reporte operativo versionado en reports/final/ está actualizado y aprobado."""
    ruta = PROJ_ROOT / "reports/final/operacion_evidencia.json"
    assert ruta.is_file()
    guardado = json.loads(ruta.read_text(encoding="utf-8"))
    actual = construir_reporte_operacion(PROJ_ROOT)

    assert guardado["estado"] == "aprobado"
    assert guardado["fastapi"]["version"] == actual["fastapi"]["version"]
    assert (
        guardado["evidencia_carga_y_recursos"]["resumen_carga"]["solicitudes_totales"]
        == actual["evidencia_carga_y_recursos"]["resumen_carga"]["solicitudes_totales"]
    )
