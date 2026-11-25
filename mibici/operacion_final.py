"""Verificación auditable de despliegue, operación y evidencias de carga de MiBici.

Comprueba de forma reproducible los contratos operativos del servicio (FastAPI, Dockerfile,
Docker Compose, Locust, scripts de demostración y evidencias preservadas de carga/recursos)
y escribe un reporte JSON en ``reports/final/operacion_evidencia.json``.

Por omisión se ejecuta sin red ni Docker, por lo que puede correr en cualquier clon limpio
y en CI. Con ``--verificar-vivo`` valida además los servicios levantados en local.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess  # nosec B404
from typing import Annotated, Any
from urllib.error import HTTPError, URLError
import urllib.request

from app.main import app as fastapi_app
from app.schemas import DemandaEntrada
from fastapi.testclient import TestClient
import typer
import yaml

from mibici.config import PROJ_ROOT

app = typer.Typer(add_completion=False)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ARCHIVOS_EVIDENCIA_CARGA = (
    "reports/load_testing/load_test_summary.json",
    "reports/load_testing/resources.csv",
    "reports/load_testing/locust_performance.png",
    "reports/load_testing/resource_usage.png",
)

SCRIPTS_OPERACION = (
    "locustfile.py",
    "scripts/smoke_test.sh",
    "scripts/run_load_test.sh",
    "scripts/monitor_resources.sh",
    "scripts/demo_final.sh",
)

CASO_DEMO_VALIDO: dict[str, Any] = {
    "estacion_id": 2,
    "dpcapacity": 15.0,
    "hora": 8,
    "dia_semana": 1,
    "es_fin_de_semana": 0,
    "rezago_24h": 3.0,
    "rezago_48h": 2.0,
    "rezago_168h": 4.0,
}

CASO_DEMO_INVALIDO: dict[str, Any] = {
    **CASO_DEMO_VALIDO,
    "hora": 24,
}


@dataclass(frozen=True)
class VerificacionOperativa:
    """Resultado serializable de una verificación de despliegue u operación."""

    nombre: str
    aprobado: bool
    detalle: str


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    contenido = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return contenido if isinstance(contenido, dict) else {}


def _leer_json(ruta: Path) -> dict[str, Any]:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise TypeError(f"{ruta} debe contener un objeto JSON")
    return contenido


def verificar_fastapi_contrato() -> tuple[VerificacionOperativa, dict[str, Any]]:
    """Valida la API FastAPI en proceso: salud, metadatos, predicción y rechazo 422."""
    with TestClient(fastapi_app) as cliente:
        r_live = cliente.get("/health/live")
        r_ready = cliente.get("/health/ready")
        r_info = cliente.get("/model-info")
        r_ok = cliente.post("/predict/p1", json=CASO_DEMO_VALIDO)
        r_err = cliente.post("/predict/p1", json=CASO_DEMO_INVALIDO)

    cuerpo_info = r_info.json() if r_info.status_code == 200 else {}
    cuerpo_ok = r_ok.json() if r_ok.status_code == 200 else {}

    aprobado = (
        r_live.status_code == 200
        and r_ready.status_code == 200
        and r_ready.json().get("model_status") == "loaded"
        and r_info.status_code == 200
        and cuerpo_info.get("alias") == "champion"
        and r_ok.status_code == 200
        and isinstance(cuerpo_ok.get("prediction"), (int, float))
        and r_err.status_code == 422
    )
    detalle = (
        f"/health/ready=200 (loaded), /model-info={cuerpo_info.get('model_name')}@"
        f"{cuerpo_info.get('alias')} v{cuerpo_info.get('version')}, "
        f"/predict/p1=200 (pred={cuerpo_ok.get('prediction')}), validación inválida=422"
        if aprobado
        else (
            f"live={r_live.status_code}, ready={r_ready.status_code}, "
            f"info={r_info.status_code}, pred={r_ok.status_code}, invalida={r_err.status_code}"
        )
    )
    resumen = {
        "model_name": cuerpo_info.get("model_name"),
        "alias": cuerpo_info.get("alias"),
        "version": cuerpo_info.get("version"),
        "prediccion_caso_demo": cuerpo_ok.get("prediction"),
        "rechazo_invalido_http": r_err.status_code,
    }
    return VerificacionOperativa("fastapi_contrato_y_prediccion", aprobado, detalle), resumen


def verificar_dockerfile(raiz: Path) -> VerificacionOperativa:
    """Verifica usuario sin privilegios, HEALTHCHECK, puerto y modelo empaquetado."""
    ruta = raiz / "Dockerfile"
    if not ruta.is_file():
        return VerificacionOperativa("dockerfile_seguridad", False, "No existe Dockerfile")
    texto = ruta.read_text(encoding="utf-8")
    requeridos = ("USER api", "HEALTHCHECK", "EXPOSE 8000", "models/p1/")
    faltantes = [item for item in requeridos if item not in texto]
    return VerificacionOperativa(
        "dockerfile_seguridad",
        not faltantes,
        "USER api, HEALTHCHECK, EXPOSE 8000 y models/p1/ presentes"
        if not faltantes
        else f"Faltan en Dockerfile: {', '.join(faltantes)}",
    )


def verificar_docker_compose(raiz: Path) -> tuple[VerificacionOperativa, dict[str, Any]]:
    """Verifica servicios, red, volumen persistente, healthchecks y límites en Compose."""
    ruta = raiz / "docker-compose.yml"
    if not ruta.is_file():
        return VerificacionOperativa(
            "docker_compose_orquestacion", False, "No existe docker-compose.yml"
        ), {}

    compose = _leer_yaml(ruta)
    servicios = compose.get("services", {})
    requeridos = ("api", "mlflow", "mlflow-init")
    faltantes = [nombre for nombre in requeridos if nombre not in servicios]

    problemas: list[str] = []
    if faltantes:
        problemas.append(f"servicios faltantes: {', '.join(faltantes)}")
    else:
        for nombre in ("api", "mlflow"):
            srv = servicios[nombre]
            for campo in ("healthcheck", "cpus", "mem_limit"):
                if campo not in srv:
                    problemas.append(f"{nombre} sin {campo}")
        if "mlflow-data" not in compose.get("volumes", {}):
            problemas.append("falta volumen mlflow-data")
        if "mibici-network" not in compose.get("networks", {}):
            problemas.append("falta red mibici-network")

    aprobado = not problemas
    resumen = (
        {
            "servicios": sorted(servicios.keys()),
            "puerto_api": servicios.get("api", {}).get("ports", []),
            "puerto_mlflow": servicios.get("mlflow", {}).get("ports", []),
            "limites_api": {
                "cpus": servicios.get("api", {}).get("cpus"),
                "mem_limit": servicios.get("api", {}).get("mem_limit"),
            },
            "limites_mlflow": {
                "cpus": servicios.get("mlflow", {}).get("cpus"),
                "mem_limit": servicios.get("mlflow", {}).get("mem_limit"),
            },
            "volumen_persistente": "mlflow-data",
            "red": "mibici-network",
        }
        if aprobado
        else {}
    )
    detalle = (
        "Servicios api, mlflow y mlflow-init con red, volumen, healthchecks y límites de CPU/RAM"
        if aprobado
        else "; ".join(problemas)
    )
    return VerificacionOperativa("docker_compose_orquestacion", aprobado, detalle), resumen


def _extraer_casos_locust(ruta_locust: Path) -> tuple[dict[str, Any], ...]:
    """Extrae la tupla literal CASOS_P1 de locustfile.py sin ejecutar importaciones externas."""
    arbol = ast.parse(ruta_locust.read_text(encoding="utf-8"), filename=str(ruta_locust))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign):
            for objetivo in nodo.targets:
                if isinstance(objetivo, ast.Name) and objetivo.id == "CASOS_P1":
                    valor = ast.literal_eval(nodo.value)
                    if isinstance(valor, (tuple, list)):
                        return tuple(valor)
    return ()


def verificar_scripts_y_locust(raiz: Path) -> VerificacionOperativa:
    """Confirma que existen Locust y scripts operativos, y que los casos de carga son válidos."""
    faltantes = [ruta for ruta in SCRIPTS_OPERACION if not (raiz / ruta).is_file()]
    if faltantes:
        return VerificacionOperativa(
            "locust_y_scripts_operacion",
            False,
            f"Archivos faltantes: {', '.join(faltantes)}",
        )

    casos = _extraer_casos_locust(raiz / "locustfile.py")
    if not casos:
        return VerificacionOperativa(
            "locust_y_scripts_operacion", False, "locustfile.py no define CASOS_P1"
        )
    for caso in casos:
        DemandaEntrada.model_validate(caso)

    return VerificacionOperativa(
        "locust_y_scripts_operacion",
        True,
        f"{len(SCRIPTS_OPERACION)} scripts/archivos presentes; "
        f"{len(casos)} casos de Locust validados con DemandaEntrada",
    )


def verificar_evidencias_carga_y_recursos(
    raiz: Path,
) -> tuple[list[VerificacionOperativa], dict[str, Any]]:
    """Valida presencia, formato e indicadores de las evidencias de Locust y docker stats."""
    resultados: list[VerificacionOperativa] = []

    faltantes = [ruta for ruta in ARCHIVOS_EVIDENCIA_CARGA if not (raiz / ruta).is_file()]
    png_invalidos = []
    for ruta_rel in (
        "reports/load_testing/locust_performance.png",
        "reports/load_testing/resource_usage.png",
    ):
        archivo = raiz / ruta_rel
        if archivo.is_file():
            with archivo.open("rb") as fh:
                if fh.read(8) != PNG_SIGNATURE:
                    png_invalidos.append(ruta_rel)

    archivos_ok = not faltantes and not png_invalidos
    detalle_archivos = (
        f"{len(ARCHIVOS_EVIDENCIA_CARGA)} evidencias preservadas (JSON, CSV y 2 PNG válidos)"
        if archivos_ok
        else f"faltantes={faltantes}, png_invalidos={png_invalidos}"
    )
    resultados.append(
        VerificacionOperativa("evidencias_carga_preservadas", archivos_ok, detalle_archivos)
    )
    if not archivos_ok:
        return resultados, {}

    resumen = _leer_json(raiz / "reports/load_testing/load_test_summary.json")
    with (raiz / "reports/load_testing/resources.csv").open(encoding="utf-8", newline="") as fh:
        filas_recursos = list(csv.DictReader(fh))

    contenedores_csv = {fila.get("container", "") for fila in filas_recursos}
    contenedores_json = set(resumen.get("recursos_por_contenedor", {}).keys())

    kpis_ok = (
        resumen.get("endpoint") == "POST /predict/p1"
        and resumen.get("solicitudes_totales", 0) > 0
        and resumen.get("solicitudes_exitosas") == resumen.get("solicitudes_totales")
        and resumen.get("tasa_error_pct") == 0.0
        and resumen.get("rps_promedio", 0) > 0
        and resumen.get("latencia_p50_ms", 0)
        <= resumen.get("latencia_p95_ms", 0)
        <= resumen.get("latencia_p99_ms", 0)
        and {"mibici-api", "mibici-mlflow"}.issubset(contenedores_csv)
        and {"mibici-api", "mibici-mlflow"}.issubset(contenedores_json)
    )
    detalle_kpis = (
        f"{resumen.get('solicitudes_totales')} solicitudes, "
        f"{resumen.get('tasa_error_pct')}% error, "
        f"{resumen.get('rps_promedio')} RPS, "
        f"p50={resumen.get('latencia_p50_ms')} ms, "
        f"p95={resumen.get('latencia_p95_ms')} ms, "
        f"p99={resumen.get('latencia_p99_ms')} ms; "
        f"{len(filas_recursos)} muestras en resources.csv"
        if kpis_ok
        else "Los KPIs de carga o las muestras de recursos no cumplen los criterios mínimos"
    )
    resultados.append(VerificacionOperativa("kpis_carga_y_recursos", kpis_ok, detalle_kpis))
    return resultados, {
        "muestras_csv": len(filas_recursos),
        "resumen_carga": resumen,
    }


def _consultar_http_local(
    url: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | str]:
    """Consulta un endpoint local (localhost/127.0.0.1) para la verificación en vivo."""
    if not url.startswith(("http://localhost:", "http://127.0.0.1:")):
        raise ValueError(f"Solo se permiten URLs locales en verificación en vivo: {url}")

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
            cuerpo = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(cuerpo)
            except json.JSONDecodeError:
                return resp.status, cuerpo
    except HTTPError as err:
        cuerpo = err.read().decode("utf-8", errors="replace")
        return err.code, cuerpo
    except URLError as err:
        return 0, str(err.reason)


def verificar_servicios_en_vivo(
    api_url: str = "http://localhost:8000",
    mlflow_url: str = "http://localhost:5001",
) -> tuple[list[VerificacionOperativa], dict[str, Any]]:
    """Valida los contenedores corriendo localmente (FastAPI, MLflow y docker stats)."""
    api_base = api_url.rstrip("/")
    mlflow_base = mlflow_url.rstrip("/")

    status_ready, body_ready = _consultar_http_local(f"{api_base}/health/ready")
    status_pred, body_pred = _consultar_http_local(f"{api_base}/predict/p1", CASO_DEMO_VALIDO)
    status_422, _ = _consultar_http_local(f"{api_base}/predict/p1", CASO_DEMO_INVALIDO)
    status_mlflow, _ = _consultar_http_local(f"{mlflow_base}/health")

    api_ok = (
        status_ready == 200
        and isinstance(body_ready, dict)
        and body_ready.get("model_status") == "loaded"
        and status_pred == 200
        and isinstance(body_pred, dict)
        and "prediction" in body_pred
        and status_422 == 422
    )
    mlflow_ok = status_mlflow == 200

    verificaciones = [
        VerificacionOperativa(
            "vivo_api_fastapi",
            api_ok,
            f"ready={status_ready}, predict={status_pred} "
            f"(pred={body_pred.get('prediction') if isinstance(body_pred, dict) else None}, "
            f"latency_ms={body_pred.get('latency_ms') if isinstance(body_pred, dict) else None}), "
            f"validacion_422={status_422}",
        ),
        VerificacionOperativa(
            "vivo_mlflow_health",
            mlflow_ok,
            f"GET {mlflow_base}/health -> HTTP {status_mlflow}",
        ),
    ]

    stats_lineas: list[str] = []
    try:
        salida = subprocess.check_output(  # nosec B603 B607
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}} | CPU={{.CPUPerc}} | MEM={{.MemUsage}} ({{.MemPerc}})",
                "mibici-api",
                "mibici-mlflow",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        stats_lineas = [linea for linea in salida.splitlines() if linea.strip()]
    except (OSError, subprocess.CalledProcessError):
        stats_lineas = []

    verificaciones.append(
        VerificacionOperativa(
            "vivo_docker_stats",
            len(stats_lineas) == 2,
            "; ".join(stats_lineas) if stats_lineas else "No se pudo leer docker stats",
        )
    )
    return verificaciones, {
        "api_url": api_base,
        "mlflow_url": mlflow_base,
        "docker_stats": stats_lineas,
    }


def construir_reporte_operacion(
    raiz: Path,
    *,
    verificar_vivo: bool = False,
    api_url: str = "http://localhost:8000",
    mlflow_url: str = "http://localhost:5001",
) -> dict[str, Any]:
    """Ejecuta las verificaciones de despliegue y operación y arma el JSON auditable."""
    verif_api, resumen_api = verificar_fastapi_contrato()
    verif_docker = verificar_dockerfile(raiz)
    verif_compose, resumen_compose = verificar_docker_compose(raiz)
    verif_locust = verificar_scripts_y_locust(raiz)
    verifs_carga, resumen_carga = verificar_evidencias_carga_y_recursos(raiz)

    verificaciones = [
        verif_api,
        verif_docker,
        verif_compose,
        verif_locust,
        *verifs_carga,
    ]
    datos_vivo: dict[str, Any] | None = None
    if verificar_vivo:
        verifs_vivo, datos_vivo = verificar_servicios_en_vivo(api_url, mlflow_url)
        verificaciones.extend(verifs_vivo)

    estado = "aprobado" if all(v.aprobado for v in verificaciones) else "inconsistente"
    reporte: dict[str, Any] = {
        "proyecto": "TC5061-Proyecto-Equipo9",
        "bloque": "despliegue_operacion_y_demo",
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "estado": estado,
        "fastapi": resumen_api,
        "docker_compose": resumen_compose,
        "evidencia_carga_y_recursos": resumen_carga,
        "comandos_demo": {
            "levantar_servicios": "docker compose up -d --build --wait",
            "smoke_tests": "make smoke-test",
            "demo_completa": "make demo-final",
            "demo_paso_a_paso": "./scripts/demo_final.sh --pausa",
            "apagar_servicios": "docker compose down",
        },
        "verificaciones": [asdict(v) for v in verificaciones],
    }
    if datos_vivo is not None:
        reporte["verificacion_en_vivo"] = datos_vivo
    return reporte


@app.command()
def main(
    salida: Annotated[
        Path,
        typer.Option(help="Ruta del reporte JSON relativa a la raíz del proyecto."),
    ] = Path("reports/final/operacion_evidencia.json"),
    verificar_vivo: Annotated[
        bool,
        typer.Option(
            "--verificar-vivo",
            help="Valida también la API (:8000), MLflow (:5001) y docker stats en vivo.",
        ),
    ] = False,
    api_url: Annotated[
        str,
        typer.Option(help="URL local de la API FastAPI para --verificar-vivo."),
    ] = "http://localhost:8000",
    mlflow_url: Annotated[
        str,
        typer.Option(help="URL local de MLflow para --verificar-vivo."),
    ] = "http://localhost:5001",
) -> None:
    """Valida el despliegue operativo y guarda la evidencia auditable en JSON."""
    reporte = construir_reporte_operacion(
        PROJ_ROOT,
        verificar_vivo=verificar_vivo,
        api_url=api_url,
        mlflow_url=mlflow_url,
    )
    ruta = salida if salida.is_absolute() else PROJ_ROOT / salida
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        json.dumps(
            {
                "estado": reporte["estado"],
                "verificaciones_aprobadas": sum(
                    1 for v in reporte["verificaciones"] if v["aprobado"]
                ),
                "verificaciones_totales": len(reporte["verificaciones"]),
            },
            ensure_ascii=False,
        )
    )
    typer.echo(f"Reporte guardado en {ruta}")
    if reporte["estado"] != "aprobado":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
