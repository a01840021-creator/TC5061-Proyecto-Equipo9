"""Auditoría reproducible de los componentes del proyecto final de MiBici.

La auditoría es deliberadamente ligera: no descarga datos, no entrena modelos y no
necesita un servidor de MLflow. Verifica contratos del repositorio y produce una
evidencia JSON que puede ejecutarse desde un clon limpio antes de la entrega.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess  # nosec B404
from typing import Annotated, Any

import typer
import yaml

from mibici.config import PROJ_ROOT

app = typer.Typer(add_completion=False)

VERSION_PYTHON_ESPERADA = "3.10.20"

COMPONENTES_REQUERIDOS: dict[str, tuple[str, ...]] = {
    "estructura_y_experimentacion": (
        "README.md",
        "MLproject",
        "python_env.yaml",
        "requirements.txt",
        "mibici/dataset.py",
        "mibici/features.py",
        "mibici/modeling/candidates.py",
    ),
    "modelo_y_registry": (
        "mibici/modeling/registry.py",
        "models/p1/MLmodel",
        "models/p1/metadata.json",
        "models/p1/registered_model_meta",
    ),
    "despliegue_y_operacion": (
        "app/main.py",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "locustfile.py",
        "reports/load_testing/load_test_summary.json",
    ),
    "monitoreo_y_finops": (
        "mibici/monitoring/drift.py",
        "mibici/monitoring/ab_test.py",
        "reports/drift/drift_summary.json",
        "reports/drift/ab_test_results.json",
        "references/finops.md",
    ),
}

ENTREGABLES_FINALES: dict[str, tuple[str, ...]] = {
    "model_card": ("MODEL_CARD.md",),
    "presentacion": ("reports/final/TC5061_Proyecto_Final_Equipo09.pdf",),
    "video_o_enlace": (
        "reports/final/TC5061_Proyecto_Final_Equipo09.mp4",
        "reports/final/enlace_video.txt",
    ),
}

ENTRY_POINTS_ESPERADOS = {"ingest", "baseline", "train", "search", "register"}


@dataclass(frozen=True)
class ResultadoChequeo:
    """Resultado serializable de un chequeo de auditoría."""

    nombre: str
    aprobado: bool
    detalle: str
    severidad: str = "error"


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    contenido = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return contenido if isinstance(contenido, dict) else {}


def _version_python_docker(dockerfile: str) -> str | None:
    coincidencia = re.search(r"^FROM\s+python:([0-9.]+)-", dockerfile, flags=re.MULTILINE)
    return coincidencia.group(1) if coincidencia else None


def _version_python_ci(workflow: dict[str, Any]) -> str | None:
    jobs = workflow.get("jobs", {})
    for trabajo in jobs.values():
        for paso in trabajo.get("steps", []):
            if str(paso.get("uses", "")).startswith("actions/setup-python"):
                valor = paso.get("with", {}).get("python-version")
                return str(valor) if valor is not None else None
    return None


def verificar_componentes(raiz: Path) -> list[ResultadoChequeo]:
    """Comprueba que cada bloque de la rúbrica tenga sus archivos de implementación."""
    resultados: list[ResultadoChequeo] = []
    for bloque, rutas in COMPONENTES_REQUERIDOS.items():
        faltantes = [ruta for ruta in rutas if not (raiz / ruta).is_file()]
        resultados.append(
            ResultadoChequeo(
                nombre=f"componentes_{bloque}",
                aprobado=not faltantes,
                detalle=(
                    f"{len(rutas)} archivos requeridos presentes"
                    if not faltantes
                    else f"Faltan: {', '.join(faltantes)}"
                ),
            )
        )
    return resultados


def verificar_version_python(raiz: Path) -> ResultadoChequeo:
    """Valida que MLflow, Docker y CI utilicen la misma versión de Python."""
    entorno = _leer_yaml(raiz / "python_env.yaml")
    dockerfile = (raiz / "Dockerfile").read_text(encoding="utf-8")
    workflow = _leer_yaml(raiz / ".github/workflows/ci.yml")
    versiones = {
        "python_env.yaml": str(entorno.get("python")),
        "Dockerfile": _version_python_docker(dockerfile),
        "GitHub Actions": _version_python_ci(workflow),
    }
    aprobado = set(versiones.values()) == {VERSION_PYTHON_ESPERADA}
    detalle = ", ".join(f"{origen}={version}" for origen, version in versiones.items())
    return ResultadoChequeo("version_python_alineada", aprobado, detalle)


def verificar_dependencias_fijadas(raiz: Path) -> ResultadoChequeo:
    """Comprueba que las dependencias reproducibles usen versiones exactas."""
    archivos = ("requirements.txt", "requirements-serving.txt")
    sin_version: list[str] = []
    for archivo in archivos:
        for numero, linea in enumerate(
            (raiz / archivo).read_text(encoding="utf-8").splitlines(), start=1
        ):
            contenido = linea.strip()
            if not contenido or contenido.startswith(("#", "-e ", "--")):
                continue
            if "==" not in contenido:
                sin_version.append(f"{archivo}:{numero}:{contenido}")
    return ResultadoChequeo(
        "dependencias_con_version_exacta",
        not sin_version,
        "Todas las dependencias usan =="
        if not sin_version
        else f"Sin versión exacta: {', '.join(sin_version)}",
    )


def verificar_mlproject(raiz: Path) -> ResultadoChequeo:
    """Valida los puntos de entrada que reproducen el ciclo experimental."""
    proyecto = _leer_yaml(raiz / "MLproject")
    encontrados = set(proyecto.get("entry_points", {}))
    faltantes = sorted(ENTRY_POINTS_ESPERADOS - encontrados)
    return ResultadoChequeo(
        "entry_points_mlflow",
        not faltantes,
        f"Disponibles: {', '.join(sorted(encontrados))}"
        if not faltantes
        else f"Faltan: {', '.join(faltantes)}",
    )


def verificar_linaje_champion(raiz: Path) -> ResultadoChequeo:
    """Comprueba que el champion empaquetado conserve identidad y corrida de origen."""
    ruta = raiz / "models/p1/metadata.json"
    if not ruta.is_file():
        return ResultadoChequeo("linaje_champion", False, f"No existe {ruta}")
    metadatos = json.loads(ruta.read_text(encoding="utf-8"))
    campos = ("model_name", "alias", "version", "run_id", "periodo_train")
    faltantes = [campo for campo in campos if not metadatos.get(campo)]
    alias_correcto = metadatos.get("alias") == "champion"
    aprobado = not faltantes and alias_correcto
    detalle = (
        f"{metadatos.get('model_name')}@{metadatos.get('alias')}, "
        f"versión {metadatos.get('version')}, run {metadatos.get('run_id')}"
    )
    if faltantes:
        detalle = f"Campos faltantes: {', '.join(faltantes)}"
    elif not alias_correcto:
        detalle = f"Alias esperado champion; recibido {metadatos.get('alias')}"
    return ResultadoChequeo("linaje_champion", aprobado, detalle)


def verificar_entregables_finales(raiz: Path) -> list[ResultadoChequeo]:
    """Reporta pendientes finales sin bloquear el trabajo técnico previo."""
    resultados: list[ResultadoChequeo] = []
    for nombre, alternativas in ENTREGABLES_FINALES.items():
        presentes = [ruta for ruta in alternativas if (raiz / ruta).is_file()]
        resultados.append(
            ResultadoChequeo(
                nombre=f"entregable_{nombre}",
                aprobado=bool(presentes),
                detalle=(
                    f"Presente: {presentes[0]}"
                    if presentes
                    else f"Pendiente; se acepta: {' o '.join(alternativas)}"
                ),
                severidad="pendiente",
            )
        )
    return resultados


def obtener_commit(raiz: Path) -> str:
    """Devuelve el commit auditado o ``desconocido`` fuera de un clon Git."""
    try:
        return subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=raiz,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def ejecutar_auditoria(raiz: Path) -> dict[str, Any]:
    """Ejecuta todos los chequeos y construye el reporte final."""
    chequeos = [
        *verificar_componentes(raiz),
        verificar_version_python(raiz),
        verificar_dependencias_fijadas(raiz),
        verificar_mlproject(raiz),
        verificar_linaje_champion(raiz),
        *verificar_entregables_finales(raiz),
    ]
    errores = [
        resultado
        for resultado in chequeos
        if not resultado.aprobado and resultado.severidad == "error"
    ]
    pendientes = [
        resultado
        for resultado in chequeos
        if not resultado.aprobado and resultado.severidad == "pendiente"
    ]
    return {
        "proyecto": "TC5061-Proyecto-Equipo9",
        "commit": obtener_commit(raiz),
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "resumen": {
            "chequeos_totales": len(chequeos),
            "errores_tecnicos": len(errores),
            "pendientes_entrega": len(pendientes),
            "base_tecnica_reproducible": not errores,
            "entrega_completa": not errores and not pendientes,
        },
        "chequeos": [asdict(resultado) for resultado in chequeos],
    }


@app.command()
def main(
    salida: Annotated[
        Path,
        typer.Option(help="Ruta del reporte JSON relativo a la raíz del proyecto."),
    ] = Path("reports/final/auditoria_reproducibilidad.json"),
    estricto: Annotated[
        bool,
        typer.Option(help="Termina con error si faltan entregables finales."),
    ] = False,
) -> None:
    """Audita el repositorio y guarda una evidencia JSON."""
    reporte = ejecutar_auditoria(PROJ_ROOT)
    ruta_salida = salida if salida.is_absolute() else PROJ_ROOT / salida
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(reporte["resumen"], ensure_ascii=False, indent=2))
    typer.echo(f"Reporte guardado en {ruta_salida}")

    resumen = reporte["resumen"]
    if not resumen["base_tecnica_reproducible"]:
        raise typer.Exit(code=1)
    if estricto and not resumen["entrega_completa"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
