"""Resumen programático del reporte de drift, para consumo de otras etapas."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Annotated

from loguru import logger
import typer

from mibici.monitoring.drift import DIR_DRIFT

app = typer.Typer(add_completion=False)

PATRON_COLUMNA = re.compile(r"column=([^,)]+)")
PATRON_METODO = re.compile(r"method=(.+?),threshold=")
PATRON_UMBRAL = re.compile(r"threshold=([0-9.]+)")
PATRON_PROPORCION = re.compile(r"drift_share=([0-9.]+)")
ARCHIVOS = {"test": "data_drift.json", "validacion": "data_drift_validacion.json"}


def hay_drift(metodo: str, valor: float, umbral: float) -> bool:
    """Decide si una variable tiene drift según el tipo de prueba.

    Las pruebas de hipótesis (valor p) marcan drift cuando el valor queda por debajo del
    umbral; las distancias (Wasserstein, Jensen-Shannon), cuando lo alcanzan o superan.
    """
    nombre = metodo.lower()
    if "p_value" in nombre or "p-value" in nombre:
        return valor < umbral
    return valor >= umbral


def resumir(ruta_json: Path) -> dict:
    """Extrae del JSON de Evidently el drift por variable y el conteo global."""
    datos = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
    variables: dict[str, dict] = {}
    conteo = None
    proporcion_umbral = 0.5
    for metrica in datos["metrics"]:
        nombre = metrica["metric_name"]
        if nombre.startswith("DriftedColumnsCount"):
            conteo = metrica["value"]
            proporcion_umbral = float(PATRON_PROPORCION.search(nombre).group(1))
        elif nombre.startswith("ValueDrift"):
            valor = float(metrica["value"])
            umbral = float(PATRON_UMBRAL.search(nombre).group(1))
            metodo = PATRON_METODO.search(nombre).group(1)
            variables[PATRON_COLUMNA.search(nombre).group(1)] = {
                "metodo": metodo,
                "valor": round(valor, 4),
                "umbral": umbral,
                "drift": hay_drift(metodo, valor, umbral),
            }
    if conteo is None or not variables:
        raise ValueError("El JSON no contiene las métricas de drift esperadas")

    # Verificación cruzada con el conteo que calculó Evidently
    detectadas = sum(v["drift"] for v in variables.values())
    if detectadas != int(conteo["count"]):
        metodos = sorted({v["metodo"] for v in variables.values()})
        raise ValueError(
            f"El resumen detecta {detectadas} columnas con drift y Evidently reporta "
            f"{int(conteo['count'])}; revisa cómo se interpretan los métodos: {metodos}"
        )

    return {
        "total_columnas": len(variables),
        "columnas_con_drift": int(conteo["count"]),
        "proporcion_con_drift": round(float(conteo["share"]), 4),
        "umbral_proporcion": proporcion_umbral,
        "drift_dataset": float(conteo["share"]) >= proporcion_umbral,
        "variables_con_drift": sorted(c for c, v in variables.items() if v["drift"]),
        "variables": variables,
    }


@app.command()
def main(
    directorio: Annotated[
        Path, typer.Option(help="Carpeta con los JSON de Evidently.")
    ] = DIR_DRIFT,
) -> None:
    """Escribe drift_summary.json con el resumen de cada periodo disponible."""
    periodos = {}
    for actual, archivo in ARCHIVOS.items():
        ruta = directorio / archivo
        if not ruta.exists():
            continue
        resumen = resumir(ruta)
        chequeos = directorio / f"chequeos_datos_{actual}.json"
        if chequeos.exists():
            c = json.loads(chequeos.read_text(encoding="utf-8"))
            for llave in ("periodo_referencia", "periodo_actual", "version_datos", "max_filas"):
                resumen[llave] = c[llave]
        periodos[actual] = resumen
    if not periodos:
        raise FileNotFoundError(f"No hay reportes de drift en {directorio}")

    destino = directorio / "drift_summary.json"
    destino.write_text(json.dumps(periodos, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Resumen guardado en {destino}")


if __name__ == "__main__":
    app()
