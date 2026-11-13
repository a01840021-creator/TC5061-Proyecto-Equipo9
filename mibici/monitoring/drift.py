"""Reporte de data drift de P1 con Evidently.

Referencia: periodo con el que se ajustan los candidatos (abril a julio).
Actual: septiembre (prueba) o, como comparación intermedia, agosto (validación).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    PERIODO_TEST,
    PERIODO_TRAIN_CANDIDATOS,
    PERIODO_VALIDACION,
    REPORTS_DIR,
    SEMILLA,
)
from mibici.dataset import leer_version_datos
from mibici.features import cargar_datos_procesados, generar_dataset_p1

# Sin telemetría: debe fijarse antes de importar Evidently (verifica el nombre de la
# variable en la documentación de la versión fijada en requirements.txt)
os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

app = typer.Typer(add_completion=False)

DIR_DRIFT = REPORTS_DIR / "drift"
FECHA = "fecha_hora"
OBJETIVO = "demanda"
FEATURES = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
]
NUMERICAS = ["dpcapacity", "rezago_24h", "rezago_48h", "rezago_168h", OBJETIVO]
# hora y dia_semana son ciclos, no cantidades: se tratan como categorías
CATEGORICAS = ["estacion_id", "hora", "dia_semana", "es_fin_de_semana"]
COLUMNAS = FEATURES + [OBJETIVO]
PERIODOS_ACTUAL = {"test": PERIODO_TEST, "validacion": PERIODO_VALIDACION}


def cortar(df: pd.DataFrame, periodo: str, columna_fecha: str = FECHA) -> pd.DataFrame:
    """Filas dentro de un periodo `inicio/fin` de config.py, con el último día incluido."""
    inicio, fin = periodo.split("/")
    limite = pd.Timestamp(fin) + pd.Timedelta(days=1)
    fechas = pd.to_datetime(df[columna_fecha])
    return df.loc[(fechas >= pd.Timestamp(inicio)) & (fechas < limite)].copy()


def validar(df: pd.DataFrame, nombre: str, ref: pd.DataFrame | None = None) -> dict:
    """Revisa columnas, nulos, infinitos, duplicados y categorías nuevas de un conjunto."""
    faltan = [c for c in COLUMNAS + [FECHA] if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas: {faltan}")
    if df.empty:
        raise ValueError(f"[{nombre}] conjunto vacío")

    info = {
        "filas": len(df),
        "fecha_min": str(df[FECHA].min()),
        "fecha_max": str(df[FECHA].max()),
        "nulos": {c: int(n) for c, n in df[COLUMNAS].isna().sum().items()},
        "infinitos": int(np.isinf(df[NUMERICAS].to_numpy(dtype="float64")).sum()),
        # La clave del tablón es (hora, estación); las filas repetidas en las variables
        # son normales, las repetidas en la clave no
        "duplicados_clave": int(df.duplicated(subset=[FECHA, "estacion_id"]).sum()),
        "tipos": {c: str(t) for c, t in df[COLUMNAS].dtypes.items()},
    }
    if ref is not None:
        info["categorias_nuevas"] = {
            c: sorted(map(str, set(df[c].dropna()) - set(ref[c].dropna()))) for c in CATEGORICAS
        }
    return info


def muestrear(df: pd.DataFrame, max_filas: int) -> pd.DataFrame:
    """Muestra determinista para que el HTML pese poco; 0 conserva todas las filas."""
    if max_filas == 0 or len(df) <= max_filas:
        return df
    return df.sample(n=max_filas, random_state=SEMILLA)


def ejecutar(ref: pd.DataFrame, act: pd.DataFrame, salida: Path, prefijo: str = "data_drift"):
    """Genera el reporte de Evidently (HTML y JSON) y devuelve el resultado."""
    salida.mkdir(parents=True, exist_ok=True)
    definicion = DataDefinition(numerical_columns=NUMERICAS, categorical_columns=CATEGORICAS)
    ds_ref = Dataset.from_pandas(ref[COLUMNAS], data_definition=definicion)
    ds_act = Dataset.from_pandas(act[COLUMNAS], data_definition=definicion)

    resultado = Report([DataDriftPreset(), DataSummaryPreset()]).run(ds_act, ds_ref)
    resultado.save_html(str(salida / f"{prefijo}.html"))
    resultado.save_json(str(salida / f"{prefijo}.json"))
    return resultado


@app.command()
def main(
    actual: str = typer.Option("test", help="Periodo actual: 'test' (septiembre) o 'validacion'."),
    max_filas: int = typer.Option(50_000, min=0, help="Filas por conjunto; 0 = todas."),
    salida: Annotated[Path, typer.Option(help="Carpeta de salida.")] = DIR_DRIFT,
) -> None:
    """Compara la referencia de entrenamiento con un periodo posterior de P1."""
    if actual not in PERIODOS_ACTUAL:
        raise typer.BadParameter("Usa 'test' o 'validacion'.")

    viajes, estaciones = cargar_datos_procesados()
    datos = generar_dataset_p1(viajes, estaciones)

    ref = cortar(datos, PERIODO_TRAIN_CANDIDATOS)
    act = cortar(datos, PERIODOS_ACTUAL[actual])
    chequeos = {
        "periodo_referencia": PERIODO_TRAIN_CANDIDATOS,
        "periodo_actual": PERIODOS_ACTUAL[actual],
        "version_datos": leer_version_datos(),
        "semilla": SEMILLA,
        "max_filas": max_filas,
        "referencia": validar(ref, "referencia"),
        "actual": validar(act, "actual", ref=ref),
    }

    prefijo = "data_drift" if actual == "test" else f"data_drift_{actual}"
    logger.info(f"Referencia {len(ref):,} filas; actual {len(act):,} filas")
    ejecutar(muestrear(ref, max_filas), muestrear(act, max_filas), salida, prefijo)

    salida.mkdir(parents=True, exist_ok=True)
    (salida / f"chequeos_datos_{actual}.json").write_text(
        json.dumps(chequeos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"Reporte guardado en {salida}")


if __name__ == "__main__":
    app()
