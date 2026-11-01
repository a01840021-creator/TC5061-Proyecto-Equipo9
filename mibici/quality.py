"""Validación reproducible de integridad y drift con DeepChecks.

El módulo usa muestras deterministas para que los reportes puedan regenerarse sin
procesar millones de filas. Los datos completos permanecen fuera del repositorio y sólo
se guardan reportes HTML y un resumen JSON sin registros de usuarios.
"""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import sys
import types
from typing import Literal

import numpy as np
from packaging.version import parse as parse_version
import pandas as pd
import typer

from mibici.config import FECHA_CORTE, REPORTS_DIR, SEMILLA
from mibici.features import (
    cargar_datos_procesados,
    generar_dataset_p1,
    generar_dataset_p2,
    particionar_temporalmente,
)

app = typer.Typer(add_completion=False)

Pregunta = Literal["P1", "P2"]
SALIDA_DEFAULT = REPORTS_DIR / "deepchecks"
FEATURES_P1 = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
]
FEATURES_P2 = [
    "estacion_id",
    "dpcapacity",
    "salidas_lag1",
    "llegadas_lag1",
    "saldo_lag1",
    "saldo_media_3h",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
]


def componentes_deepchecks():
    """Importa DeepChecks con compatibilidad temporal para NumPy 2.x.

    DeepChecks 0.19.1 todavía consulta ``np.Inf`` y ``pkg_resources``, interfaces
    retiradas de NumPy y Setuptools recientes. El proyecto conserva versiones seguras y
    repone únicamente las interfaces que DeepChecks necesita. Este bloque puede retirarse
    cuando una versión posterior de DeepChecks corrija ambas incompatibilidades.
    """
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]

    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        compatibilidad = types.ModuleType("pkg_resources")
        compatibilidad.parse_version = parse_version
        compatibilidad.working_set = [
            types.SimpleNamespace(key=(distribucion.metadata.get("Name") or "").lower())
            for distribucion in metadata.distributions()
        ]
        sys.modules["pkg_resources"] = compatibilidad

    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    return Dataset, data_integrity, train_test_validation


def muestrear_deterministicamente(
    datos: pd.DataFrame,
    max_filas: int,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Devuelve como máximo ``max_filas`` con una muestra estable y sin modificar la entrada."""
    if max_filas < 1:
        raise ValueError("max_filas debe ser mayor o igual a 1")
    if len(datos) <= max_filas:
        return datos.copy().reset_index(drop=True)
    return datos.sample(n=max_filas, random_state=semilla).reset_index(drop=True)


def construir_particiones(pregunta: Pregunta) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Construye las particiones temporales y devuelve variables y etiqueta de P1 o P2."""
    viajes, estaciones = cargar_datos_procesados()
    if pregunta == "P1":
        datos = generar_dataset_p1(viajes, estaciones)
        variables, etiqueta = FEATURES_P1, "demanda"
    elif pregunta == "P2":
        datos = generar_dataset_p2(viajes, estaciones)
        variables, etiqueta = FEATURES_P2, "riesgo"
    else:
        raise ValueError(f"Pregunta no soportada: {pregunta}")

    train, test = particionar_temporalmente(datos, fecha_corte=FECHA_CORTE)
    columnas = ["fecha_hora", *variables, etiqueta]
    return train[columnas], test[columnas], variables, etiqueta


def ejecutar_validacion(
    pregunta: Pregunta,
    max_filas: int = 5_000,
    directorio_salida: Path = SALIDA_DEFAULT,
) -> dict[str, object]:
    """Ejecuta integridad y validación train-test y guarda evidencia reproducible.

    El drift se documenta como señal de monitoreo y no bloquea la ejecución: una división
    temporal puede cambiar legítimamente entre meses. Los errores de esquema, nulos o
    duplicados quedan visibles en los reportes para que el equipo los investigue.
    """
    Dataset, data_integrity, train_test_validation = componentes_deepchecks()
    train, test, variables, etiqueta = construir_particiones(pregunta)
    train = muestrear_deterministicamente(train, max_filas)
    test = muestrear_deterministicamente(test, max_filas)

    categoricas = [
        columna
        for columna in ["estacion_id", "hora", "dia_semana", "es_fin_de_semana"]
        if columna in variables
    ]
    train_ds = Dataset(
        train,
        label=etiqueta,
        features=variables,
        cat_features=categoricas,
        datetime_name="fecha_hora",
    )
    test_ds = Dataset(
        test,
        label=etiqueta,
        features=variables,
        cat_features=categoricas,
        datetime_name="fecha_hora",
    )

    resultados = {
        "integridad_train": data_integrity().run(train_ds),
        "integridad_test": data_integrity().run(test_ds),
        "validacion_train_test": train_test_validation().run(train_ds, test_ds),
    }

    salida_pregunta = Path(directorio_salida) / pregunta.lower()
    salida_pregunta.mkdir(parents=True, exist_ok=True)
    resumen: dict[str, object] = {
        "pregunta": pregunta,
        "semilla_muestreo": SEMILLA,
        "max_filas_por_particion": max_filas,
        "filas_train_evaluadas": len(train),
        "filas_test_evaluadas": len(test),
        "reportes": {},
    }
    for nombre, resultado in resultados.items():
        ruta = salida_pregunta / f"{nombre}.html"
        resultado.save_as_html(str(ruta), as_widget=False)
        no_aprobados = resultado.get_not_passed_checks()
        resumen["reportes"][nombre] = {
            "archivo": str(ruta.relative_to(REPORTS_DIR)),
            "condiciones_aprobadas": bool(resultado.passed(fail_if_warning=False)),
            "chequeos_no_aprobados": [item.check.name() for item in no_aprobados],
        }

    ruta_resumen = salida_pregunta / "resumen.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


@app.command()
def main(
    pregunta: str = typer.Option("all", help="P1, P2 o all."),
    max_filas: int = typer.Option(5_000, min=1, help="Muestra máxima por partición."),
    salida: Path = typer.Option(  # noqa: B008
        SALIDA_DEFAULT, help="Directorio para HTML y JSON."
    ),
) -> None:
    """Genera reportes de calidad sin versionar ni exponer los datos fuente."""
    seleccion = pregunta.upper()
    preguntas = ["P1", "P2"] if seleccion == "ALL" else [seleccion]
    if any(item not in {"P1", "P2"} for item in preguntas):
        raise typer.BadParameter("pregunta debe ser P1, P2 o all")

    for item in preguntas:
        resumen = ejecutar_validacion(item, max_filas=max_filas, directorio_salida=salida)
        typer.echo(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
