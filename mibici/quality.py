"""Validación reproducible de integridad y drift con DeepChecks.

El módulo usa muestras deterministas para que los reportes puedan regenerarse sin
procesar millones de filas. Los datos completos permanecen fuera del repositorio y sólo
se guardan reportes HTML y un resumen JSON sin registros de usuarios.

Cubre cuatro tablas:

- P1 y P2: tablones de entrenamiento y prueba, con ``data_integrity`` y
  ``train_test_validation``.
- P3: perfiles de estaciones que agrupan los modelos de P3, con ``data_integrity``.
- Dataset base: viajes y nomenclatura crudos (``data/raw/``) y procesados
  (``data/processed/``), con ``data_integrity`` y conteos sobre todos los datos.

Los conteos de duplicados exactos, nulos, tipos y estaciones fuera de la nomenclatura se
calculan con pandas sobre todas las filas, porque una muestra no contiene casi ningún
par de duplicados y DeepChecks no tiene un chequeo para la relación entre dos tablas.
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

from mibici.config import (
    FECHA_CORTE,
    MIN_SALIDAS_ESTACION_ACTIVA,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RUTA_ESTACIONES,
    RUTA_VIAJES,
    SEMILLA,
    URL_NOMENCLATURA,
    URLS_VIAJES,
)
from mibici.dataset import (
    COLUMNAS_ESTACIONES,
    COLUMNAS_FECHA,
    COLUMNAS_ID_VIAJES,
    COLUMNAS_PERSONALES,
    COLUMNAS_VIAJES,
    FORMATO_FECHA,
    leer_csv,
    validar_columnas,
)
from mibici.features import (
    cargar_datos_procesados,
    generar_dataset_p1,
    generar_dataset_p2,
    generar_dataset_p3,
    particionar_temporalmente,
)
from mibici.modeling.baselines import columnas_p3

app = typer.Typer(add_completion=False)

Pregunta = Literal["P1", "P2"]
PREGUNTAS = ("P1", "P2", "P3", "BASE")
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
# Columnas de viajes que no son datos personales: las únicas que llegan a DeepChecks
COLUMNAS_VIAJES_SIN_PERSONALES = [c for c in COLUMNAS_VIAJES if c not in COLUMNAS_PERSONALES]
COLUMNAS_ID_ESTACIONES = ["id", "dpcapacity"]
COLUMNAS_COORDENADAS = ["latitude", "longitude"]


def componentes_deepchecks():
    """Importa DeepChecks con compatibilidad temporal para NumPy 2.x.

    DeepChecks 0.19.1 todavía consulta ``np.Inf``, ``np.NaN`` y ``pkg_resources``,
    interfaces retiradas de NumPy y Setuptools recientes. Sin ``np.NaN``, el chequeo
    ``Feature Feature Correlation`` de ``data_integrity`` no se ejecuta. El proyecto
    conserva versiones seguras y repone únicamente las interfaces que DeepChecks
    necesita. Este bloque puede retirarse cuando una versión posterior de DeepChecks
    corrija las incompatibilidades.
    """
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]

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


def suite_integridad_con_nulos():
    """``data_integrity`` más ``Percent Of Nulls`` sin tolerancia.

    La suite por omisión sólo revisa si una columna mezcla tipos de nulo (``Mixed
    Nulls``), no cuántos nulos tiene. Para P3 y el dataset base cualquier nulo es un
    hallazgo, así que se agrega el chequeo con un umbral de 0 %.
    """
    _, data_integrity, _ = componentes_deepchecks()
    from deepchecks.tabular.checks import PercentOfNulls

    suite = data_integrity()
    suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0))
    return suite


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


def resumir_resultado(resultado, ruta_html: Path, directorio_salida: Path) -> dict[str, object]:
    """Guarda el HTML de una suite y resume sus chequeos sin copiar valores de los datos.

    Lista los nombres de los chequeos aprobados, los no aprobados (fallos y advertencias),
    los que no se ejecutaron y, por cada condición no aprobada, su nombre, su categoría y
    el texto con el que DeepChecks la explica.
    """
    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    resultado.save_as_html(str(ruta_html), as_widget=False)
    no_aprobados = resultado.get_not_passed_checks()
    return {
        "archivo": ruta_html.relative_to(directorio_salida).as_posix(),
        "condiciones_aprobadas": bool(resultado.passed(fail_if_warning=False)),
        "chequeos_aprobados": [item.check.name() for item in resultado.get_passed_checks()],
        "chequeos_no_aprobados": [item.check.name() for item in no_aprobados],
        "condiciones_no_aprobadas": [
            {
                "chequeo": item.check.name(),
                "condicion": condicion.name,
                "categoria": condicion.category.value,
                "detalle": condicion.details,
            }
            for item in no_aprobados
            for condicion in item.conditions_results
            if not condicion.is_pass()
        ],
        "chequeos_no_ejecutados": {
            item.check.name(): str(item.exception) for item in resultado.get_not_ran_checks()
        },
    }


def escribir_resumen(resumen: dict[str, object], directorio: Path) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    (directorio / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# P1 y P2 -----------------------------------------------------------------------------


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

    directorio_salida = Path(directorio_salida)
    salida_pregunta = directorio_salida / pregunta.lower()
    resumen: dict[str, object] = {
        "pregunta": pregunta,
        "semilla_muestreo": SEMILLA,
        "max_filas_por_particion": max_filas,
        "filas_train_evaluadas": len(train),
        "filas_test_evaluadas": len(test),
        "reportes": {
            nombre: resumir_resultado(
                resultado, salida_pregunta / f"{nombre}.html", directorio_salida
            )
            for nombre, resultado in resultados.items()
        },
    }
    escribir_resumen(resumen, salida_pregunta)
    return resumen


# P3 ----------------------------------------------------------------------------------


def diagnosticar_perfiles_p3(perfiles: pd.DataFrame, estaciones: pd.DataFrame) -> dict[str, int]:
    """Cuenta los problemas de integridad de la tabla de perfiles de P3.

    - ``ids_repetidos``: estaciones con más de un perfil.
    - ``nulos``: celdas vacías en las variables del agrupamiento.
    - ``proporciones_fuera_de_rango``: valores de las variables fuera de [0, 1]; todas son
      proporciones.
    - ``perfiles_repetidos``: filas cuyas variables del agrupamiento son idénticas a las de
      una fila anterior. Las estaciones sin salidas antes del corte comparten el perfil de
      ceros, así que se reportan también aparte en ``estaciones_sin_salidas`` y
      ``perfiles_repetidos_con_salidas`` cuenta sólo los repetidos entre las demás.
    - ``estaciones_fuera_de_nomenclatura``: perfiles cuyo ``id`` no está en la nomenclatura.
    - ``estaciones_sin_perfil``: estaciones de la nomenclatura sin perfil.
    """
    variables = columnas_p3(perfiles)
    valores = perfiles[variables]
    con_salidas = perfiles["n_salidas"] >= MIN_SALIDAS_ESTACION_ACTIVA
    conocidas = set(estaciones["id"].dropna())
    con_perfil = set(perfiles["id"].dropna())
    return {
        "filas": len(perfiles),
        "ids_repetidos": int(perfiles["id"].duplicated().sum()),
        "nulos": int(valores.isna().sum().sum()),
        "proporciones_fuera_de_rango": int(((valores < 0) | (valores > 1)).sum().sum()),
        "perfiles_repetidos": int(valores.duplicated().sum()),
        "estaciones_sin_salidas": int((~con_salidas).sum()),
        "perfiles_repetidos_con_salidas": int(valores[con_salidas].duplicated().sum()),
        "estaciones_fuera_de_nomenclatura": len(con_perfil - conocidas),
        "estaciones_sin_perfil": len(conocidas - con_perfil),
    }


def validar_perfiles_p3(
    perfiles: pd.DataFrame,
    estaciones: pd.DataFrame,
    max_filas: int = 5_000,
    directorio_salida: Path = SALIDA_DEFAULT,
) -> dict[str, object]:
    """Valida con ``data_integrity`` las variables con que los modelos de P3 agrupan.

    DeepChecks recibe el ``id`` como índice y las 26 variables de ``columnas_p3``; nombre,
    ubicación y capacidad no entran al agrupamiento y no se evalúan aquí.
    """
    Dataset, _, _ = componentes_deepchecks()
    variables = columnas_p3(perfiles)
    muestra = muestrear_deterministicamente(perfiles[["id", *variables]], max_filas)
    dataset = Dataset(muestra, index_name="id", features=variables, cat_features=[])
    resultado = suite_integridad_con_nulos().run(dataset)

    directorio_salida = Path(directorio_salida)
    salida = directorio_salida / "p3"
    resumen: dict[str, object] = {
        "pregunta": "P3",
        "semilla_muestreo": SEMILLA,
        "max_filas": max_filas,
        "filas_evaluadas": len(muestra),
        "diagnostico": diagnosticar_perfiles_p3(perfiles, estaciones),
        "reportes": {
            "integridad_perfiles": resumir_resultado(
                resultado, salida / "integridad_perfiles.html", directorio_salida
            )
        },
    }
    escribir_resumen(resumen, salida)
    return resumen


def ejecutar_validacion_p3(
    max_filas: int = 5_000, directorio_salida: Path = SALIDA_DEFAULT
) -> dict[str, object]:
    """Construye los perfiles de P3 como lo hacen sus modelos y los valida."""
    viajes, estaciones = cargar_datos_procesados()
    perfiles = generar_dataset_p3(viajes, estaciones, solo_train=True)
    return validar_perfiles_p3(perfiles, estaciones, max_filas, directorio_salida)


# Dataset base ------------------------------------------------------------------------


def contar_no_convertibles(crudo: pd.DataFrame) -> dict[str, int]:
    """Cuenta los valores no nulos que la ingesta no podría tipar, por columna.

    Para los identificadores, valores que no son enteros; para las fechas, valores con un
    formato distinto de ``FORMATO_FECHA``. Los nulos se cuentan aparte.
    """
    conteo = {}
    for columna in COLUMNAS_ID_VIAJES:
        texto = crudo[columna].dropna().astype(str)
        conteo[columna] = int((~texto.str.fullmatch(r"\s*[+-]?\d+\s*")).sum())
    for columna in COLUMNAS_FECHA:
        texto = crudo[columna].dropna()
        fechas = pd.to_datetime(texto, format=FORMATO_FECHA, errors="coerce")
        conteo[columna] = int(fechas.isna().sum())
    return conteo


def tipar_viajes_crudos(crudo: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tipos de la ingesta sin fallar: lo que no se puede convertir queda nulo."""
    viajes = crudo[COLUMNAS_VIAJES].copy()
    for columna in COLUMNAS_FECHA:
        viajes[columna] = pd.to_datetime(viajes[columna], format=FORMATO_FECHA, errors="coerce")
    for columna in COLUMNAS_ID_VIAJES:
        viajes[columna] = pd.to_numeric(viajes[columna], errors="coerce")
    return viajes


def tipar_estaciones_crudas(crudo: pd.DataFrame) -> pd.DataFrame:
    estaciones = crudo[COLUMNAS_ESTACIONES].copy()
    for columna in COLUMNAS_ID_ESTACIONES + COLUMNAS_COORDENADAS:
        estaciones[columna] = pd.to_numeric(estaciones[columna], errors="coerce")
    return estaciones


def verificar_estaciones_de_viajes(
    viajes: pd.DataFrame, estaciones: pd.DataFrame
) -> dict[str, object]:
    """Relación entre viajes y nomenclatura, con la regla de ``quitar_sin_estacion``.

    DeepChecks valida una tabla a la vez y no tiene un chequeo de integridad referencial,
    así que la verificación se hace aquí: un viaje queda fuera si su origen o su destino
    no es un ``id`` de la nomenclatura. Los identificadores nulos no cuentan como fuera,
    porque ya se reportan como nulos. ``estaciones_ausentes`` da, por cada estación
    ausente, cuántos viajes la involucran.
    """
    conocidas = set(estaciones["id"].dropna())
    origen_fuera = viajes["origen_id"].notna() & ~viajes["origen_id"].isin(conocidas)
    destino_fuera = viajes["destino_id"].notna() & ~viajes["destino_id"].isin(conocidas)
    fuera = origen_fuera | destino_fuera
    ausentes = sorted(
        set(viajes.loc[origen_fuera, "origen_id"]) | set(viajes.loc[destino_fuera, "destino_id"])
    )
    descartados = viajes[fuera]
    return {
        "viajes_fuera_de_nomenclatura": int(fuera.sum()),
        "viajes_con_origen_fuera": int(origen_fuera.sum()),
        "viajes_con_destino_fuera": int(destino_fuera.sum()),
        "estaciones_ausentes": {
            str(int(i)): int(
                ((descartados["origen_id"] == i) | (descartados["destino_id"] == i)).sum()
            )
            for i in ausentes
        },
    }


def contar_nulos(tabla: pd.DataFrame) -> dict[str, object]:
    """Nulos por columna en las columnas no personales y un total para las personales.

    De las columnas personales sólo se reporta cuántas celdas vacías tienen, sin ningún
    valor; la clave aparece únicamente si la tabla las contiene.
    """
    personales = [c for c in tabla.columns if c in COLUMNAS_PERSONALES]
    conteo: dict[str, object] = {
        "por_columna": {
            c: int(tabla[c].isna().sum()) for c in tabla.columns if c not in COLUMNAS_PERSONALES
        }
    }
    if personales:
        conteo["en_columnas_personales"] = int(tabla[personales].isna().sum().sum())
    return conteo


def diagnosticar_viajes(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> dict[str, object]:
    """Duplicados y relación con la nomenclatura de una tabla de viajes, en todas sus filas.

    Un duplicado exacto es una fila idéntica en todas las columnas de ``COLUMNAS_VIAJES``
    presentes: las ocho del CSV en los crudos, igual que en la ingesta, y las cinco que
    quedan en los procesados.
    """
    regla = [c for c in COLUMNAS_VIAJES if c in viajes.columns]
    return {
        "filas": len(viajes),
        "columnas_regla_duplicados": regla,
        "duplicados_exactos": int(viajes.duplicated(subset=regla).sum()),
        "viaje_id_repetidos": int(viajes["viaje_id"].duplicated().sum()),
        "estaciones": verificar_estaciones_de_viajes(viajes, estaciones),
    }


def diagnosticar_estaciones(estaciones: pd.DataFrame) -> dict[str, object]:
    """Conteos de integridad de la nomenclatura sobre todas sus filas."""
    return {
        "filas": len(estaciones),
        "duplicados_exactos": int(estaciones.duplicated().sum()),
        "ids_repetidos": int(estaciones["id"].duplicated().sum()),
        "capacidad_cero": int((estaciones["dpcapacity"] == 0).sum()),
    }


def leer_viajes_crudos(rutas: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Lee los CSV de viajes como texto, cuenta nulos y valores no convertibles y tipa.

    Devuelve los viajes con las ocho columnas, porque la regla de duplicados exactos de la
    ingesta las usa todas, y los conteos sobre el texto original de todos los archivos:
    los nulos se cuentan antes de tipar para no confundirlos con los no convertibles.
    """
    tablas = []
    nulos = dict.fromkeys(COLUMNAS_VIAJES_SIN_PERSONALES, 0)
    nulos_personales = 0
    no_convertibles = dict.fromkeys(COLUMNAS_ID_VIAJES + COLUMNAS_FECHA, 0)
    for ruta in rutas:
        crudo = leer_csv(ruta)
        validar_columnas(crudo, COLUMNAS_VIAJES, ruta.name)
        crudo = crudo[COLUMNAS_VIAJES]
        conteo = contar_nulos(crudo)
        for columna, n in conteo["por_columna"].items():
            nulos[columna] += n
        nulos_personales += conteo["en_columnas_personales"]
        for columna, n in contar_no_convertibles(crudo).items():
            no_convertibles[columna] += n
        tablas.append(tipar_viajes_crudos(crudo))
    conteos = {
        "nulos": {"por_columna": nulos, "en_columnas_personales": nulos_personales},
        "valores_no_convertibles": no_convertibles,
    }
    return pd.concat(tablas, ignore_index=True), conteos


def dataset_viajes(Dataset, viajes: pd.DataFrame):
    """Dataset de DeepChecks para viajes, sin columnas personales."""
    return Dataset(
        viajes[COLUMNAS_VIAJES_SIN_PERSONALES],
        index_name="viaje_id",
        datetime_name="inicio_del_viaje",
        features=["fin_del_viaje", "origen_id", "destino_id"],
        cat_features=["origen_id", "destino_id"],
    )


def dataset_estaciones(Dataset, estaciones: pd.DataFrame):
    return Dataset(
        estaciones[COLUMNAS_ESTACIONES],
        index_name="id",
        features=["name", "latitude", "longitude", "dpcapacity"],
        cat_features=["name"],
    )


def validar_dataset_base(
    rutas_viajes_crudos: list[Path],
    ruta_nomenclatura: Path,
    ruta_viajes_procesados: Path,
    ruta_estaciones_procesadas: Path,
    max_filas: int = 5_000,
    directorio_salida: Path = SALIDA_DEFAULT,
) -> dict[str, object]:
    """Valida los datos crudos y los procesados para documentar lo que corrige la ingesta.

    En los crudos, los duplicados exactos se cuentan con las ocho columnas y después se
    quitan ``usuario_id``, ``genero`` y ``anio_de_nacimiento``; DeepChecks nunca las
    recibe. Los viajes se validan sobre una muestra determinista de ``max_filas`` y las
    estaciones completas. Todos los conteos de ``diagnostico`` usan todas las filas.
    """
    Dataset, _, _ = componentes_deepchecks()
    directorio_salida = Path(directorio_salida)
    salida = directorio_salida / "base"

    viajes_crudos, conteos_texto = leer_viajes_crudos(rutas_viajes_crudos)
    nomenclatura_cruda = leer_csv(ruta_nomenclatura)
    validar_columnas(nomenclatura_cruda, COLUMNAS_ESTACIONES, ruta_nomenclatura.name)
    nomenclatura_cruda = nomenclatura_cruda[COLUMNAS_ESTACIONES]
    nomenclatura = tipar_estaciones_crudas(nomenclatura_cruda)
    diagnostico_crudos = {
        "viajes": {**diagnosticar_viajes(viajes_crudos, nomenclatura), **conteos_texto},
        "estaciones": {
            **diagnosticar_estaciones(nomenclatura),
            "nulos": contar_nulos(nomenclatura_cruda),
            "valores_no_convertibles": {
                c: int((nomenclatura[c].isna() & nomenclatura_cruda[c].notna()).sum())
                for c in COLUMNAS_ID_ESTACIONES + COLUMNAS_COORDENADAS
            },
        },
    }
    # Los datos personales sólo se usaron para contar duplicados exactos
    viajes_crudos = viajes_crudos.drop(columns=COLUMNAS_PERSONALES)
    muestra_crudos = muestrear_deterministicamente(viajes_crudos, max_filas)
    del viajes_crudos

    viajes_procesados = pd.read_parquet(ruta_viajes_procesados)
    estaciones_procesadas = pd.read_parquet(ruta_estaciones_procesadas)
    diagnostico_procesados = {
        "viajes": {
            **diagnosticar_viajes(viajes_procesados, estaciones_procesadas),
            "nulos": contar_nulos(viajes_procesados),
            "tipos": {c: str(t) for c, t in viajes_procesados.dtypes.items()},
        },
        "estaciones": {
            **diagnosticar_estaciones(estaciones_procesadas),
            "nulos": contar_nulos(estaciones_procesadas),
            "tipos": {c: str(t) for c, t in estaciones_procesadas.dtypes.items()},
        },
    }
    muestra_procesados = muestrear_deterministicamente(viajes_procesados, max_filas)
    del viajes_procesados

    suites = {
        "crudos_viajes": dataset_viajes(Dataset, muestra_crudos),
        "crudos_estaciones": dataset_estaciones(Dataset, nomenclatura),
        "procesados_viajes": dataset_viajes(Dataset, muestra_procesados),
        "procesados_estaciones": dataset_estaciones(Dataset, estaciones_procesadas),
    }
    reportes = {
        nombre: resumir_resultado(
            suite_integridad_con_nulos().run(dataset),
            salida / f"{nombre}.html",
            directorio_salida,
        )
        for nombre, dataset in suites.items()
    }

    resumen: dict[str, object] = {
        "pregunta": "BASE",
        "semilla_muestreo": SEMILLA,
        "max_filas_viajes": max_filas,
        "filas_evaluadas": {nombre: len(dataset.data) for nombre, dataset in suites.items()},
        "diagnostico": {"crudos": diagnostico_crudos, "procesados": diagnostico_procesados},
        "reportes": reportes,
    }
    escribir_resumen(resumen, salida)
    return resumen


def ejecutar_validacion_base(
    max_filas: int = 5_000, directorio_salida: Path = SALIDA_DEFAULT
) -> dict[str, object]:
    """Valida los seis meses de viajes y la nomenclatura de ``data/raw/`` y sus Parquet."""
    rutas = [RAW_DATA_DIR / f"datos_abiertos_{mes}.csv" for mes in URLS_VIAJES]
    nomenclatura = RAW_DATA_DIR / Path(URL_NOMENCLATURA).name
    faltantes = [r for r in [*rutas, nomenclatura, RUTA_VIAJES, RUTA_ESTACIONES] if not r.exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan {[r.name for r in faltantes]}. Ejecuta 'python -m mibici.dataset' primero."
        )
    return validar_dataset_base(
        rutas, nomenclatura, RUTA_VIAJES, RUTA_ESTACIONES, max_filas, directorio_salida
    )


def seleccionar_preguntas(pregunta: str) -> list[str]:
    """Traduce la opción ``--pregunta`` a la lista de validaciones que se ejecutan."""
    seleccion = pregunta.upper()
    if seleccion == "ALL":
        return list(PREGUNTAS)
    if seleccion not in PREGUNTAS:
        raise typer.BadParameter("pregunta debe ser p1, p2, p3, base o all")
    return [seleccion]


@app.command()
def main(
    pregunta: str = typer.Option("all", help="p1, p2, p3, base o all."),
    max_filas: int = typer.Option(
        5_000, min=1, help="Muestra máxima por partición de P1 y P2 y por tabla de viajes."
    ),
    salida: Path = typer.Option(  # noqa: B008
        SALIDA_DEFAULT, help="Directorio para HTML y JSON."
    ),
) -> None:
    """Genera reportes de calidad sin versionar ni exponer los datos fuente."""
    for item in seleccionar_preguntas(pregunta):
        if item in {"P1", "P2"}:
            resumen = ejecutar_validacion(item, max_filas=max_filas, directorio_salida=salida)
        elif item == "P3":
            resumen = ejecutar_validacion_p3(max_filas=max_filas, directorio_salida=salida)
        else:
            resumen = ejecutar_validacion_base(max_filas=max_filas, directorio_salida=salida)
        typer.echo(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
