import hashlib
import json
from pathlib import Path
import re
import unicodedata

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    FECHA_FIN,
    FECHA_INICIO,
    MANIFIESTO_DATOS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RUTA_ESTACIONES,
    RUTA_VIAJES,
    URL_NOMENCLATURA,
    URLS_VIAJES,
)

app = typer.Typer()

TIMEOUT_DESCARGA = (10, 60)  # (conexión, lectura) en segundos
# mibici.net responde 406 al User-Agent por defecto de python-requests
CABECERAS = {"User-Agent": "mibici-ingesta (TC5061-Proyecto-Equipo9)"}
CODIFICACIONES = ("utf-8", "latin-1")

COLUMNAS_VIAJES = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_de_nacimiento",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
]
# Datos personales que ninguna de las tres preguntas del proyecto necesita
COLUMNAS_PERSONALES = ["usuario_id", "genero", "anio_de_nacimiento"]
COLUMNAS_FECHA = ["inicio_del_viaje", "fin_del_viaje"]
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"
COLUMNAS_ID_VIAJES = ["viaje_id", "origen_id", "destino_id"]

COLUMNAS_ESTACIONES = ["id", "name", "latitude", "longitude", "dpcapacity"]


class ErrorIngesta(Exception):
    """Falla de la ingesta con un mensaje listo para mostrarse al usuario."""


def descargar(url: str, destino: Path) -> None:
    """Descarga `url` en streaming hacia `destino`, salvo que el archivo ya exista."""
    if destino.exists():
        logger.info(f"{destino.name} ya existe, no se descarga de nuevo.")
        return

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    logger.info(f"Descargando {url}")
    try:
        with requests.get(
            url, stream=True, timeout=TIMEOUT_DESCARGA, headers=CABECERAS
        ) as respuesta:
            respuesta.raise_for_status()
            total = int(respuesta.headers.get("Content-Length", 0)) or None
            with (
                open(parcial, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=destino.name) as barra,
            ):
                for bloque in respuesta.iter_content(chunk_size=1 << 20):
                    f.write(bloque)
                    barra.update(len(bloque))
        parcial.rename(destino)
    except (requests.RequestException, OSError) as e:
        parcial.unlink(missing_ok=True)
        raise ErrorIngesta(f"No se pudo descargar {url}: {e}") from e


def normalizar_nombre(nombre: str) -> str:
    """Pasa un nombre de columna a snake_case ASCII sin acentos (Año -> anio)."""
    nombre = nombre.replace("﻿", "").strip()
    nombre = nombre.replace("ñ", "ni").replace("Ñ", "Ni")
    nombre = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", nombre.lower()).strip("_")


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV probando UTF-8 y, si falla, latin-1; deja las columnas normalizadas."""
    for codificacion in CODIFICACIONES:
        try:
            df = pd.read_csv(ruta, encoding=codificacion, dtype=str)
        except UnicodeDecodeError:
            logger.warning(f"{ruta.name}: no se pudo leer como {codificacion}, se prueba otra.")
            continue
        logger.info(f"{ruta.name}: leído con codificación {codificacion}.")
        df.columns = [normalizar_nombre(c) for c in df.columns]
        return df
    raise ErrorIngesta(f"{ruta.name}: no se pudo leer con {' ni '.join(CODIFICACIONES)}.")


def validar_columnas(df: pd.DataFrame, esperadas: list[str], archivo: str) -> None:
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ErrorIngesta(
            f"{archivo}: faltan las columnas {faltantes}. Columnas encontradas: {list(df.columns)}"
        )


def a_entero(df: pd.DataFrame, columnas: list[str], archivo: str) -> None:
    for col in columnas:
        if df[col].isna().any():
            raise ErrorIngesta(f"{archivo}: la columna {col} tiene {df[col].isna().sum()} nulos.")
        try:
            df[col] = df[col].astype("int64")
        except ValueError as e:
            raise ErrorIngesta(f"{archivo}: la columna {col} no es entera: {e}") from e


def procesar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee y tipa un mes de viajes. Las columnas personales se quitan después de deduplicar."""
    df = leer_csv(ruta)
    validar_columnas(df, COLUMNAS_VIAJES, ruta.name)
    df = df[COLUMNAS_VIAJES].copy()
    for col in COLUMNAS_FECHA:
        df[col] = pd.to_datetime(df[col], format=FORMATO_FECHA)
    a_entero(df, COLUMNAS_ID_VIAJES, ruta.name)
    return df


def procesar_estaciones(ruta: Path) -> pd.DataFrame:
    df = leer_csv(ruta)
    validar_columnas(df, COLUMNAS_ESTACIONES, ruta.name)
    df = df[COLUMNAS_ESTACIONES].copy()
    a_entero(df, ["id", "dpcapacity"], ruta.name)
    df["latitude"] = df["latitude"].astype("float64")
    df["longitude"] = df["longitude"].astype("float64")
    return df


def quitar_sin_estacion(
    viajes: pd.DataFrame, estaciones: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Descarta viajes con origen o destino ausente de la nomenclatura.

    Devuelve los viajes conservados y, por cada estación ausente, cuántos viajes
    descartados la involucran (un viaje con ambas puntas ausentes cuenta en cada una).
    """
    conocidas = estaciones["id"]
    sin_origen = ~viajes["origen_id"].isin(conocidas)
    sin_destino = ~viajes["destino_id"].isin(conocidas)
    descartados = viajes[sin_origen | sin_destino]
    ausentes = sorted(
        set(viajes.loc[sin_origen, "origen_id"]) | set(viajes.loc[sin_destino, "destino_id"])
    )
    por_estacion = {
        int(i): int(((descartados["origen_id"] == i) | (descartados["destino_id"] == i)).sum())
        for i in ausentes
    }
    return viajes[~(sin_origen | sin_destino)], por_estacion


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def calcular_version_datos(hashes: list[str]) -> str:
    """Primeros 12 caracteres del SHA-256 de los hashes de los archivos, en orden."""
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()[:12]


def escribir_manifiesto(archivos: list[dict], destino: Path = MANIFIESTO_DATOS) -> str:
    version = calcular_version_datos([a["sha256"] for a in archivos])
    manifiesto = {"version_datos": version, "archivos": archivos}
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(f"Manifiesto escrito en {destino}")
    return version


def leer_version_datos(manifiesto: Path = MANIFIESTO_DATOS) -> str:
    """Devuelve `version_datos` del manifiesto, para registrarla como parámetro en MLflow."""
    return json.loads(manifiesto.read_text(encoding="utf-8"))["version_datos"]


@app.command()
def main():
    try:
        ejecutar()
    except ErrorIngesta as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


def ejecutar() -> None:
    fuentes = {f"datos_abiertos_{mes}.csv": url for mes, url in URLS_VIAJES.items()}
    fuentes[Path(URL_NOMENCLATURA).name] = URL_NOMENCLATURA

    logger.info("Descargando datos crudos...")
    for nombre, url in fuentes.items():
        descargar(url, RAW_DATA_DIR / nombre)

    logger.info("Procesando viajes...")
    archivos = []
    tablas = []
    for mes, url in URLS_VIAJES.items():
        ruta = RAW_DATA_DIR / f"datos_abiertos_{mes}.csv"
        df = procesar_viajes(ruta)
        tablas.append(df.assign(mes_archivo=mes))
        archivos.append(
            {"nombre": ruta.name, "url": url, "registros": len(df), "sha256": sha256_archivo(ruta)}
        )
    viajes = pd.concat(tablas, ignore_index=True)
    n_crudos = len(viajes)

    logger.info("Procesando estaciones...")
    ruta_nomenclatura = RAW_DATA_DIR / Path(URL_NOMENCLATURA).name
    estaciones = procesar_estaciones(ruta_nomenclatura)
    archivos.append(
        {
            "nombre": ruta_nomenclatura.name,
            "url": URL_NOMENCLATURA,
            "registros": len(estaciones),
            "sha256": sha256_archivo(ruta_nomenclatura),
        }
    )

    # Duplicado exacto: fila idéntica en todas las columnas del CSV original
    viajes = viajes.drop_duplicates(subset=COLUMNAS_VIAJES, ignore_index=True)
    n_duplicados = n_crudos - len(viajes)
    logger.info(f"Duplicados exactos eliminados: {n_duplicados}")

    viajes, descartados_por_estacion = quitar_sin_estacion(viajes, estaciones)
    n_sin_estacion = n_crudos - n_duplicados - len(viajes)
    logger.info(f"Viajes descartados por estación ausente de la nomenclatura: {n_sin_estacion}")

    viajes_por_mes = viajes["mes_archivo"].value_counts().sort_index()
    viajes = viajes.drop(columns=COLUMNAS_PERSONALES + ["mes_archivo"]).reset_index(drop=True)

    fuera = ~viajes["inicio_del_viaje"].between(
        pd.Timestamp(FECHA_INICIO),
        pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1),
        inclusive="left",
    )
    if fuera.any():
        logger.warning(
            f"{fuera.sum()} viajes inician fuera de {FECHA_INICIO} a {FECHA_FIN}; no se filtran."
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    viajes.to_parquet(RUTA_VIAJES, index=False)
    estaciones.to_parquet(RUTA_ESTACIONES, index=False)

    version = escribir_manifiesto(archivos)

    logger.success("Ingesta completa.")
    typer.echo("\nResumen de la ingesta")
    for mes, n in viajes_por_mes.items():
        typer.echo(f"  Viajes {mes}: {n:,}")
    typer.echo(f"  Registros crudos: {n_crudos:,}")
    typer.echo(f"  Duplicados exactos eliminados: {n_duplicados:,}")
    typer.echo(f"  Viajes descartados por estación fuera de la nomenclatura: {n_sin_estacion:,}")
    for estacion, n in descartados_por_estacion.items():
        typer.echo(f"    Estación {estacion}: {n:,} viajes")
    typer.echo(f"  Total de viajes: {len(viajes):,}")
    typer.echo(f"  Estaciones: {len(estaciones):,}")
    typer.echo(f"  Versión de datos: {version}")


if __name__ == "__main__":
    app()
