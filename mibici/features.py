from pathlib import Path
from typing import Tuple

from loguru import logger
import numpy as np
import pandas as pd

from mibici.config import (
    FECHA_FIN,
    FECHA_INICIO,
    PROCESSED_DATA_DIR,
)

# Estaciones con capacidad 0 en nomenclatura que se descartan en P2
ESTACIONES_CAPACIDAD_CERO = [99, 109, 116, 138]
FECHA_CORTE_TEST = "2026-09-01"


def cargar_datos_procesados() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los DataFrames procesados de viajes y estaciones desde Parquet."""
    ruta_viajes = PROCESSED_DATA_DIR / "viajes.parquet"
    ruta_estaciones = PROCESSED_DATA_DIR / "estaciones.parquet"

    if not ruta_viajes.exists() or not ruta_estaciones.exists():
        raise FileNotFoundError(
            "No se encontraron los archivos procesados en data/processed/. "
            "Ejecuta 'make data' primero."
        )

    viajes = pd.read_parquet(ruta_viajes)
    estaciones = pd.read_parquet(ruta_estaciones)
    return viajes, estaciones


def filtrar_viajes_invalidos(df_viajes: pd.DataFrame) -> pd.DataFrame:
    """Descarta viajes de menos de 1 minuto con la misma estacion de origen y destino.

    Representan el 3.6% de los viajes (81,947) y corresponden a prestamos fallidos o
    devoluciones inmediatas que sesgan la demanda real y el balance de inventario.
    """
    duracion = (df_viajes["fin_del_viaje"] - df_viajes["inicio_del_viaje"]).dt.total_seconds()
    espurio = (df_viajes["origen_id"] == df_viajes["destino_id"]) & (duracion < 60)
    logger.info(f"Viajes espurios descartados (<1 min mismo origen-destino): {espurio.sum():,}")
    return df_viajes[~espurio].copy()


def crear_grid_estacion_hora(
    estaciones_ids: np.ndarray,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN,
) -> pd.DataFrame:
    """Genera el producto cartesiano de estaciones y horas completas para el periodo.

    Completa con ceros las combinaciones sin viajes para evitar sesgos de seleccion.
    """
    rango_horas = pd.date_range(
        start=f"{fecha_inicio} 00:00:00",
        end=f"{fecha_fin} 23:00:00",
        freq="h",
    )
    idx = pd.MultiIndex.from_product(
        [rango_horas, estaciones_ids],
        names=["fecha_hora", "estacion_id"],
    )
    return pd.DataFrame(index=idx).reset_index()


def generar_dataset_p1(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el tablon para predecir salidas por estacion y hora (P1).

    Predice el dia d usando solo informacion hasta d-1:
    - rezagos de 24h, 48h y 168h (misma hora semana previa)
    - variables de calendario (hora, dia de la semana, fin de semana)
    - capacidad instalada de la estacion
    """
    logger.info("Construyendo tablon para P1...")
    viajes_limpios = filtrar_viajes_invalidos(viajes)

    viajes_limpios["fecha_hora"] = viajes_limpios["inicio_del_viaje"].dt.floor("h")
    conteo = (
        viajes_limpios.groupby(["fecha_hora", "origen_id"])
        .size()
        .reset_index(name="demanda")
        .rename(columns={"origen_id": "estacion_id"})
    )

    grid = crear_grid_estacion_hora(estaciones["id"].values)
    df_p1 = pd.merge(grid, conteo, on=["fecha_hora", "estacion_id"], how="left")
    df_p1["demanda"] = df_p1["demanda"].fillna(0).astype(int)

    cap_map = estaciones.set_index("id")["dpcapacity"].to_dict()
    df_p1["dpcapacity"] = df_p1["estacion_id"].map(cap_map).fillna(0).astype(int)

    df_p1 = df_p1.sort_values(["estacion_id", "fecha_hora"]).reset_index(drop=True)

    # Variables de calendario
    df_p1["hora"] = df_p1["fecha_hora"].dt.hour
    df_p1["dia_semana"] = df_p1["fecha_hora"].dt.dayofweek
    df_p1["es_fin_de_semana"] = df_p1["dia_semana"].isin([5, 6]).astype(int)
    df_p1["mes"] = df_p1["fecha_hora"].dt.month

    # Rezagos para predecir el dia siguiente sin fuga
    df_p1["rezago_24h"] = df_p1.groupby("estacion_id")["demanda"].shift(24)
    df_p1["rezago_48h"] = df_p1.groupby("estacion_id")["demanda"].shift(48)
    df_p1["rezago_168h"] = df_p1.groupby("estacion_id")["demanda"].shift(168)

    # Descartar primera semana que no tiene rezago semanal
    return df_p1.dropna(subset=["rezago_168h"]).reset_index(drop=True)


def generar_dataset_p2(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    umbral_capacidad: float = 0.30,
) -> pd.DataFrame:
    """Construye el tablon para clasificar riesgo de vaciado o llenado en 2h (P2).

    Saldo neto futuro = llegadas(t+1, t+2) - salidas(t+1, t+2).
    Riesgo = 1 si |saldo| >= 0.30 * dpcapacity, 0 en otro caso.
    Usa solo informacion disponible hasta t (rezagos y medias moviles).
    """
    logger.info("Construyendo tablon para P2...")
    viajes_limpios = filtrar_viajes_invalidos(viajes)

    estaciones_validas = estaciones[~estaciones["id"].isin(ESTACIONES_CAPACIDAD_CERO)].copy()
    valid_ids = estaciones_validas["id"].values

    viajes_limpios["hora_inicio"] = viajes_limpios["inicio_del_viaje"].dt.floor("h")
    viajes_limpios["hora_fin"] = viajes_limpios["fin_del_viaje"].dt.floor("h")

    salidas = (
        viajes_limpios[viajes_limpios["origen_id"].isin(valid_ids)]
        .groupby(["hora_inicio", "origen_id"])
        .size()
        .reset_index(name="salidas")
        .rename(columns={"hora_inicio": "fecha_hora", "origen_id": "estacion_id"})
    )
    llegadas = (
        viajes_limpios[viajes_limpios["destino_id"].isin(valid_ids)]
        .groupby(["hora_fin", "destino_id"])
        .size()
        .reset_index(name="llegadas")
        .rename(columns={"hora_fin": "fecha_hora", "destino_id": "estacion_id"})
    )

    grid = crear_grid_estacion_hora(valid_ids)
    df_p2 = pd.merge(grid, salidas, on=["fecha_hora", "estacion_id"], how="left")
    df_p2 = pd.merge(df_p2, llegadas, on=["fecha_hora", "estacion_id"], how="left")
    df_p2["salidas"] = df_p2["salidas"].fillna(0).astype(int)
    df_p2["llegadas"] = df_p2["llegadas"].fillna(0).astype(int)
    df_p2["saldo_t"] = df_p2["llegadas"] - df_p2["salidas"]

    cap_map = estaciones_validas.set_index("id")["dpcapacity"].to_dict()
    df_p2["dpcapacity"] = df_p2["estacion_id"].map(cap_map).astype(int)

    df_p2 = df_p2.sort_values(["estacion_id", "fecha_hora"]).reset_index(drop=True)

    # Ventana futura t+1 y t+2
    lead1 = df_p2.groupby("estacion_id")["saldo_t"].shift(-1)
    lead2 = df_p2.groupby("estacion_id")["saldo_t"].shift(-2)
    saldo_futuro = lead1 + lead2

    df_p2["saldo_futuro_2h"] = saldo_futuro
    df_p2["riesgo"] = (saldo_futuro.abs() >= (df_p2["dpcapacity"] * umbral_capacidad)).astype(int)

    # Variables historicas observadas hasta t
    df_p2["salidas_lag1"] = df_p2.groupby("estacion_id")["salidas"].shift(1)
    df_p2["llegadas_lag1"] = df_p2.groupby("estacion_id")["llegadas"].shift(1)
    df_p2["saldo_lag1"] = df_p2.groupby("estacion_id")["saldo_t"].shift(1)
    df_p2["saldo_media_3h"] = (
        df_p2.groupby("estacion_id")["saldo_t"]
        .transform(lambda s: s.rolling(window=3, min_periods=3).mean())
    )

    df_p2["hora"] = df_p2["fecha_hora"].dt.hour
    df_p2["dia_semana"] = df_p2["fecha_hora"].dt.dayofweek
    df_p2["es_fin_de_semana"] = df_p2["dia_semana"].isin([5, 6]).astype(int)

    df_p2 = df_p2.dropna(subset=["saldo_futuro_2h", "salidas_lag1", "saldo_media_3h"]).reset_index(drop=True)

    # Descartar las 2 ultimas horas de agosto para que el target no use septiembre en train
    fuga = (df_p2["fecha_hora"] >= "2026-08-31 22:00:00") & (df_p2["fecha_hora"] < "2026-09-01 00:00:00")
    return df_p2[~fuga].reset_index(drop=True)


def generar_dataset_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    solo_train: bool = True,
) -> pd.DataFrame:
    """Construye matriz de perfil de uso por estacion para clustering (P3).

    Calcula la proporcion de viajes por hora, porcentaje en fin de semana y
    balance de salidas vs llegadas. Por defecto usa solo abril a agosto.
    """
    logger.info("Construyendo perfiles para P3...")
    viajes_limpios = filtrar_viajes_invalidos(viajes)

    if solo_train:
        viajes_limpios = viajes_limpios[viajes_limpios["inicio_del_viaje"] < FECHA_CORTE_TEST].copy()

    viajes_limpios["hora"] = viajes_limpios["inicio_del_viaje"].dt.hour
    viajes_limpios["es_fds"] = viajes_limpios["inicio_del_viaje"].dt.dayofweek.isin([5, 6]).astype(int)

    # 1. Distribucion de salidas en las 24 horas
    salidas_h = viajes_limpios.groupby(["origen_id", "hora"]).size().unstack(fill_value=0)
    total_salidas = salidas_h.sum(axis=1).replace(0, np.nan)
    prop_h = salidas_h.div(total_salidas, axis=0).fillna(0)
    prop_h.columns = [f"prop_hora_{h:02d}" for h in range(24)]

    # 2. Uso en fin de semana
    prop_fds = viajes_limpios.groupby("origen_id")["es_fds"].mean().rename("prop_fin_de_semana")

    # 3. Ratio salidas frente a total de actividad
    salidas_est = viajes_limpios.groupby("origen_id").size().rename("salidas")
    llegadas_est = viajes_limpios.groupby("destino_id").size().rename("llegadas")
    flujos = pd.concat([salidas_est, llegadas_est], axis=1).fillna(0)
    suma = (flujos["salidas"] + flujos["llegadas"]).replace(0, np.nan)
    flujos["ratio_salidas"] = (flujos["salidas"] / suma).fillna(0.5)

    df_p3 = estaciones.copy().set_index("id")
    df_p3 = df_p3.join(prop_h, how="left").fillna(0)
    df_p3 = df_p3.join(prop_fds, how="left").fillna(0)
    df_p3 = df_p3.join(flujos[["ratio_salidas"]], how="left").fillna(0.5)

    return df_p3.reset_index()


def particionar_temporalmente(
    df: pd.DataFrame,
    columna_fecha: str = "fecha_hora",
    fecha_corte: str = FECHA_CORTE_TEST,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en entrenamiento y prueba segun la fecha de corte."""
    corte = pd.Timestamp(fecha_corte)
    train = df[df[columna_fecha] < corte].copy()
    test = df[df[columna_fecha] >= corte].copy()
    logger.info(f"Particion: Train={len(train):,}, Test={len(test):,}")
    return train, test
