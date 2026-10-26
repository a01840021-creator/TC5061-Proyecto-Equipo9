from numbers import Integral

from loguru import logger
import numpy as np
import pandas as pd

from mibici.config import (
    DURACION_MINIMA_VIAJE_S,
    FECHA_CORTE,
    FECHA_FIN,
    FECHA_INICIO,
    HORIZONTE_P2_HORAS,
    RUTA_ESTACIONES,
    RUTA_VIAJES,
    UMBRAL_RIESGO_P2,
)


def cargar_datos_procesados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los viajes y las estaciones que deja la ingesta en `data/processed/`.

    Es el punto de entrada común de P1, P2 y P3: las tres preguntas parten de los mismos
    Parquet limpios y versionados, así que ninguna vuelve a leer ni a limpiar los CSV.
    """
    if not RUTA_VIAJES.exists() or not RUTA_ESTACIONES.exists():
        raise FileNotFoundError(
            "No se encontraron los archivos procesados en data/processed/. "
            "Ejecuta 'python -m mibici.dataset' primero."
        )

    viajes = pd.read_parquet(RUTA_VIAJES)
    estaciones = pd.read_parquet(RUTA_ESTACIONES)
    return viajes, estaciones


def filtrar_viajes_invalidos(df_viajes: pd.DataFrame) -> pd.DataFrame:
    """Descarta los viajes más cortos que `DURACION_MINIMA_VIAJE_S` (un minuto) que terminan
    en la estación donde iniciaron.

    Son el 3.6 % de los viajes (81,947) y corresponden a préstamos fallidos o devoluciones
    inmediatas. No son demanda real (P1), no mueven bicicletas entre estaciones (P2) y
    pueden distorsionar el perfil horario de las estaciones donde abundan (P3), por eso se
    quitan antes de construir cualquiera de los tres conjuntos.

    El filtro mira el fin del viaje, que puede caer después del cierre de la ventana de una
    predicción. No usa información posterior a la emisión: cada predicción se emite al
    cierre de su ventana más `RETRASO_DISPONIBILIDAD_SEG`, que es el mismo valor que
    `DURACION_MINIMA_VIAJE_S`. Un viaje espurio que empezó antes del cierre ya terminó a la
    emisión, y uno que sigue abierto a la emisión dura al menos ese retraso y no es espurio,
    así que la decisión del filtro ya es observable al emitir.
    """
    duracion = (df_viajes["fin_del_viaje"] - df_viajes["inicio_del_viaje"]).dt.total_seconds()
    espurio = (df_viajes["origen_id"] == df_viajes["destino_id"]) & (
        duracion < DURACION_MINIMA_VIAJE_S
    )
    logger.info(
        f"Viajes espurios descartados (menos de {DURACION_MINIMA_VIAJE_S} s, "
        f"mismo origen y destino): {espurio.sum():,}"
    )
    return df_viajes[~espurio].copy()


def crear_grid_estacion_hora(
    estaciones_ids: np.ndarray,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN,
) -> pd.DataFrame:
    """Genera todas las combinaciones de estación y hora completa del periodo.

    P1 y P2 se agregan por estación y hora. Si sólo se conservaran las horas con viajes,
    desaparecerían las horas con cero salidas (la mayoría) y los rezagos se calcularían
    sobre horas no consecutivas. La rejilla completa garantiza que cada desplazamiento de
    `shift` equivalga a una hora real y que los ceros entren al entrenamiento y a la
    evaluación.
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
    """Construye el tablón de P1: viajes que inician por estación y hora.

    La predicción para las 24 horas del día d se emite al cierre del día d menos 1 más
    `RETRASO_DISPONIBILIDAD_SEG`, cuando ya se sabe cuáles de las salidas de ese día son
    viajes espurios (ver `filtrar_viajes_invalidos`). Toda variable usa sólo información
    observable a la emisión:

    - `rezago_24h`, `rezago_48h` y `rezago_168h`: salidas de la misma hora uno, dos y
      siete días antes. La demanda de una estación se parece a la de ayer y, más todavía,
      a la del mismo día de la semana anterior; el rezago semanal es además el baseline.
    - `hora`, `dia_semana` y `es_fin_de_semana`: la demanda sigue ciclos diarios y
      semanales que se conocen de antemano. No se incluye el mes: entrenamiento, validación
      y prueba cubren meses distintos, así que el modelo vería valores que nunca se repiten.
    - `dpcapacity`: atributo fijo del tamaño de la estación, como contexto para los modelos
      candidatos; su correlación con la demanda media de entrenamiento es débil (0.29).

    Se descarta la primera semana, que no tiene rezago semanal.
    """
    logger.info("Construyendo tablón para P1...")
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

    # Rezagos de al menos 24 horas: disponibles al emitir, tras el cierre del día anterior
    df_p1["rezago_24h"] = df_p1.groupby("estacion_id")["demanda"].shift(24)
    df_p1["rezago_48h"] = df_p1.groupby("estacion_id")["demanda"].shift(48)
    df_p1["rezago_168h"] = df_p1.groupby("estacion_id")["demanda"].shift(168)

    # Descartar la primera semana, que no tiene rezago semanal
    return df_p1.dropna(subset=["rezago_168h"]).reset_index(drop=True)


def generar_dataset_p2(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    umbral_capacidad: float = UMBRAL_RIESGO_P2,
    horizonte_horas: int = HORIZONTE_P2_HORAS,
) -> pd.DataFrame:
    """Construye el tablón de P2: riesgo de vaciado o llenado en las próximas horas.

    MiBici no publica el inventario de cada estación, pero cada llegada suma una
    bicicleta y cada salida resta una. La etiqueta usa ese saldo neto como aproximación
    del cambio de inventario: vale 1 si en las `horizonte_horas` siguientes el valor
    absoluto de llegadas menos salidas alcanza `umbral_capacidad` veces los anclajes de
    la estación. Un saldo negativo grande anticipa que se vacía; uno positivo, que se llena.

    La fila de la hora t cubre el intervalo [t, t+1) y la predicción se emite en t+1 más
    `RETRASO_DISPONIBILIDAD_SEG`, cuando ya se sabe cuáles de sus salidas son viajes
    espurios (ver `filtrar_viajes_invalidos`).
    Las variables usan sólo horas hasta t: salidas, llegadas y saldo de la hora previa, y
    el saldo medio de las últimas tres horas, porque el desbalance reciente es la
    información más cercana sobre el desbalance próximo; además hora, día de la semana y fin de semana, porque el desbalance sigue
    los ciclos de traslado. Se excluyen las estaciones con capacidad cero en la
    nomenclatura, para las que el umbral no está definido, y se purgan las últimas horas
    de entrenamiento cuya ventana futura cae en el periodo de prueba.

    `horizonte_horas` debe ser un entero mayor o igual a 1; con cero no habría ventana
    futura ni purga, y la etiqueta usaría igualmente la hora siguiente.
    """
    es_entero = isinstance(horizonte_horas, Integral) and not isinstance(horizonte_horas, bool)
    if not es_entero or horizonte_horas < 1:
        raise ValueError(
            f"horizonte_horas debe ser un entero mayor o igual a 1; se recibió {horizonte_horas!r}."
        )

    logger.info("Construyendo tablón para P2...")
    viajes_limpios = filtrar_viajes_invalidos(viajes)

    sin_capacidad = estaciones["dpcapacity"] == 0
    excluidas = estaciones.loc[sin_capacidad, "id"].tolist()
    logger.info(f"Estaciones con capacidad cero excluidas de P2: {excluidas}")
    estaciones_validas = estaciones[~sin_capacidad].copy()
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

    # Ventana futura: horas t+1 a t+horizonte
    saldo_por_estacion = df_p2.groupby("estacion_id")["saldo_t"]
    saldo_futuro = saldo_por_estacion.shift(-1)
    for paso in range(2, horizonte_horas + 1):
        saldo_futuro = saldo_futuro + saldo_por_estacion.shift(-paso)

    columna_futuro = f"saldo_futuro_{horizonte_horas}h"
    df_p2[columna_futuro] = saldo_futuro
    df_p2["riesgo"] = (saldo_futuro.abs() >= (df_p2["dpcapacity"] * umbral_capacidad)).astype(int)

    # Variables históricas observadas hasta t
    df_p2["salidas_lag1"] = df_p2.groupby("estacion_id")["salidas"].shift(1)
    df_p2["llegadas_lag1"] = df_p2.groupby("estacion_id")["llegadas"].shift(1)
    df_p2["saldo_lag1"] = df_p2.groupby("estacion_id")["saldo_t"].shift(1)
    df_p2["saldo_media_3h"] = df_p2.groupby("estacion_id")["saldo_t"].transform(
        lambda s: s.rolling(window=3, min_periods=3).mean()
    )

    df_p2["hora"] = df_p2["fecha_hora"].dt.hour
    df_p2["dia_semana"] = df_p2["fecha_hora"].dt.dayofweek
    df_p2["es_fin_de_semana"] = df_p2["dia_semana"].isin([5, 6]).astype(int)

    df_p2 = df_p2.dropna(subset=[columna_futuro, "salidas_lag1", "saldo_media_3h"]).reset_index(
        drop=True
    )

    # Purgar las últimas horas de entrenamiento cuya etiqueta usaría el periodo de prueba
    return purgar_ventana_futura(df_p2, FECHA_CORTE, horizonte_horas)


def purgar_ventana_futura(
    df: pd.DataFrame,
    fecha_corte: str,
    horizonte_horas: int,
    columna_fecha: str = "fecha_hora",
) -> pd.DataFrame:
    """Quita las horas previas al corte cuya ventana futura de `horizonte_horas` lo cruza.

    La etiqueta de la hora t usa las horas t+1 a t+horizonte, así que las últimas
    `horizonte_horas` antes del corte se etiquetan con horas del periodo siguiente.
    """
    corte = pd.Timestamp(fecha_corte)
    purga = (df[columna_fecha] >= corte - pd.Timedelta(hours=horizonte_horas)) & (
        df[columna_fecha] < corte
    )
    return df[~purga].reset_index(drop=True)


def generar_dataset_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    solo_train: bool = True,
) -> pd.DataFrame:
    """Construye la matriz de perfiles de uso por estación para el agrupamiento de P3.

    P3 busca estaciones que se usan de forma parecida, no estaciones con el mismo volumen.
    Por eso cada estación se describe con proporciones y no con conteos:

    - `prop_hora_00` a `prop_hora_23`: proporción de sus salidas en cada hora del día;
      separa estaciones de salida matutina, de regreso vespertino o de uso nocturno.
    - `prop_fin_de_semana`: proporción de salidas en sábado y domingo; distingue uso
      laboral de uso recreativo.
    - `ratio_salidas`: salidas entre el total de salidas y llegadas; indica si la
      estación funciona más como origen o como destino.

    Las estaciones sin salidas quedan con perfil horario cero y `ratio_salidas` de 0.5.
    La columna `n_salidas` no es una variable del agrupamiento: sirve para identificar
    esas estaciones. Con `solo_train=True`, las salidas se cuentan por inicio anterior al
    corte y las llegadas por fin anterior al corte, de forma independiente, para que
    ningún perfil use un evento ocurrido desde el corte. Los perfiles se emiten en el corte
    más `RETRASO_DISPONIBILIDAD_SEG`, cuando ya se sabe cuáles de las últimas salidas antes
    del corte son viajes espurios (ver `filtrar_viajes_invalidos`).
    """
    logger.info("Construyendo perfiles para P3...")
    viajes_limpios = filtrar_viajes_invalidos(viajes)

    salidas = viajes_limpios
    llegadas = viajes_limpios
    if solo_train:
        corte = pd.Timestamp(FECHA_CORTE)
        salidas = viajes_limpios[viajes_limpios["inicio_del_viaje"] < corte]
        llegadas = viajes_limpios[viajes_limpios["fin_del_viaje"] < corte]

    salidas = salidas.assign(
        hora=salidas["inicio_del_viaje"].dt.hour,
        es_fds=salidas["inicio_del_viaje"].dt.dayofweek.isin([5, 6]).astype(int),
    )

    # 1. Distribución de salidas en las 24 horas
    salidas_h = (
        salidas.groupby(["origen_id", "hora"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    total_salidas = salidas_h.sum(axis=1).replace(0, np.nan)
    prop_h = salidas_h.div(total_salidas, axis=0).fillna(0)
    prop_h.columns = [f"prop_hora_{h:02d}" for h in range(24)]

    # 2. Uso en fin de semana
    prop_fds = salidas.groupby("origen_id")["es_fds"].mean().rename("prop_fin_de_semana")

    # 3. Proporción de salidas sobre el total de actividad
    salidas_est = salidas.groupby("origen_id").size().rename("n_salidas")
    llegadas_est = llegadas.groupby("destino_id").size().rename("n_llegadas")
    flujos = pd.concat([salidas_est, llegadas_est], axis=1).fillna(0)
    suma = (flujos["n_salidas"] + flujos["n_llegadas"]).replace(0, np.nan)
    flujos["ratio_salidas"] = (flujos["n_salidas"] / suma).fillna(0.5)

    df_p3 = estaciones.copy().set_index("id")
    df_p3 = df_p3.join(prop_h, how="left").fillna(0)
    df_p3 = df_p3.join(prop_fds, how="left").fillna(0)
    df_p3 = df_p3.join(flujos[["ratio_salidas"]], how="left").fillna(0.5)
    df_p3 = df_p3.join(salidas_est, how="left").fillna({"n_salidas": 0})
    df_p3["n_salidas"] = df_p3["n_salidas"].astype(int)

    return df_p3.reset_index()


def particionar_temporalmente(
    df: pd.DataFrame,
    columna_fecha: str = "fecha_hora",
    fecha_corte: str = FECHA_CORTE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide un tablón en entrenamiento (antes del corte) y prueba (desde el corte).

    P1 y P2 son pronósticos: una partición aleatoria pondría horas del periodo de prueba
    en el entrenamiento y el modelo vería un futuro que en operación no tendría. El corte
    por fecha reproduce cómo se usarán los modelos.
    """
    corte = pd.Timestamp(fecha_corte)
    train = df[df[columna_fecha] < corte].copy()
    test = df[df[columna_fecha] >= corte].copy()
    logger.info(f"Partición: entrenamiento={len(train):,}, prueba={len(test):,}")
    return train, test
