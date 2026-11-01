import numpy as np
import pandas as pd
import pytest

from mibici import features
from mibici.config import (
    DURACION_MINIMA_VIAJE_S,
    FECHA_CORTE,
    FECHA_INICIO,
    RETRASO_DISPONIBILIDAD_SEG,
)
from mibici.features import (
    cargar_datos_procesados,
    crear_grid_estacion_hora,
    filtrar_viajes_invalidos,
    generar_dataset_p1,
    generar_dataset_p2,
    generar_dataset_p3,
    particionar_temporalmente,
    purgar_ventana_futura,
)

ESTACIONES_PRUEBA = pd.DataFrame(
    {"id": [1, 2], "name": ["a", "b"], "latitude": [0.0, 0.0], "longitude": [0.0, 0.0]}
).assign(dpcapacity=10)


@pytest.mark.parametrize("horizonte", [0, -1, 1.5, "2", True])
def test_p2_rechaza_horizonte_invalido(horizonte):
    viajes = pd.DataFrame(columns=["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"])
    estaciones = pd.DataFrame({"id": [1], "dpcapacity": [10]})
    with pytest.raises(ValueError, match="horizonte_horas"):
        generar_dataset_p2(viajes, estaciones, horizonte_horas=horizonte)


def test_p2_acepta_horizonte_de_una_hora():
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2026-04-02 08:10:00"]),
            "fin_del_viaje": pd.to_datetime(["2026-04-02 08:40:00"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    )
    estaciones = pd.DataFrame({"id": [1, 2], "dpcapacity": [10, 10]})
    df_p2 = generar_dataset_p2(viajes, estaciones, horizonte_horas=1)
    assert "saldo_futuro_1h" in df_p2.columns


def test_p1_no_incluye_mes():
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2026-04-02 08:10:00"]),
            "fin_del_viaje": pd.to_datetime(["2026-04-02 08:40:00"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    )
    estaciones = pd.DataFrame({"id": [1, 2], "dpcapacity": [10, 10]})
    assert "mes" not in generar_dataset_p1(viajes, estaciones).columns


def test_p3_no_cuenta_llegadas_desde_el_corte():
    corte = pd.Timestamp(FECHA_CORTE)
    viajes = pd.DataFrame(
        {
            # Sale antes del corte y llega después: cuenta la salida, no la llegada
            "inicio_del_viaje": [corte - pd.Timedelta(minutes=10), corte - pd.Timedelta(days=1)],
            "fin_del_viaje": [corte + pd.Timedelta(minutes=10), corte - pd.Timedelta(hours=23)],
            "origen_id": [1, 2],
            "destino_id": [2, 1],
        }
    )
    estaciones = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "latitude": [0.0, 0.0], "longitude": [0.0, 0.0]}
    ).assign(dpcapacity=10)
    perfiles = generar_dataset_p3(viajes, estaciones, solo_train=True).set_index("id")
    # Estación 1: una salida y una llegada antes del corte; estación 2: una salida y ninguna
    # llegada antes del corte, porque la suya ocurre después
    assert perfiles.loc[1, "ratio_salidas"] == 0.5
    assert perfiles.loc[2, "ratio_salidas"] == 1.0
    assert perfiles["n_salidas"].tolist() == [1, 1]


def viajes_con(*extra: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Un viaje normal de 1 a 2 una hora antes del corte, más viajes de 1 a 1."""
    corte = pd.Timestamp(FECHA_CORTE)
    inicios = [corte - pd.Timedelta(minutes=50)] + [inicio for inicio, _ in extra]
    fines = [corte - pd.Timedelta(minutes=20)] + [fin for _, fin in extra]
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicios,
            "fin_del_viaje": fines,
            "origen_id": [1] + [1] * len(extra),
            "destino_id": [2] + [1] * len(extra),
        }
    )


def tablones(viajes: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        generar_dataset_p1(viajes, ESTACIONES_PRUEBA),
        generar_dataset_p2(viajes, ESTACIONES_PRUEBA),
        generar_dataset_p3(viajes, ESTACIONES_PRUEBA, solo_train=True),
    ]


def test_filtro_espurio_usa_el_retraso_de_disponibilidad():
    assert DURACION_MINIMA_VIAJE_S == RETRASO_DISPONIBILIDAD_SEG


@pytest.mark.parametrize("horas_despues_del_corte", [0, 7])
def test_espurio_que_termina_dentro_del_retraso_no_cambia_variables(horas_despues_del_corte):
    # Cierre de una ventana horaria; en el corte coinciden las de P1, P2 y P3
    cierre = pd.Timestamp(FECHA_CORTE) + pd.Timedelta(hours=horas_despues_del_corte)
    retraso = pd.Timedelta(seconds=RETRASO_DISPONIBILIDAD_SEG)
    # Como el viaje 45706651: sale 15 s antes del cierre y vuelve 7 s después
    espurio = (cierre - pd.Timedelta(seconds=15), cierre + pd.Timedelta(seconds=7))
    assert espurio[1] < cierre + retraso
    assert filtrar_viajes_invalidos(viajes_con(espurio)).shape[0] == 1

    # A la emisión, cierre más retraso, ya se sabe que el viaje volvió a su origen en menos
    # de un minuto: el escenario observable es el mismo que si el viaje no existiera
    for con_espurio, sin_espurio in zip(tablones(viajes_con(espurio)), tablones(viajes_con())):
        pd.testing.assert_frame_equal(con_espurio, sin_espurio)


def test_viaje_abierto_a_la_emision_cuenta_sin_importar_cuando_termina():
    cierre = pd.Timestamp(FECHA_CORTE)
    emision = cierre + pd.Timedelta(seconds=RETRASO_DISPONIBILIDAD_SEG)
    inicio = cierre - pd.Timedelta(seconds=15)
    # Mismo pasado observable a la emisión; sólo cambia un fin todavía futuro
    pronto = viajes_con((inicio, emision + pd.Timedelta(seconds=15)))
    tarde = viajes_con((inicio, cierre + pd.Timedelta(minutes=45)))
    assert filtrar_viajes_invalidos(pronto).shape[0] == 2
    assert filtrar_viajes_invalidos(tarde).shape[0] == 2
    for a, b in zip(tablones(pronto), tablones(tarde)):
        pd.testing.assert_frame_equal(a, b)


# Rejilla, partición, purga y carga ----------------------------------------------------


def test_grid_estacion_hora_cubre_todas_las_combinaciones():
    grid = crear_grid_estacion_hora(np.array([5, 7]), "2026-04-01", "2026-04-02")
    assert len(grid) == 48 * 2
    assert grid["fecha_hora"].min() == pd.Timestamp("2026-04-01 00:00")
    assert grid["fecha_hora"].max() == pd.Timestamp("2026-04-02 23:00")
    assert set(grid["estacion_id"]) == {5, 7}
    assert not grid.duplicated().any()


def test_grid_sin_estaciones_queda_vacio():
    assert crear_grid_estacion_hora(np.array([], dtype=int), "2026-04-01", "2026-04-01").empty


def test_particion_temporal_pone_el_corte_en_prueba():
    corte = pd.Timestamp(FECHA_CORTE)
    datos = pd.DataFrame(
        {"fecha_hora": [corte - pd.Timedelta(hours=1), corte, corte + pd.Timedelta(hours=1)]}
    )
    train, test = particionar_temporalmente(datos)
    assert train["fecha_hora"].tolist() == [corte - pd.Timedelta(hours=1)]
    assert test["fecha_hora"].tolist() == [corte, corte + pd.Timedelta(hours=1)]


def test_particion_temporal_no_modifica_la_entrada():
    datos = pd.DataFrame({"fecha_hora": pd.date_range("2026-08-31 22:00", periods=4, freq="h")})
    train, _ = particionar_temporalmente(datos)
    train["fecha_hora"] = pd.NaT
    assert datos["fecha_hora"].notna().all()


@pytest.mark.parametrize("horizonte", [1, 2, 3])
def test_purgar_ventana_futura_quita_solo_las_horas_que_cruzan_el_corte(horizonte):
    corte = pd.Timestamp(FECHA_CORTE)
    horas = pd.date_range(corte - pd.Timedelta(hours=5), corte + pd.Timedelta(hours=2), freq="h")
    resultado = purgar_ventana_futura(pd.DataFrame({"fecha_hora": horas}), FECHA_CORTE, horizonte)
    quitadas = set(horas) - set(resultado["fecha_hora"])
    assert quitadas == {corte - pd.Timedelta(hours=h) for h in range(1, horizonte + 1)}


def test_p1_rezagos_usan_la_misma_estacion_y_hora(tablon_p1):
    fila = tablon_p1.iloc[500]
    misma = tablon_p1[tablon_p1["estacion_id"] == fila["estacion_id"]].set_index("fecha_hora")
    for horas in (24, 48, 168):
        assert (
            fila[f"rezago_{horas}h"]
            == misma.loc[fila["fecha_hora"] - pd.Timedelta(hours=horas), "demanda"]
        )
    # La primera semana se descarta porque no tiene rezago semanal
    assert tablon_p1["fecha_hora"].min() == pd.Timestamp(FECHA_INICIO) + pd.Timedelta(days=7)


def test_p2_excluye_estaciones_con_capacidad_cero():
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2026-04-02 08:10:00"]),
            "fin_del_viaje": pd.to_datetime(["2026-04-02 08:40:00"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    )
    estaciones = pd.DataFrame({"id": [1, 2], "dpcapacity": [10, 0]})
    assert set(generar_dataset_p2(viajes, estaciones)["estacion_id"]) == {1}


def test_cargar_datos_procesados_sin_archivos(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "RUTA_VIAJES", tmp_path / "viajes.parquet")
    monkeypatch.setattr(features, "RUTA_ESTACIONES", tmp_path / "estaciones.parquet")
    with pytest.raises(FileNotFoundError, match="python -m mibici.dataset"):
        cargar_datos_procesados()


def test_cargar_datos_procesados_lee_los_parquet(tmp_path, monkeypatch, datos_sinteticos):
    viajes, estaciones = datos_sinteticos
    viajes.to_parquet(tmp_path / "viajes.parquet", index=False)
    estaciones.to_parquet(tmp_path / "estaciones.parquet", index=False)
    monkeypatch.setattr(features, "RUTA_VIAJES", tmp_path / "viajes.parquet")
    monkeypatch.setattr(features, "RUTA_ESTACIONES", tmp_path / "estaciones.parquet")
    leidos, leidas = cargar_datos_procesados()
    pd.testing.assert_frame_equal(leidos, viajes)
    pd.testing.assert_frame_equal(leidas, estaciones)
