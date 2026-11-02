"""Coherencia de las constantes de mibici/config.py de las que dependen los resultados."""

from datetime import date

from mibici import config


def test_periodos_se_derivan_de_las_fechas():
    assert config.FECHA_FIN_TRAIN == "2026-08-31"
    assert config.FECHA_FIN_TRAIN_CANDIDATOS == "2026-07-31"
    assert config.PERIODO_TRAIN == f"{config.FECHA_INICIO}/2026-08-31"
    assert config.PERIODO_TEST == f"{config.FECHA_CORTE}/{config.FECHA_FIN}"
    assert config.PERIODO_TRAIN_CANDIDATOS == f"{config.FECHA_INICIO}/2026-07-31"
    assert config.PERIODO_VALIDACION == f"{config.FECHA_VALIDACION}/2026-08-31"
    assert config.PERIODO_CLUSTERING == config.PERIODO_TRAIN


def test_fechas_en_orden():
    fechas = [
        config.FECHA_INICIO,
        config.FECHA_VALIDACION,
        config.FECHA_CORTE,
        config.FECHA_FIN,
    ]
    assert fechas == sorted(fechas)
    assert all(date.fromisoformat(f) for f in fechas)


def test_un_archivo_de_viajes_por_mes_del_periodo():
    inicio, fin = date.fromisoformat(config.FECHA_INICIO), date.fromisoformat(config.FECHA_FIN)
    meses = [f"{inicio.year}_{m:02d}" for m in range(inicio.month, fin.month + 1)]
    assert list(config.URLS_VIAJES) == meses
    for mes, url in config.URLS_VIAJES.items():
        assert url.startswith("https://") and url.endswith(f"datos_abiertos_{mes}.csv")
    assert config.URL_NOMENCLATURA.endswith(f"nomenclatura_{fin.year}_{fin.month:02d}.csv")


def test_semilla_entera_fija():
    assert isinstance(config.SEMILLA, int)
    assert config.SEMILLA == 42


def test_registro_y_metricas_cubren_las_tres_preguntas():
    assert set(config.MODELOS_REGISTRO) == set(config.METRICAS_SELECCION) == {"P1", "P2", "P3"}
    assert all(n.startswith(config.EXPERIMENTO) for n in config.MODELOS_REGISTRO.values())
    assert config.METRICAS_SELECCION["P1"] == ("val_mae", False)
    assert config.METRICAS_SELECCION["P2"] == ("val_average_precision", True)
    assert config.METRICAS_SELECCION["P3"] == ("train_silhouette", True)
    assert config.ALIAS_CHAMPION != config.ALIAS_CHALLENGER


def test_umbrales_en_rango():
    assert 0 < config.UMBRAL_RIESGO_P2 <= 1
    assert 0 < config.UMBRAL_DECISION_P2 < 1
    assert config.HORIZONTE_P2_HORAS >= 1
    assert config.MIN_SALIDAS_ESTACION_ACTIVA >= 1
    assert config.ESTACIONES_DEFAULT in (config.ESTACIONES_TODAS, config.ESTACIONES_ACTIVAS)


def test_rutas_dentro_del_proyecto():
    for ruta in (config.RUTA_VIAJES, config.RUTA_ESTACIONES, config.MANIFIESTO_DATOS):
        assert config.PROJ_ROOT in ruta.parents
    assert config.RUTA_CONFIG.name == "config.py"
    # El manifiesto se versiona; los datos procesados no
    assert config.DATA_DIR not in config.MANIFIESTO_DATOS.parents
