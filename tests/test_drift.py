import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.config import (
    PERIODO_TEST,
    PERIODO_TEST_PRIMERA_QUINCENA,
    PERIODO_TRAIN_CANDIDATOS,
    PERIODO_VALIDACION,
)
from mibici.monitoring import drift


def _tablon(n=600, desplazamiento=0.0, semilla=0, inicio="2026-04-01"):
    r = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            drift.FECHA: pd.date_range(inicio, periods=n, freq="h"),
            "estacion_id": r.integers(1, 20, n),
            "dpcapacity": r.integers(10, 30, n),
            "hora": r.integers(0, 24, n),
            "dia_semana": r.integers(0, 7, n),
            "es_fin_de_semana": r.integers(0, 2, n),
            "rezago_24h": r.poisson(3 + desplazamiento, n).astype(float),
            "rezago_48h": r.poisson(3 + desplazamiento, n).astype(float),
            "rezago_168h": r.poisson(3 + desplazamiento, n).astype(float),
            drift.OBJETIVO: r.poisson(3 + desplazamiento, n),
        }
    )


def test_referencia_termina_antes_del_actual():
    fin_ref = PERIODO_TRAIN_CANDIDATOS.split("/")[1]
    inicio_act = PERIODO_TEST.split("/")[0]
    assert fin_ref < inicio_act


def test_cortar_respeta_limites_e_incluye_ultimo_dia():
    out = drift.cortar(_tablon(), "2026-04-10/2026-04-12")
    assert out[drift.FECHA].min() >= pd.Timestamp("2026-04-10")
    assert out[drift.FECHA].max() == pd.Timestamp("2026-04-12 23:00")


def test_validar_columna_faltante():
    with pytest.raises(ValueError, match="faltan columnas"):
        drift.validar(_tablon().drop(columns=["hora"]), "x")


def test_validar_conjunto_vacio():
    with pytest.raises(ValueError, match="vacío"):
        drift.validar(_tablon().iloc[0:0], "x")


def test_validar_detecta_categoria_nueva_e_infinitos():
    ref, act = _tablon(), _tablon(semilla=1)
    act.loc[0, "estacion_id"] = 999
    act.loc[1, "rezago_24h"] = np.inf
    info = drift.validar(act, "actual", ref=ref)
    assert "999" in info["categorias_nuevas"]["estacion_id"]
    assert info["infinitos"] == 1


def test_muestrear_es_determinista():
    df = _tablon()
    assert drift.muestrear(df, 100).equals(drift.muestrear(df, 100))
    assert len(drift.muestrear(df, 0)) == len(df)


def test_ejecutar_genera_html_y_json(tmp_path):
    drift.ejecutar(_tablon(), _tablon(desplazamiento=6, semilla=1), tmp_path)
    assert (tmp_path / "data_drift.html").exists()
    assert (tmp_path / "data_drift.json").exists()


def test_cada_periodo_actual_tiene_su_prefijo():
    assert drift.PERIODOS_ACTUAL == {
        "test": PERIODO_TEST,
        "validacion": PERIODO_VALIDACION,
        "primera_quincena": PERIODO_TEST_PRIMERA_QUINCENA,
    }
    prefijos = {actual: drift.prefijo_reporte(actual) for actual in drift.PERIODOS_ACTUAL}
    assert prefijos == {
        "test": "data_drift",
        "validacion": "data_drift_validacion",
        "primera_quincena": "data_drift_primera_quincena",
    }


def test_cli_primera_quincena_escribe_sus_archivos_sin_tocar_los_demas(tmp_path, monkeypatch):
    """Referencia de abril a julio y actual del 1 al 15 de septiembre, con prefijo propio."""
    # Horas del 1 de abril al 30 de septiembre
    tablon = _tablon(n=183 * 24)
    monkeypatch.setattr(drift, "cargar_datos_procesados", lambda: (None, None))
    monkeypatch.setattr(drift, "generar_dataset_p1", lambda viajes, estaciones: tablon)
    monkeypatch.setattr(drift, "leer_version_datos", lambda: "v-prueba")
    for archivo in ("data_drift.json", "chequeos_datos_test.json"):
        (tmp_path / archivo).write_text("{}", encoding="utf-8")

    resultado = CliRunner().invoke(
        drift.app,
        ["--actual", "primera_quincena", "--max-filas", "500", "--salida", str(tmp_path)],
    )

    assert resultado.exit_code == 0, resultado.output
    assert (tmp_path / "data_drift_primera_quincena.html").exists()
    assert (tmp_path / "data_drift_primera_quincena.json").exists()
    chequeos = json.loads(
        (tmp_path / "chequeos_datos_primera_quincena.json").read_text(encoding="utf-8")
    )
    assert chequeos["periodo_referencia"] == PERIODO_TRAIN_CANDIDATOS
    assert chequeos["periodo_actual"] == "2026-09-01/2026-09-15"
    assert chequeos["actual"]["fecha_min"] == "2026-09-01 00:00:00"
    assert chequeos["actual"]["fecha_max"] == "2026-09-15 23:00:00"
    # Los reportes de septiembre completo no se sobrescriben
    assert (tmp_path / "data_drift.json").read_text(encoding="utf-8") == "{}"
    assert (tmp_path / "chequeos_datos_test.json").read_text(encoding="utf-8") == "{}"


def test_cli_rechaza_un_periodo_desconocido(tmp_path):
    resultado = CliRunner().invoke(drift.app, ["--actual", "octubre", "--salida", str(tmp_path)])

    assert resultado.exit_code != 0
    assert "primera_quincena" in resultado.output
