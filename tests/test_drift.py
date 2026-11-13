import numpy as np
import pandas as pd
import pytest

from mibici.config import PERIODO_TEST, PERIODO_TRAIN_CANDIDATOS
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
