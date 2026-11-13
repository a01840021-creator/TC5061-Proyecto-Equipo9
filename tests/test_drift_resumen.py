import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import drift, resumen


def _tablon(n=600, desplazamiento=0.0, semilla=0):
    r = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            drift.FECHA: pd.date_range("2026-04-01", periods=n, freq="h"),
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


def test_resumir_lee_el_json_real_de_evidently(tmp_path):
    drift.ejecutar(_tablon(), _tablon(desplazamiento=6, semilla=1), tmp_path)
    out = resumen.resumir(tmp_path / "data_drift.json")
    assert out["total_columnas"] == 9
    assert out["columnas_con_drift"] == len(out["variables_con_drift"])
    assert "demanda" in out["variables_con_drift"]
    assert "dpcapacity" not in out["variables_con_drift"]


def test_resumir_rechaza_json_sin_metricas(tmp_path):
    ruta = tmp_path / "vacio.json"
    ruta.write_text('{"metrics": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="métricas de drift"):
        resumen.resumir(ruta)


def _json_evidently(tmp_path, conteo):
    import json

    metricas = [
        {
            "metric_name": "DriftedColumnsCount(drift_share=0.5)",
            "value": {"count": float(conteo), "share": conteo / 2},
        },
        {
            "metric_name": "ValueDrift(column=a,method=K-S p_value,threshold=0.05)",
            "value": 0.01,
        },
        {
            "metric_name": "ValueDrift(column=b,method=K-S p_value,threshold=0.05)",
            "value": 0.9,
        },
    ]
    ruta = tmp_path / "fabricado.json"
    ruta.write_text(json.dumps({"metrics": metricas}), encoding="utf-8")
    return ruta


def test_resumir_pruebas_de_valor_p_marcan_drift_con_valor_bajo(tmp_path):
    out = resumen.resumir(_json_evidently(tmp_path, conteo=1))
    assert out["variables_con_drift"] == ["a"]


def test_resumir_falla_si_no_coincide_con_el_conteo_de_evidently(tmp_path):
    with pytest.raises(ValueError, match="Evidently reporta"):
        resumen.resumir(_json_evidently(tmp_path, conteo=2))
