from mibici.monitoring.interpretacion import (
    construir_informe,
    interpretar_periodo,
    recomendar_continuous_training,
)


def _periodo(*, drift_rezagos=True, drift_objetivo=True, drift_dataset=False):
    variables = {
        "estacion_id": {"metodo": "Jensen-Shannon", "valor": 0.02, "umbral": 0.1, "drift": False},
        "rezago_24h": {
            "metodo": "Wasserstein",
            "valor": 0.18,
            "umbral": 0.1,
            "drift": drift_rezagos,
        },
        "rezago_48h": {
            "metodo": "Wasserstein",
            "valor": 0.16,
            "umbral": 0.1,
            "drift": drift_rezagos,
        },
        "rezago_168h": {
            "metodo": "Wasserstein",
            "valor": 0.15,
            "umbral": 0.1,
            "drift": drift_rezagos,
        },
        "demanda": {"metodo": "Wasserstein", "valor": 0.20, "umbral": 0.1, "drift": drift_objetivo},
    }
    n = sum(v["drift"] for v in variables.values())
    return {
        "variables": variables,
        "drift_dataset": drift_dataset,
        "proporcion_con_drift": n / len(variables),
        "periodo_referencia": "2026-04-01/2026-07-31",
        "periodo_actual": "2026-09-01/2026-09-30",
        "version_datos": "abc123",
    }


def test_interpretacion_no_oculta_variables_si_drift_global_es_falso():
    out = interpretar_periodo("test", _periodo(drift_dataset=False))
    assert out["drift_dataset_evidently"] is False
    assert out["prioridad"] == "alta"
    assert out["entradas_criticas_con_drift"] == [
        "rezago_168h",
        "rezago_24h",
        "rezago_48h",
    ]
    assert out["objetivo_con_drift"] is True
    assert any("no marcó drift global" in texto for texto in out["hallazgos"])


def test_interpretacion_distingue_objetivo_de_entradas():
    out = interpretar_periodo(
        "test", _periodo(drift_rezagos=False, drift_objetivo=True)
    )
    assert out["entradas_con_drift"] == []
    assert out["objetivo_con_drift"] is True
    assert out["detalle_drift"]["demanda"]["rol"] == "objetivo"
    assert out["prioridad"] == "media"


def test_sin_drift_recomienda_monitorear():
    periodo = interpretar_periodo(
        "validacion", _periodo(drift_rezagos=False, drift_objetivo=False)
    )
    out = recomendar_continuous_training({"validacion": periodo})
    assert out["accion"] == "mantener_champion_y_monitorear"


def test_drift_sin_degradacion_solo_pide_evaluar_challenger():
    periodo = interpretar_periodo("test", _periodo())
    out = recomendar_continuous_training({"test": periodo})
    assert out["accion"] == "evaluar_challenger"
    assert out["challenger_supera_champion"] is False


def test_calidad_fallida_bloquea_reentrenamiento():
    periodo = interpretar_periodo("test", _periodo())
    out = recomendar_continuous_training({"test": periodo}, calidad_datos_ok=False)
    assert out["accion"] == "bloquear_y_corregir_datos"


def test_promocion_requiere_toda_la_evidencia():
    periodo = interpretar_periodo("test", _periodo())
    out = recomendar_continuous_training(
        {"test": periodo},
        degradacion_confirmada=True,
        challenger_supera_champion=True,
    )
    assert out["accion"] == "promover_challenger_gradualmente"
    assert "rollback" in out["justificacion"]


def test_construir_informe_conserva_advertencia_metodologica():
    out = construir_informe({"test": _periodo()})
    assert "interpretaciones" in out
    assert "continuous_training" in out
    assert "no demuestra" in out["advertencia"]
