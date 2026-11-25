"""Pruebas del Model Card y la reconciliación local del Model Registry."""

import json
from pathlib import Path

from mibici.governance.registry_report import (
    SECCIONES_MODEL_CARD,
    construir_reporte,
    validar_model_card,
    validar_registry,
)


def _metadata(alias: str, version: str, run_id: str, fin: str) -> dict:
    return {
        "model_name": "mibici-equipo09-p1-demanda",
        "alias": alias,
        "version": version,
        "model_uri": f"models:/mibici-equipo09-p1-demanda@{alias}",
        "run_id": run_id,
        "periodo_train": f"2026-04-01/{fin}",
    }


def _ab(champion: dict, challenger: dict) -> dict:
    def modelo(metadata: dict, mae: float) -> dict:
        return {
            "mae": mae,
            "rmse": 1.6,
            "wape_pct": 59.0,
            "r2": 0.66,
            "sesgo": -0.2,
            "registro": {
                clave: metadata[clave]
                for clave in ("model_name", "alias", "version", "run_id", "periodo_train")
            },
        }

    return {
        "modelo_a": modelo(champion, 0.88),
        "modelo_b": modelo(challenger, 0.87),
        "comparativa": {"delta_mae": -0.01, "delta_mae_pct": -1.1},
        "analisis_por_dia": {
            "n_dias": 15,
            "dias_ganados_challenger": 10,
            "ic_95_bootstrap": [-0.01, 0.02],
            "p_valor_t_dia": 0.2,
        },
        "decision": {
            "recomendacion": "mantener_champion",
            "base": "analisis_por_dia",
            "justificacion": "La evidencia diaria no es concluyente.",
        },
        "metadatos": {
            "periodo_evaluacion": "2026-09-16/2026-09-30",
            "n_observaciones": 174240,
            "verificacion_temporal": {"primera_fecha_evaluada": "2026-09-16"},
        },
    }


def _model_card(champion: dict) -> str:
    secciones = "\n".join(
        f"## {numero}. {titulo}\nContenido."
        for numero, titulo in enumerate(SECCIONES_MODEL_CARD, 1)
    )
    valores = "\n".join(
        str(champion[campo])
        for campo in ("model_name", "alias", "version", "run_id", "periodo_train")
    )
    return f"# Model Card\n{valores}\n{secciones}\n"


def _escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido), encoding="utf-8")


def test_model_card_completo_y_alineado():
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    assert validar_model_card(_model_card(champion), champion) == []


def test_model_card_detecta_seccion_e_identidad_faltantes():
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    contenido = _model_card(champion).replace("## 7. Limitaciones", "## 7. Riesgos")
    contenido = contenido.replace("run-champion", "run-ausente")
    errores = validar_model_card(contenido, champion)
    assert "Falta la sección: Limitaciones" in errores
    assert any("run_id=run-champion" in error for error in errores)


def test_registry_valida_identidad_temporalidad_y_decision():
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    challenger = _metadata("challenger", "3", "run-challenger", "2026-09-15")
    verificaciones = validar_registry(champion, challenger, _ab(champion, challenger))
    assert all(item["aprobado"] for item in verificaciones)


def test_registry_detecta_ab_con_version_distinta():
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    challenger = _metadata("challenger", "3", "run-challenger", "2026-09-15")
    resultado_ab = _ab(champion, challenger)
    resultado_ab["modelo_b"]["registro"]["version"] = "2"
    verificaciones = validar_registry(champion, challenger, resultado_ab)
    resultado = next(
        item
        for item in verificaciones
        if item["nombre"] == "ab_usa_versiones_exportadas"
    )
    assert not resultado["aprobado"]


def test_registry_detecta_traslape_temporal():
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    challenger = _metadata("challenger", "3", "run-challenger", "2026-09-20")
    verificaciones = validar_registry(champion, challenger, _ab(champion, challenger))
    resultado = next(
        item
        for item in verificaciones
        if item["nombre"] == "evaluacion_fuera_de_entrenamiento"
    )
    assert not resultado["aprobado"]


def test_construir_reporte_aprobado_y_serializable(tmp_path):
    champion = _metadata("champion", "1", "run-champion", "2026-07-31")
    challenger = _metadata("challenger", "3", "run-challenger", "2026-09-15")
    _escribir_json(tmp_path / "models/p1/metadata.json", champion)
    _escribir_json(tmp_path / "models/p1_challenger/metadata.json", challenger)
    _escribir_json(tmp_path / "reports/drift/ab_test_results.json", _ab(champion, challenger))
    (tmp_path / "MODEL_CARD.md").write_text(_model_card(champion), encoding="utf-8")

    reporte = construir_reporte(tmp_path)

    assert reporte["estado"] == "aprobado"
    assert reporte["produccion"]["version"] == "1"
    assert reporte["candidato"]["version"] == "3"
    json.dumps(reporte, allow_nan=False)
