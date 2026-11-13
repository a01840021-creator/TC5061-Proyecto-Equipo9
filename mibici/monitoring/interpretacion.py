"""Interpreta el drift de P1 y propone una acción de Continuous Training.

El módulo consume ``drift_summary.json`` generado por ``monitoring.resumen``.
No sustituye a Evidently: convierte sus resultados estadísticos en evidencia
operativa, separando variables de entrada, objetivo y decisión de promoción.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import typer

from mibici.config import REPORTS_DIR

app = typer.Typer(add_completion=False)

DIR_DRIFT = REPORTS_DIR / "drift"
ARCHIVO_RESUMEN = "drift_summary.json"
OBJETIVO = "demanda"
ENTRADAS_P1 = {
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
}
ENTRADAS_CRITICAS = {"rezago_24h", "rezago_48h", "rezago_168h"}


def _razon_umbral(detalle: dict[str, Any]) -> float:
    """Expresa el score como múltiplo de su umbral; evita dividir entre cero."""
    umbral = float(detalle["umbral"])
    return round(float(detalle["valor"]) / umbral, 4) if umbral else 0.0


def interpretar_periodo(nombre: str, resumen: dict[str, Any]) -> dict[str, Any]:
    """Separa drift de entradas y objetivo y asigna una prioridad operativa."""
    variables = resumen["variables"]
    entradas_drift = sorted(c for c in ENTRADAS_P1 if variables.get(c, {}).get("drift", False))
    criticas_drift = sorted(set(entradas_drift) & ENTRADAS_CRITICAS)
    objetivo_drift = bool(variables.get(OBJETIVO, {}).get("drift", False))

    if criticas_drift and objetivo_drift:
        prioridad = "alta"
    elif entradas_drift or objetivo_drift:
        prioridad = "media"
    else:
        prioridad = "baja"

    detalle_drift = {
        columna: {
            **variables[columna],
            "razon_umbral": _razon_umbral(variables[columna]),
            "rol": "objetivo" if columna == OBJETIVO else "entrada",
        }
        for columna in sorted(set(entradas_drift) | ({OBJETIVO} if objetivo_drift else set()))
    }

    hallazgos = []
    if entradas_drift:
        hallazgos.append(f"Drift en entradas del modelo: {', '.join(entradas_drift)}.")
    if objetivo_drift:
        hallazgos.append("La distribución del objetivo demanda también cambió.")
    if not entradas_drift and not objetivo_drift:
        hallazgos.append("No se detectó drift en entradas ni en el objetivo.")
    if not resumen["drift_dataset"] and (entradas_drift or objetivo_drift):
        hallazgos.append(
            "Evidently no marcó drift global porque la proporción de columnas no alcanzó "
            "el umbral, pero sí existen cambios relevantes a nivel de variable."
        )

    return {
        "periodo": nombre,
        "periodo_referencia": resumen.get("periodo_referencia"),
        "periodo_actual": resumen.get("periodo_actual"),
        "version_datos": resumen.get("version_datos"),
        "drift_dataset_evidently": bool(resumen["drift_dataset"]),
        "proporcion_columnas_drift": float(resumen["proporcion_con_drift"]),
        "entradas_con_drift": entradas_drift,
        "entradas_criticas_con_drift": criticas_drift,
        "objetivo_con_drift": objetivo_drift,
        "prioridad": prioridad,
        "detalle_drift": detalle_drift,
        "hallazgos": hallazgos,
    }


def recomendar_continuous_training(
    interpretaciones: dict[str, dict[str, Any]],
    *,
    calidad_datos_ok: bool = True,
    degradacion_confirmada: bool = False,
    challenger_supera_champion: bool = False,
) -> dict[str, Any]:
    """Aplica una política conservadora y auditable de Continuous Training.

    El drift inicia una investigación, pero no basta para promover una versión.
    La promoción requiere calidad de datos, degradación comprobada y un challenger
    superior sobre el mismo conjunto actual.
    """
    relevantes = [
        nombre
        for nombre, resultado in interpretaciones.items()
        if resultado["entradas_criticas_con_drift"] or resultado["objetivo_con_drift"]
    ]

    if not calidad_datos_ok:
        accion = "bloquear_y_corregir_datos"
        justificacion = "La calidad de datos falló; reentrenar propagaría el problema."
    elif not relevantes:
        accion = "mantener_champion_y_monitorear"
        justificacion = "No hay drift relevante que justifique generar una nueva versión."
    elif not degradacion_confirmada:
        accion = "evaluar_challenger"
        justificacion = (
            "Hay drift en variables críticas u objetivo, pero falta demostrar degradación "
            "del champion con etiquetas del periodo actual."
        )
    elif not challenger_supera_champion:
        accion = "reentrenar_o_ajustar_challenger"
        justificacion = (
            "La degradación está confirmada, pero todavía no existe evidencia de que el "
            "challenger sea superior."
        )
    else:
        accion = "promover_challenger_gradualmente"
        justificacion = (
            "La calidad es adecuada, la degradación fue confirmada y el challenger supera "
            "al champion; procede canary o A/B con rollback antes de mover el alias champion."
        )

    return {
        "accion": accion,
        "periodos_con_senal_relevante": relevantes,
        "calidad_datos_ok": calidad_datos_ok,
        "degradacion_confirmada": degradacion_confirmada,
        "challenger_supera_champion": challenger_supera_champion,
        "justificacion": justificacion,
        "criterios_promocion": [
            "Mismo conjunto de evaluación para champion y challenger.",
            "Mejora en MAE con intervalo de confianza favorable.",
            "Sin deterioro material en RMSE, R² ni latencia.",
            "Datos validados y linaje completo en MLflow.",
            "Despliegue gradual con capacidad de rollback.",
        ],
    }


def construir_informe(datos: dict[str, Any], **estado: bool) -> dict[str, Any]:
    """Construye el artefacto que alimentará el reporte y el A/B testing."""
    interpretaciones = {
        nombre: interpretar_periodo(nombre, periodo) for nombre, periodo in datos.items()
    }
    return {
        "interpretaciones": interpretaciones,
        "continuous_training": recomendar_continuous_training(interpretaciones, **estado),
        "advertencia": (
            "El drift estadístico no demuestra por sí solo degradación predictiva ni autoriza "
            "la promoción automática de un modelo."
        ),
    }


@app.command()
def main(
    directorio: Annotated[Path, typer.Option(help="Carpeta con drift_summary.json.")] = DIR_DRIFT,
    degradacion_confirmada: bool = typer.Option(
        False, help="El champion empeoró en el periodo actual."
    ),
    challenger_supera_champion: bool = typer.Option(False, help="El A/B favorece al challenger."),
    calidad_datos_ok: bool = typer.Option(
        True, help="Los chequeos de calidad del periodo actual fueron satisfactorios."
    ),
) -> None:
    """Genera interpretacion_drift.json para el reporte de la entrega."""
    origen = directorio / ARCHIVO_RESUMEN
    if not origen.exists():
        raise FileNotFoundError(f"No existe {origen}")
    datos = json.loads(origen.read_text(encoding="utf-8"))
    informe = construir_informe(
        datos,
        calidad_datos_ok=calidad_datos_ok,
        degradacion_confirmada=degradacion_confirmada,
        challenger_supera_champion=challenger_supera_champion,
    )
    destino = directorio / "interpretacion_drift.json"
    destino.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Interpretación guardada en {destino}")


if __name__ == "__main__":
    app()
