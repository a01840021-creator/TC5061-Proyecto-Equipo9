"""Valida el Model Card y reconcilia la evidencia local del Model Registry.

El reporte no necesita el servidor de MLflow: contrasta las instantáneas exportadas del
champion y del challenger con el resultado versionado del A/B testing. Así deja una
evidencia reproducible de la decisión de producción desde cualquier clon del repositorio.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Annotated, Any

import typer

from mibici.config import PROJ_ROOT

app = typer.Typer(add_completion=False)

SECCIONES_MODEL_CARD = (
    "Detalles del modelo",
    "Uso previsto",
    "Datos",
    "Variables y salida",
    "Entrenamiento",
    "Evaluación",
    "Limitaciones",
    "Consideraciones éticas",
    "Despliegue y seguridad",
    "Monitoreo y continuous training",
    "Gobernanza y linaje",
    "FinOps",
    "Reproducibilidad",
    "Responsables y cambios",
)


def _leer_json(ruta: Path) -> dict[str, Any]:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise TypeError(f"{ruta} debe contener un objeto JSON")
    return contenido


def _fin_periodo(periodo: str) -> str:
    partes = periodo.split("/")
    if len(partes) != 2:
        raise ValueError(f"Periodo inválido: {periodo}")
    return partes[1]


def validar_model_card(contenido: str, champion: dict[str, Any]) -> list[str]:
    """Devuelve las inconsistencias del Model Card contra el champion exportado."""
    titulos = {
        coincidencia.group(1).strip()
        for coincidencia in re.finditer(r"^##\s+(?:\d+\.\s+)?(.+?)\s*$", contenido, re.MULTILINE)
    }
    errores = [
        f"Falta la sección: {seccion}"
        for seccion in SECCIONES_MODEL_CARD
        if seccion not in titulos
    ]
    for campo in ("model_name", "alias", "version", "run_id", "periodo_train"):
        valor = str(champion.get(campo, "")).strip()
        if not valor:
            errores.append(f"El champion no declara {campo}")
        elif valor not in contenido:
            errores.append(f"El Model Card no contiene {campo}={valor}")
    return errores


def validar_registry(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    ab_test: dict[str, Any],
) -> list[dict[str, Any]]:
    """Contrasta identidades, versiones, ventanas temporales y decisión del A/B."""
    verificaciones: list[dict[str, Any]] = []

    def agregar(nombre: str, aprobado: bool, detalle: str) -> None:
        verificaciones.append({"nombre": nombre, "aprobado": aprobado, "detalle": detalle})

    mismo_modelo = champion.get("model_name") == challenger.get("model_name")
    agregar(
        "mismo_modelo_registrado",
        mismo_modelo,
        f"champion={champion.get('model_name')}; challenger={challenger.get('model_name')}",
    )
    aliases = champion.get("alias") == "champion" and challenger.get("alias") == "challenger"
    agregar(
        "aliases_operativos",
        aliases,
        f"champion={champion.get('alias')}; challenger={challenger.get('alias')}",
    )
    versiones = bool(champion.get("version")) and bool(challenger.get("version"))
    versiones = versiones and champion.get("version") != challenger.get("version")
    agregar(
        "versiones_distintas",
        versiones,
        f"champion=v{champion.get('version')}; challenger=v{challenger.get('version')}",
    )
    linaje = all(
        metadatos.get(campo)
        for metadatos in (champion, challenger)
        for campo in ("run_id", "periodo_train", "model_uri")
    )
    agregar("linaje_minimo", linaje, "run_id, periodo_train y model_uri presentes")

    registro_a = ab_test.get("modelo_a", {}).get("registro", {})
    registro_b = ab_test.get("modelo_b", {}).get("registro", {})
    ab_coincide = all(
        registro.get(campo) == metadatos.get(campo)
        for registro, metadatos in ((registro_a, champion), (registro_b, challenger))
        for campo in ("model_name", "alias", "version", "run_id", "periodo_train")
    )
    agregar(
        "ab_usa_versiones_exportadas",
        ab_coincide,
        "Las identidades del A/B coinciden con ambas instantáneas del Registry",
    )

    primera_evaluacion = (
        ab_test.get("metadatos", {}).get("verificacion_temporal", {}).get("primera_fecha_evaluada")
    )
    try:
        evaluacion_posterior = bool(primera_evaluacion) and all(
            primera_evaluacion > _fin_periodo(str(modelo.get("periodo_train", "")))
            for modelo in (champion, challenger)
        )
    except ValueError:
        evaluacion_posterior = False
    agregar(
        "evaluacion_fuera_de_entrenamiento",
        evaluacion_posterior,
        f"primera evaluación={primera_evaluacion}",
    )

    recomendacion = ab_test.get("decision", {}).get("recomendacion")
    decision_coherente = recomendacion == "mantener_champion"
    agregar(
        "decision_de_produccion",
        decision_coherente,
        f"recomendación={recomendacion}; alias champion permanece en v{champion.get('version')}",
    )
    return verificaciones


def construir_reporte(raiz: Path) -> dict[str, Any]:
    """Construye el inventario auditable del modelo de producción y su alternativa."""
    champion = _leer_json(raiz / "models/p1/metadata.json")
    challenger = _leer_json(raiz / "models/p1_challenger/metadata.json")
    ab_test = _leer_json(raiz / "reports/drift/ab_test_results.json")
    contenido_card = (raiz / "MODEL_CARD.md").read_text(encoding="utf-8")

    verificaciones = validar_registry(champion, challenger, ab_test)
    errores_card = validar_model_card(contenido_card, champion)
    verificaciones.append(
        {
            "nombre": "model_card_completo_y_alineado",
            "aprobado": not errores_card,
            "detalle": "Model Card completo" if not errores_card else "; ".join(errores_card),
        }
    )

    modelo_a = ab_test["modelo_a"]
    modelo_b = ab_test["modelo_b"]
    analisis = ab_test["analisis_por_dia"]
    decision = ab_test["decision"]
    return {
        "proyecto": "TC5061-Proyecto-Equipo9",
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "aprobado"
        if all(item["aprobado"] for item in verificaciones)
        else "inconsistente",
        "modelo_registrado": champion["model_name"],
        "produccion": champion,
        "candidato": challenger,
        "evaluacion_actual": {
            "periodo": ab_test["metadatos"]["periodo_evaluacion"],
            "n_observaciones": ab_test["metadatos"]["n_observaciones"],
            "champion": {
                clave: modelo_a[clave] for clave in ("mae", "rmse", "wape_pct", "r2", "sesgo")
            },
            "challenger": {
                clave: modelo_b[clave] for clave in ("mae", "rmse", "wape_pct", "r2", "sesgo")
            },
            "diferencia_mae": ab_test["comparativa"]["delta_mae"],
            "diferencia_mae_pct": ab_test["comparativa"]["delta_mae_pct"],
            "dias_ganados_challenger": analisis["dias_ganados_challenger"],
            "n_dias": analisis["n_dias"],
            "ic_95_diferencia_mae_diaria": analisis["ic_95_bootstrap"],
            "p_valor_t_dia": analisis["p_valor_t_dia"],
        },
        "decision": {
            "recomendacion": decision["recomendacion"],
            "base": decision["base"],
            "justificacion": decision["justificacion"],
            "accion": "Conservar champion v1; ajustar o ampliar la evaluación del challenger.",
        },
        "verificaciones": verificaciones,
    }


@app.command()
def main(
    salida: Annotated[
        Path,
        typer.Option(help="Ruta del inventario JSON, relativa a la raíz del proyecto."),
    ] = Path("reports/final/registry_inventory.json"),
) -> None:
    """Valida la gobernanza local y guarda el inventario del Registry."""
    reporte = construir_reporte(PROJ_ROOT)
    ruta = salida if salida.is_absolute() else PROJ_ROOT / salida
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps({"estado": reporte["estado"]}, ensure_ascii=False))
    typer.echo(f"Reporte guardado en {ruta}")
    if reporte["estado"] != "aprobado":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
