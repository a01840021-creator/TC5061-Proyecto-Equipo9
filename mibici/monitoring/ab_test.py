"""Comparación A/B testing entre modelos Champion y Challenger para P1.

Evalúa dos modelos sobre exactamente el mismo conjunto de prueba (septiembre 2026),
calcula métricas de error y latencia, realiza pruebas de hipótesis estadísticas pareadas
(t-Student y Wilcoxon) y emite una recomendación justificada de promoción según la
política de Continuous Training.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle  # nosec B403
import time
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import typer

from mibici.config import (
    PERIODO_TEST,
    PERIODO_TRAIN,
    REPORTS_DIR,
    SEMILLA,
)
from mibici.dataset import leer_version_datos
from mibici.features import cargar_datos_procesados, generar_dataset_p1
from mibici.monitoring.drift import FEATURES, cortar

app = typer.Typer(add_completion=False)

DIR_DRIFT = REPORTS_DIR / "drift"
MODELO_CHAMPION_DEFAULT = Path("models/p1")
FEATURES_P1 = FEATURES


def cargar_modelo(origen: str | Path) -> Any:
    """Carga un modelo desde una carpeta local, archivo pickle o URI de MLflow."""
    origen_str = str(origen)

    # 1. Si es URI de MLflow (models:/ o runs:/)
    if origen_str.startswith(("models:/", "runs:/")):
        import mlflow.pyfunc

        logger.info(f"Cargando modelo desde MLflow: {origen_str}")
        return mlflow.pyfunc.load_model(origen_str)

    ruta = Path(origen)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo en la ruta: {ruta}")

    # 2. Si es directorio con model.pkl
    if ruta.is_dir():
        archivo_pkl = ruta / "model.pkl"
        if archivo_pkl.exists():
            with open(archivo_pkl, "rb") as f:
                return pickle.load(f)  # nosec B301
        # Intentar cargar como modelo MLflow local
        import mlflow.pyfunc

        return mlflow.pyfunc.load_model(str(ruta))

    # 3. Si es archivo pickle directo
    if ruta.suffix in {".pkl", ".pickle"}:
        with open(ruta, "rb") as f:
            return pickle.load(f)  # nosec B301

    raise ValueError(f"Formato de modelo no soportado en: {ruta}")


def entrenar_challenger_referencia(
    periodo_entrenamiento: str = PERIODO_TRAIN,
) -> HistGradientBoostingRegressor:
    """Entrena un modelo Challenger reproducible ampliando datos hasta agosto 2026.

    Sigue la recomendación de la política de Continuous Training: reentrenar P1 con
    toda la información previa a septiembre (abril a agosto 2026) manteniendo
    la misma arquitectura de HistGradientBoostingRegressor.
    """
    logger.info(f"Entrenando Challenger de referencia con datos de: {periodo_entrenamiento}")
    viajes, estaciones = cargar_datos_procesados()
    datos = generar_dataset_p1(viajes, estaciones)
    train_df = cortar(datos, periodo_entrenamiento)

    X_train = train_df[FEATURES_P1]
    y_train = train_df["demanda"].to_numpy()

    modelo = HistGradientBoostingRegressor(
        l2_regularization=1.0,
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=63,
        random_state=SEMILLA,
    )
    modelo.fit(X_train, y_train)
    logger.info("Challenger de referencia entrenado exitosamente.")
    return modelo


def preparar_datos_evaluacion(
    periodo: str = PERIODO_TEST,
    max_filas: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga y prepara las variables predictoras y el objetivo para la evaluación."""
    viajes, estaciones = cargar_datos_procesados()
    datos = generar_dataset_p1(viajes, estaciones)
    eval_df = cortar(datos, periodo)

    if max_filas > 0 and len(eval_df) > max_filas:
        eval_df = eval_df.sample(n=max_filas, random_state=SEMILLA).copy()

    X = eval_df[FEATURES_P1].copy()
    y = eval_df["demanda"].to_numpy().astype(float)
    return X, y


def predecir_con_latencia(modelo: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Genera predicciones midiendo el tiempo de inferencia total y por muestra."""
    t0 = time.perf_counter()
    y_pred = modelo.predict(X)
    t1 = time.perf_counter()
    latencia_total_s = t1 - t0
    latencia_ms_muestra = (latencia_total_s / max(len(X), 1)) * 1000.0
    return np.asarray(y_pred, dtype=float), latencia_ms_muestra


def evaluar_desempeno(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    latencia_ms: float,
    nombre: str,
) -> dict[str, Any]:
    """Calcula métricas de regresión y errores residuales por fila."""
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(root_mean_squared_error(y_real, y_pred))
    denominador_wape = float(np.sum(np.abs(y_real)))
    wape = float(np.sum(np.abs(y_real - y_pred)) / max(denominador_wape, 1e-8) * 100.0)
    r2 = float(r2_score(y_real, y_pred))
    errores_absolutos = np.abs(y_real - y_pred)

    return {
        "nombre": nombre,
        "n_muestras": len(y_real),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape_pct": round(wape, 2),
        "r2": round(r2, 4),
        "latencia_ms_muestra": round(latencia_ms, 4),
        "_errores_absolutos": errores_absolutos,
    }


def ejecutar_ab_testing(
    eval_a: dict[str, Any],
    eval_b: dict[str, Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Ejecuta la comparación estadística pareada entre dos evaluaciones."""
    err_a = eval_a["_errores_absolutos"]
    err_b = eval_b["_errores_absolutos"]

    if len(err_a) != len(err_b):
        raise ValueError("Ambos modelos deben evaluarse exactamente sobre las mismas filas.")

    n = len(err_a)
    # diff > 0 significa que error_a > error_b (es decir, el modelo B comete MENOR error)
    diff = err_a - err_b
    diferencia_media_mae = float(np.mean(diff))
    error_estandar = float(stats.sem(diff))
    ic_95_inf = float(diferencia_media_mae - 1.96 * error_estandar)
    ic_95_sup = float(diferencia_media_mae + 1.96 * error_estandar)

    # Prueba t pareada de dos colas
    t_stat, p_val_t = stats.ttest_rel(err_a, err_b)

    # Prueba de Wilcoxon pareada (submuestreo si n es muy grande para no degradar memoria)
    if n > 50_000:
        rng = np.random.default_rng(SEMILLA)
        idx = rng.choice(n, size=20_000, replace=False)
        w_stat, p_val_w = stats.wilcoxon(err_a[idx], err_b[idx])
    else:
        w_stat, p_val_w = stats.wilcoxon(err_a, err_b)

    # Deltas
    delta_mae = eval_b["mae"] - eval_a["mae"]
    delta_mae_pct = (delta_mae / eval_a["mae"]) * 100.0 if eval_a["mae"] > 0 else 0.0

    delta_rmse = eval_b["rmse"] - eval_a["rmse"]
    delta_rmse_pct = (delta_rmse / eval_a["rmse"]) * 100.0 if eval_a["rmse"] > 0 else 0.0

    delta_wape = eval_b["wape_pct"] - eval_a["wape_pct"]
    delta_r2 = eval_b["r2"] - eval_a["r2"]
    delta_latencia = eval_b["latencia_ms_muestra"] - eval_a["latencia_ms_muestra"]

    # Criterios de decisión para Continuous Training
    es_significativo = bool(p_val_t < alpha)
    mejora_mae = bool(delta_mae < 0)
    ic_favorable = bool(ic_95_inf > 0)
    no_deteriora_rmse = bool(delta_rmse <= eval_a["rmse"] * 0.02)
    no_deteriora_r2 = bool(eval_b["r2"] >= eval_a["r2"] * 0.95)

    if mejora_mae and es_significativo and ic_favorable and no_deteriora_rmse and no_deteriora_r2:
        recomendacion = "promover_challenger"
        justificacion = (
            f"El Challenger reduce el MAE en {abs(delta_mae_pct):.2f}% con significancia estadística "
            f"(p = {p_val_t:.2e} < {alpha}) y un intervalo de confianza 95% favorable "
            f"[{ic_95_inf:.4f}, {ic_95_sup:.4f}], sin deteriorar RMSE ni R²."
        )
    elif mejora_mae and not es_significativo:
        recomendacion = "mantener_champion"
        justificacion = (
            "El Challenger muestra una leve reducción en MAE, pero la diferencia no es estadísticamente "
            f"significativa al nivel alpha={alpha} (p = {p_val_t:.4f}). Se sugiere mantener el Champion."
        )
    else:
        recomendacion = "descartar_challenger"
        justificacion = (
            "El Challenger no supera al Champion en las métricas clave del periodo de prueba. "
            "Se debe mantener el modelo Champion actual."
        )

    return {
        "modelo_a": {k: v for k, v in eval_a.items() if not k.startswith("_")},
        "modelo_b": {k: v for k, v in eval_b.items() if not k.startswith("_")},
        "comparativa": {
            "delta_mae": round(float(delta_mae), 4),
            "delta_mae_pct": round(float(delta_mae_pct), 2),
            "delta_rmse": round(float(delta_rmse), 4),
            "delta_rmse_pct": round(float(delta_rmse_pct), 2),
            "delta_wape_pct": round(float(delta_wape), 2),
            "delta_r2": round(float(delta_r2), 4),
            "delta_latencia_ms": round(float(delta_latencia), 4),
        },
        "inferencia_estadistica": {
            "t_stat": round(float(t_stat), 4),
            "p_valor_t": float(p_val_t),
            "wilcoxon_stat": round(float(w_stat), 2),
            "p_valor_wilcoxon": float(p_val_w),
            "alpha": alpha,
            "es_estadisticamente_significativo": es_significativo,
            "diferencia_media_mae": round(diferencia_media_mae, 4),
            "ic_95_diferencia": [round(ic_95_inf, 4), round(ic_95_sup, 4)],
        },
        "decision": {
            "recomendacion": recomendacion,
            "criterios_cumplidos": {
                "mejora_mae": mejora_mae,
                "estadisticamente_significativo": es_significativo,
                "ic_favorable": ic_favorable,
                "no_deteriora_rmse": no_deteriora_rmse,
                "no_deteriora_r2": no_deteriora_r2,
            },
            "justificacion": justificacion,
        },
    }


def guardar_resultados_ab(
    resultados: dict[str, Any],
    carpeta_salida: Path = DIR_DRIFT,
) -> Path:
    """Guarda el artefacto JSON auditable de la prueba A/B."""
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_archivo = carpeta_salida / "ab_test_results.json"
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultados de A/B testing guardados en: {ruta_archivo}")
    return ruta_archivo


def formatear_tabla_markdown(resultados: dict[str, Any]) -> str:
    """Genera la tabla comparativa de métricas en formato Markdown limpio."""
    mod_a = resultados["modelo_a"]
    mod_b = resultados["modelo_b"]
    comp = resultados["comparativa"]
    inf = resultados["inferencia_estadistica"]

    lineas = [
        "| Métrica | Champion (v1) | Challenger (v2) | Diferencia Absoluta | Variación (%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **MAE** | {mod_a['mae']:.4f} | {mod_b['mae']:.4f} | {comp['delta_mae']:+.4f} | {comp['delta_mae_pct']:+.2f}% |",
        f"| **RMSE** | {mod_a['rmse']:.4f} | {mod_b['rmse']:.4f} | {comp['delta_rmse']:+.4f} | {comp['delta_rmse_pct']:+.2f}% |",
        f"| **WAPE** | {mod_a['wape_pct']:.2f}% | {mod_b['wape_pct']:.2f}% | {comp['delta_wape_pct']:+.2f}% | - |",
        f"| **$R^2$** | {mod_a['r2']:.4f} | {mod_b['r2']:.4f} | {comp['delta_r2']:+.4f} | - |",
        f"| **Latencia (ms/muestra)** | {mod_a['latencia_ms_muestra']:.4f} | {mod_b['latencia_ms_muestra']:.4f} | {comp['delta_latencia_ms']:+.4f} | - |",
        "",
        f"**Prueba de significancia:** $t = {inf['t_stat']:.4f}$, $p = {inf['p_valor_t']:.2e}$ (alfa = {inf['alpha']}).",
        f"**Intervalo de confianza 95% para la mejora:** [{inf['ic_95_diferencia'][0]:+.4f}, {inf['ic_95_diferencia'][1]:+.4f}].",
        f"**Veredicto:** `{resultados['decision']['recomendacion'].upper()}` - {resultados['decision']['justificacion']}",
    ]
    return "\n".join(lineas)


@app.command()
def main(
    modelo_a: str = typer.Option(
        str(MODELO_CHAMPION_DEFAULT),
        help="Ruta local, carpeta o URI de MLflow del modelo Champion (A).",
    ),
    modelo_b: str = typer.Option(
        "auto",
        help="Ruta o URI del modelo Challenger (B). Si es 'auto', entrena el Challenger de referencia.",
    ),
    periodo: str = typer.Option(
        PERIODO_TEST,
        help="Periodo de evaluación (por omisión, septiembre de 2026).",
    ),
    max_filas: int = typer.Option(
        0,
        help="Filas máximas de evaluación; 0 = todas las filas disponibles.",
    ),
    alpha: float = typer.Option(
        0.05,
        help="Nivel de significancia estadística.",
    ),
    salida: Annotated[
        Path, typer.Option(help="Carpeta para guardar el reporte JSON.")
    ] = DIR_DRIFT,
) -> None:
    """Ejecuta el A/B Testing entre Champion y Challenger sobre el periodo de prueba."""
    logger.info("Iniciando pipeline de A/B Testing para P1...")
    logger.info(f"Modelo Champion: {modelo_a}")
    logger.info(f"Modelo Challenger: {modelo_b}")
    logger.info(f"Periodo de evaluación: {periodo}")

    # 1. Cargar datos de prueba
    X, y = preparar_datos_evaluacion(periodo=periodo, max_filas=max_filas)
    logger.info(f"Evaluando sobre {len(X):,} observaciones de {periodo}.")

    # 2. Cargar o entrenar modelo A (Champion)
    mod_a = cargar_modelo(modelo_a)

    # 3. Cargar o entrenar modelo B (Challenger)
    if modelo_b.lower() == "auto":
        mod_b = entrenar_challenger_referencia()
    else:
        mod_b = cargar_modelo(modelo_b)

    # 4. Inferencia con medición de latencia
    y_pred_a, lat_a = predecir_con_latencia(mod_a, X)
    y_pred_b, lat_b = predecir_con_latencia(mod_b, X)

    # 5. Evaluar desempeño individual
    eval_a = evaluar_desempeno(y, y_pred_a, lat_a, nombre="champion_v1")
    eval_b = evaluar_desempeno(y, y_pred_b, lat_b, nombre="challenger_v2")

    # 6. Comparación estadística pareada
    resultados = ejecutar_ab_testing(eval_a, eval_b, alpha=alpha)
    resultados["metadatos"] = {
        "periodo_evaluacion": periodo,
        "n_observaciones": len(X),
        "version_datos": leer_version_datos(),
        "modelo_a_origen": str(modelo_a),
        "modelo_b_origen": str(modelo_b),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 7. Guardar artefacto JSON
    guardar_resultados_ab(resultados, carpeta_salida=salida)

    # 8. Imprimir tabla formateada
    print("\n" + "=" * 70)
    print("REPORTE DE A/B TESTING: P1 DEMANDA (SEPTIEMBRE 2026)")
    print("=" * 70)
    print(formatear_tabla_markdown(resultados))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    app()
