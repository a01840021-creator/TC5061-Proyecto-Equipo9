"""Comparación A/B testing entre modelos Champion y Challenger para P1.

Evalúa los modelos sobre exactamente las mismas filas del periodo del A/B (por omisión, del
16 al 30 de septiembre de 2026, posterior a todo el entrenamiento del champion y del
challenger reentrenado), calcula métricas de error y latencia y emite una recomendación
justificada de promoción según la política de Continuous Training.

La decisión se toma con un análisis por día: para cada día se calcula la diferencia de MAE
(champion menos challenger), se remuestrean los días con reemplazo para obtener un
intervalo bootstrap de la diferencia media y se aplica una prueba t pareada sobre los MAE
diarios. Las pruebas pareadas por fila (t-Student y Wilcoxon) se conservan como
referencia: suponen filas independientes, y las filas de una misma hora comparten la
demanda de todo el sistema, así que con cientos de miles de filas exageran la evidencia.

Con ``--incluir-referencia`` evalúa también, sobre las mismas filas, un challenger entrenado
de abril a agosto que no se registra (la opción ``auto`` de ``--modelo-b``).

Antes de predecir se verifica que la primera fecha evaluada sea posterior al fin del
entrenamiento de cada modelo: el periodo que declara el ``metadata.json`` de su carpeta
exportada o, para ``auto``, el periodo con que se entrena aquí. Un modelo que no declara su
periodo no se evalúa.

Las estadísticas se deciden y se guardan con precisión completa; sólo la tabla Markdown y
los textos redondean. El JSON es estricto (sin ``Infinity`` ni ``NaN``): cuando un
estadístico no está definido se guarda nulo junto con el caso que lo explica.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
import json
import math
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
    METADATOS_SERVICIO,
    MODELO_CHALLENGER_DIR,
    MODELO_SERVICIO_DIR,
    N_BOOTSTRAP_AB,
    PERIODO_AB,
    PERIODO_TRAIN,
    PROJ_ROOT,
    REPORTS_DIR,
    SEMILLA,
)
from mibici.dataset import leer_version_datos
from mibici.features import cargar_datos_procesados, generar_dataset_p1
from mibici.modeling.candidates import limpiar_x
from mibici.monitoring.drift import FEATURES, cortar

app = typer.Typer(add_completion=False)

# El challenger de referencia (auto) se entrena de abril a agosto
PERIODO_TRAIN_REFERENCIA = PERIODO_TRAIN
DIR_DRIFT = REPORTS_DIR / "drift"
# Rutas relativas a la raíz, para que el JSON no guarde rutas locales de quien ejecuta
MODELO_CHAMPION_DEFAULT = MODELO_SERVICIO_DIR.relative_to(PROJ_ROOT)
MODELO_CHALLENGER_DEFAULT = MODELO_CHALLENGER_DIR.relative_to(PROJ_ROOT)
FEATURES_P1 = FEATURES
REFERENCIA_ABR_AGO = "challenger_abr_ago_sin_registrar"
ETIQUETA_REFERENCIA = "Challenger abr-ago (sin registrar)"
NOTA_POR_FILA = (
    "Referencia, no decide: las pruebas por fila suponen filas independientes, pero las "
    "estaciones de una misma hora y día comparten la demanda del sistema, así que el error "
    "estándar por fila subestima la incertidumbre. La decisión usa el análisis por día."
)
# Casos en que todas las diferencias pareadas son iguales: la varianza es cero y la t no
# está definida, así que el estadístico y su valor p se guardan nulos con este caso
DIFERENCIAS_NULAS = "diferencias_nulas"
DIFERENCIAS_CONSTANTES = "diferencias_constantes_no_nulas"
CONVENCION_T_DEGENERADA = {
    DIFERENCIAS_NULAS: (
        "Todas las diferencias son cero: la t y su valor p no están definidos y la prueba "
        "no cuenta como significativa."
    ),
    DIFERENCIAS_CONSTANTES: (
        "Todas las diferencias son iguales y distintas de cero: la varianza es cero, la t y "
        "su valor p no están definidos y, por convención, la prueba cuenta como "
        "significativa (es el límite de la t cuando la varianza tiende a cero). No es una "
        "garantía estadística: con pocos días conviene revisar las diferencias diarias."
    ),
}
CONVENCION_DECISION = (
    "Se promueve sólo si el MAE baja, el límite inferior del intervalo bootstrap 95% de la "
    "diferencia diaria de MAE es mayor que cero, la t pareada por día es significativa "
    "(p < alfa) y RMSE y R² no empeoran más allá de lo tolerado. Todo se evalúa con "
    "precisión completa, sin redondear. Si todas las diferencias diarias son iguales, la t "
    "queda nula y se aplica la convención de analisis_por_dia.convencion_t_dia."
)


def resolver_ruta(origen: str | Path) -> Path:
    """Ruta tal cual o, si no existe y es relativa, relativa a la raíz del repositorio."""
    ruta = Path(origen)
    if not ruta.exists() and not ruta.is_absolute() and (PROJ_ROOT / ruta).exists():
        return PROJ_ROOT / ruta
    return ruta


def cargar_modelo(origen: str | Path) -> Any:
    """Carga un modelo desde una carpeta local, archivo pickle o URI de MLflow."""
    origen_str = str(origen)

    # 1. Si es URI de MLflow (models:/ o runs:/)
    if origen_str.startswith(("models:/", "runs:/")):
        import mlflow.pyfunc

        logger.info(f"Cargando modelo desde MLflow: {origen_str}")
        return mlflow.pyfunc.load_model(origen_str)

    ruta = resolver_ruta(origen)
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


def describir_modelo(origen: str | Path, rol: str) -> dict[str, Any]:
    """Nombre, alias y versión de un modelo, tomados del ``metadata.json`` de su carpeta.

    ``auto`` es el challenger de abril a agosto que se entrena al vuelo y no se registra.
    Un origen sin ``metadata.json`` (un archivo o una URI) se identifica por su nombre.
    """
    base = {"rol": rol, "origen": str(origen)}
    if str(origen).lower() == "auto":
        return {
            **base,
            "nombre": REFERENCIA_ABR_AGO,
            "etiqueta": ETIQUETA_REFERENCIA,
            "model_name": None,
            "alias": None,
            "version": None,
            "periodo_train": PERIODO_TRAIN_REFERENCIA,
        }
    ruta = resolver_ruta(origen)
    archivo = ruta / METADATOS_SERVICIO
    if ruta.is_dir() and archivo.is_file():
        metadatos = json.loads(archivo.read_text(encoding="utf-8"))
        alias, version = metadatos["alias"], str(metadatos["version"])
        return {
            **base,
            "nombre": f"{alias}_v{version}",
            "etiqueta": f"{alias.capitalize()} (v{version})",
            "model_name": metadatos["model_name"],
            "alias": alias,
            "version": version,
            "run_id": metadatos.get("run_id"),
            "periodo_train": metadatos.get("periodo_train"),
        }
    nombre = Path(str(origen)).stem or str(origen)
    return {
        **base,
        "nombre": nombre,
        "etiqueta": nombre,
        "model_name": None,
        "alias": None,
        "version": None,
        "periodo_train": None,
    }


def verificar_sin_traslape(fechas: np.ndarray, modelos: list[dict[str, Any]]) -> dict[str, Any]:
    """Verifica que la primera fecha evaluada sea posterior al entrenamiento de cada modelo.

    ``modelos`` son descripciones de ``describir_modelo``. Falla si un modelo no declara su
    periodo de entrenamiento o si ese periodo termina el mismo día de la primera fecha
    evaluada o después. Devuelve lo verificado, para dejarlo en el JSON.
    """
    if len(fechas) == 0:
        raise ValueError("No hay filas que evaluar en el periodo del A/B.")
    primera = pd.Timestamp(min(fechas)).date()
    fines = {}
    for modelo in modelos:
        etiqueta = modelo.get("etiqueta", modelo["origen"])
        periodo = modelo.get("periodo_train")
        if not periodo:
            raise ValueError(
                f"El modelo {etiqueta} ({modelo['origen']}) no declara su periodo de "
                "entrenamiento, así que no se puede verificar que el A/B no evalúe sobre sus "
                "datos de entrenamiento. Expórtalo con mibici.modeling.serving para que su "
                "metadata.json incluya periodo_train."
            )
        fin = date.fromisoformat(periodo.split("/")[1])
        if primera <= fin:
            raise ValueError(
                f"El A/B empieza el {primera} y el modelo {etiqueta} se entrenó con "
                f"{periodo}: evaluaría sobre sus datos de entrenamiento. El periodo de "
                f"evaluación debe empezar después del {fin}."
            )
        fines[modelo["nombre"]] = periodo
    return {"primera_fecha_evaluada": str(primera), "periodos_train": fines}


@lru_cache(maxsize=1)
def cargar_tablon_p1() -> pd.DataFrame:
    """Tablón de P1 completo, construido una sola vez por ejecución."""
    viajes, estaciones = cargar_datos_procesados()
    return generar_dataset_p1(viajes, estaciones)


def entrenar_challenger_referencia(
    periodo_entrenamiento: str = PERIODO_TRAIN_REFERENCIA,
) -> HistGradientBoostingRegressor:
    """Entrena un modelo Challenger reproducible ampliando datos hasta agosto 2026.

    Sigue la recomendación de la política de Continuous Training: reentrenar P1 con
    toda la información previa a septiembre (abril a agosto 2026) manteniendo
    la misma arquitectura de HistGradientBoostingRegressor. No se registra; sirve de
    referencia frente al challenger registrado.
    """
    logger.info(f"Entrenando Challenger de referencia con datos de: {periodo_entrenamiento}")
    train_df = cortar(cargar_tablon_p1(), periodo_entrenamiento)

    X_train = limpiar_x(train_df, FEATURES_P1)
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
    periodo: str = PERIODO_AB,
    max_filas: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Variables predictoras, objetivo y fecha (día) de cada fila del periodo de evaluación.

    X se prepara con ``limpiar_x``, igual que en el entrenamiento de los candidatos.
    """
    eval_df = cortar(cargar_tablon_p1(), periodo)

    if max_filas > 0 and len(eval_df) > max_filas:
        eval_df = eval_df.sample(n=max_filas, random_state=SEMILLA).copy()

    X = limpiar_x(eval_df, FEATURES_P1)
    y = eval_df["demanda"].to_numpy().astype(float)
    fechas = eval_df["fecha_hora"].dt.date.to_numpy()
    return X, y, fechas


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
    metadatos: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calcula métricas de regresión y errores residuales por fila.

    ``metadatos`` (de ``describir_modelo``) aporta la etiqueta de la tabla y el registro.
    Las métricas se guardan con precisión completa; la tabla las redondea al mostrarlas.
    """
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(root_mean_squared_error(y_real, y_pred))
    denominador_wape = float(np.sum(np.abs(y_real)))
    wape = float(np.sum(np.abs(y_real - y_pred)) / max(denominador_wape, 1e-8) * 100.0)
    r2 = float(r2_score(y_real, y_pred))
    sesgo = float(np.mean(y_pred - y_real))
    errores_absolutos = np.abs(y_real - y_pred)

    resultado = {
        "nombre": nombre,
        "n_muestras": len(y_real),
        "mae": mae,
        "rmse": rmse,
        "wape_pct": wape,
        "r2": r2,
        "sesgo": sesgo,
        "latencia_ms_muestra": float(latencia_ms),
        "_errores_absolutos": errores_absolutos,
    }
    if metadatos:
        resultado["etiqueta"] = metadatos.get("etiqueta", nombre)
        resultado["registro"] = {k: v for k, v in metadatos.items() if k != "etiqueta"}
    return resultado


def finito(valor: float | None) -> float | None:
    """El valor como ``float`` o ``None`` si no es finito, para un JSON estricto."""
    if valor is None:
        return None
    valor = float(valor)
    return valor if math.isfinite(valor) else None


def t_pareada(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, str | None]:
    """Prueba t pareada de dos colas entre ``a`` y ``b``: estadístico, valor p y caso.

    Si todas las diferencias son iguales la varianza es cero y la t no está definida: el
    estadístico y el valor p son ``None`` y el caso es ``DIFERENCIAS_NULAS`` o
    ``DIFERENCIAS_CONSTANTES`` (ver ``CONVENCION_T_DEGENERADA``). Si no, el caso es ``None``.
    """
    diferencias = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diferencias) < 2:
        raise ValueError("La prueba t pareada requiere al menos dos pares.")
    if np.ptp(diferencias) == 0:
        caso = DIFERENCIAS_NULAS if diferencias[0] == 0 else DIFERENCIAS_CONSTANTES
        return None, None, caso
    t_stat, p_valor = stats.ttest_rel(a, b)
    return finito(t_stat), finito(p_valor), None


def es_significativa(p_valor: float | None, caso: str | None, alpha: float) -> bool:
    """p < alfa o, si la t no está definida, la convención de ``CONVENCION_T_DEGENERADA``."""
    if caso is not None:
        return caso == DIFERENCIAS_CONSTANTES
    return p_valor is not None and p_valor < alpha


def bootstrap_dias(
    diferencias: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_AB,
    semilla: int = SEMILLA,
) -> dict[str, Any]:
    """Bootstrap de días con reemplazo de la diferencia media de MAE.

    ``diferencias`` tiene una entrada por día (MAE champion menos MAE challenger), así que
    una diferencia positiva favorece al challenger. Devuelve el intervalo de confianza del
    95% por percentiles y la proporción de remuestreos en que gana el challenger.
    """
    diferencias = np.asarray(diferencias, dtype=float)
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, len(diferencias), size=(n_bootstrap, len(diferencias)))
    medias = diferencias[indices].mean(axis=1)
    ic_inf, ic_sup = np.percentile(medias, [2.5, 97.5])
    return {
        "diferencia_mae_media": float(diferencias.mean()),
        "ic_95_bootstrap": [float(ic_inf), float(ic_sup)],
        "proporcion_remuestreos_gana_challenger": float(np.mean(medias > 0)),
        "n_bootstrap": n_bootstrap,
        "semilla": semilla,
    }


def mae_por_dia(errores_absolutos: np.ndarray, fechas: np.ndarray) -> pd.Series:
    """MAE de cada día, ordenado por fecha."""
    return pd.Series(np.asarray(errores_absolutos, dtype=float)).groupby(fechas).mean()


def analisis_por_dia(
    err_a: np.ndarray,
    err_b: np.ndarray,
    fechas: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = N_BOOTSTRAP_AB,
    semilla: int = SEMILLA,
) -> dict[str, Any]:
    """Compara los MAE diarios de dos modelos: bootstrap de días y t pareada por día.

    Cada día pesa lo mismo, tenga las filas que tenga. Todo se guarda con precisión
    completa, para poder reconstruir las estadísticas desde el JSON.
    """
    fechas = np.asarray(fechas)
    mae_a = mae_por_dia(err_a, fechas)
    mae_b = mae_por_dia(err_b, fechas)
    diferencias = mae_a - mae_b
    if len(diferencias) < 2:
        raise ValueError("El análisis por día requiere al menos dos días.")
    t_stat, p_valor, caso = t_pareada(mae_a.to_numpy(), mae_b.to_numpy())
    bootstrap = bootstrap_dias(diferencias.to_numpy(), n_bootstrap=n_bootstrap, semilla=semilla)
    return {
        "n_dias": len(diferencias),
        "dias_ganados_challenger": int((diferencias > 0).sum()),
        "diferencia_mae_media": bootstrap["diferencia_mae_media"],
        "ic_95_bootstrap": bootstrap["ic_95_bootstrap"],
        "proporcion_remuestreos_gana_challenger": bootstrap[
            "proporcion_remuestreos_gana_challenger"
        ],
        "n_bootstrap": n_bootstrap,
        "semilla": semilla,
        "t_stat_dia": t_stat,
        "p_valor_t_dia": p_valor,
        "caso_degenerado_t_dia": caso,
        "convencion_t_dia": CONVENCION_T_DEGENERADA.get(caso),
        "alpha": alpha,
        "por_dia": [
            {
                "fecha": str(fecha),
                "mae_a": float(mae_a[fecha]),
                "mae_b": float(mae_b[fecha]),
                "diferencia_mae": float(diferencias[fecha]),
            }
            for fecha in diferencias.index
        ],
    }


def comparar_por_fila(err_a: np.ndarray, err_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Pruebas pareadas por fila (t y Wilcoxon), que se conservan como referencia.

    Si todas las diferencias son iguales, la t queda nula con su caso (``t_pareada``); si
    todas son cero, también Wilcoxon, que no tiene pares con diferencia que ordenar.
    """
    n = len(err_a)
    # diff > 0 significa que error_a > error_b (es decir, el modelo B comete MENOR error)
    diff = err_a - err_b
    diferencia_media_mae = float(np.mean(diff))
    error_estandar = float(stats.sem(diff))
    ic_95_inf = float(diferencia_media_mae - 1.96 * error_estandar)
    ic_95_sup = float(diferencia_media_mae + 1.96 * error_estandar)

    # Prueba t pareada de dos colas
    t_stat, p_val_t, caso_t = t_pareada(err_a, err_b)

    # Prueba de Wilcoxon pareada (submuestreo si n es muy grande para no degradar memoria)
    if n > 50_000:
        rng = np.random.default_rng(SEMILLA)
        idx = rng.choice(n, size=20_000, replace=False)
        muestra_a, muestra_b = err_a[idx], err_b[idx]
    else:
        muestra_a, muestra_b = err_a, err_b
    if np.array_equal(muestra_a, muestra_b):
        w_stat, p_val_w, caso_w = None, None, DIFERENCIAS_NULAS
    else:
        w_stat, p_val_w = stats.wilcoxon(muestra_a, muestra_b)
        caso_w = None

    return {
        "n_filas": int(n),
        "t_stat": t_stat,
        "p_valor_t": p_val_t,
        "caso_degenerado_t": caso_t,
        "wilcoxon_stat": finito(w_stat),
        "p_valor_wilcoxon": finito(p_val_w),
        "caso_degenerado_wilcoxon": caso_w,
        "alpha": alpha,
        "es_estadisticamente_significativo": es_significativa(p_val_t, caso_t, alpha),
        "diferencia_media_mae": finito(diferencia_media_mae),
        "ic_95_diferencia": [finito(ic_95_inf), finito(ic_95_sup)],
        "nota": NOTA_POR_FILA,
    }


def calcular_comparativa(eval_a: dict[str, Any], eval_b: dict[str, Any]) -> dict[str, float]:
    """Diferencias de cada métrica del modelo B respecto del A (B menos A)."""
    delta_mae = eval_b["mae"] - eval_a["mae"]
    delta_mae_pct = (delta_mae / eval_a["mae"]) * 100.0 if eval_a["mae"] > 0 else 0.0

    delta_rmse = eval_b["rmse"] - eval_a["rmse"]
    delta_rmse_pct = (delta_rmse / eval_a["rmse"]) * 100.0 if eval_a["rmse"] > 0 else 0.0

    delta_wape = eval_b["wape_pct"] - eval_a["wape_pct"]
    delta_r2 = eval_b["r2"] - eval_a["r2"]
    delta_latencia = eval_b["latencia_ms_muestra"] - eval_a["latencia_ms_muestra"]

    return {
        "delta_mae": float(delta_mae),
        "delta_mae_pct": float(delta_mae_pct),
        "delta_rmse": float(delta_rmse),
        "delta_rmse_pct": float(delta_rmse_pct),
        "delta_wape_pct": float(delta_wape),
        "delta_r2": float(delta_r2),
        "delta_latencia_ms": float(delta_latencia),
    }


def ejecutar_ab_testing(
    eval_a: dict[str, Any],
    eval_b: dict[str, Any],
    alpha: float = 0.05,
    fechas: np.ndarray | None = None,
    n_bootstrap: int = N_BOOTSTRAP_AB,
    semilla: int = SEMILLA,
) -> dict[str, Any]:
    """Ejecuta la comparación estadística pareada entre dos evaluaciones.

    ``fechas`` da el día de cada fila; el análisis por día es el que decide. La
    recomendación es ``promover_challenger`` sólo si el límite inferior del intervalo
    bootstrap es mayor que cero, la t por día tiene p < alpha y RMSE y R² no empeoran
    (``CONVENCION_DECISION``). Los criterios usan valores sin redondear.
    """
    err_a = eval_a["_errores_absolutos"]
    err_b = eval_b["_errores_absolutos"]

    if len(err_a) != len(err_b):
        raise ValueError("Ambos modelos deben evaluarse exactamente sobre las mismas filas.")
    if fechas is None or len(fechas) != len(err_a):
        raise ValueError("El análisis por día requiere la fecha de cada fila evaluada.")

    por_dia = analisis_por_dia(err_a, err_b, fechas, alpha, n_bootstrap, semilla)
    por_fila = comparar_por_fila(err_a, err_b, alpha)
    comparativa = calcular_comparativa(eval_a, eval_b)

    # Criterios de decisión para Continuous Training, con el análisis por día
    mejora_mae = bool(np.mean(err_a) - np.mean(err_b) > 0)
    ic_inf, ic_sup = por_dia["ic_95_bootstrap"]
    ic_bootstrap_favorable = bool(ic_inf > 0)
    t_dia_significativa = es_significativa(
        por_dia["p_valor_t_dia"], por_dia["caso_degenerado_t_dia"], alpha
    )
    no_deteriora_rmse = bool(comparativa["delta_rmse"] <= eval_a["rmse"] * 0.02)
    no_deteriora_r2 = bool(eval_b["r2"] >= eval_a["r2"] * 0.95)
    significativo = ic_bootstrap_favorable and t_dia_significativa

    reduccion = abs(comparativa["delta_mae_pct"])
    dias = f"{por_dia['dias_ganados_challenger']} de {por_dia['n_dias']} días ganados"
    evidencia = (
        f"intervalo bootstrap 95% de la diferencia diaria de MAE [{ic_inf:+.4f}, {ic_sup:+.4f}], "
        f"t por día {texto_t(por_dia['t_stat_dia'], por_dia['p_valor_t_dia'], prefijo='= ')}"
    )
    if mejora_mae and significativo and no_deteriora_rmse and no_deteriora_r2:
        recomendacion = "promover_challenger"
        justificacion = (
            f"El Challenger reduce el MAE en {reduccion:.2f}% ({dias}); {evidencia}: el "
            f"intervalo excluye el cero y p < alfa = {alpha}, sin deteriorar RMSE ni R²."
        )
    elif mejora_mae and significativo:
        recomendacion = "mantener_champion"
        justificacion = (
            f"El Challenger reduce el MAE en {reduccion:.2f}% con evidencia por día ({evidencia}), "
            "pero deteriora RMSE o R² más allá de lo tolerado. Se mantiene el Champion."
        )
    elif mejora_mae:
        recomendacion = "mantener_champion"
        justificacion = (
            f"El Challenger reduce el MAE en {reduccion:.2f}% ({dias}), pero la evidencia por "
            f"día no es concluyente al nivel alfa = {alpha}: {evidencia}. Se mantiene el Champion."
        )
    else:
        recomendacion = "descartar_challenger"
        justificacion = (
            f"El Challenger no reduce el MAE del Champion en el periodo evaluado ({dias}; "
            f"{evidencia}). Se debe mantener el modelo Champion actual."
        )

    return {
        "modelo_a": {k: v for k, v in eval_a.items() if not k.startswith("_")},
        "modelo_b": {k: v for k, v in eval_b.items() if not k.startswith("_")},
        "comparativa": comparativa,
        "analisis_por_dia": por_dia,
        "referencia_por_fila": por_fila,
        "decision": {
            "recomendacion": recomendacion,
            "base": "analisis_por_dia",
            "criterios_cumplidos": {
                "mejora_mae": mejora_mae,
                "ic_bootstrap_favorable": ic_bootstrap_favorable,
                "t_por_dia_significativa": t_dia_significativa,
                "no_deteriora_rmse": no_deteriora_rmse,
                "no_deteriora_r2": no_deteriora_r2,
            },
            "convencion": CONVENCION_DECISION,
            "justificacion": justificacion,
        },
    }


def agregar_referencia(
    resultados: dict[str, Any], comparacion_referencia: dict[str, Any]
) -> dict[str, Any]:
    """Añade al resultado el challenger de abril a agosto, que no participa en la decisión."""
    resultados[REFERENCIA_ABR_AGO] = {
        "modelo": comparacion_referencia["modelo_b"],
        "comparativa": comparacion_referencia["comparativa"],
        "analisis_por_dia": comparacion_referencia["analisis_por_dia"],
        "referencia_por_fila": comparacion_referencia["referencia_por_fila"],
        "nota": "Referencia informativa evaluada sobre las mismas filas; no decide.",
    }
    return resultados


def resumen_modelos(resultados: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Métricas de cada modelo evaluado, por su rol: champion, challenger y referencia."""
    modelos = {"champion": resultados["modelo_a"]}
    rol_b = resultados["modelo_b"].get("registro", {}).get("rol", "challenger_registrado")
    modelos[rol_b] = resultados["modelo_b"]
    if REFERENCIA_ABR_AGO in resultados:
        modelos[REFERENCIA_ABR_AGO] = resultados[REFERENCIA_ABR_AGO]["modelo"]
    return modelos


def guardar_resultados_ab(
    resultados: dict[str, Any],
    carpeta_salida: Path = DIR_DRIFT,
) -> Path:
    """Guarda el artefacto JSON auditable de la prueba A/B."""
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_archivo = carpeta_salida / "ab_test_results.json"
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        # Estricto: un Infinity o un NaN es un error, no un token ajeno al estándar JSON
        json.dump(resultados, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info(f"Resultados de A/B testing guardados en: {ruta_archivo}")
    return ruta_archivo


def texto_t(t_stat: float | None, p_valor: float | None, prefijo: str = "t = ") -> str:
    """Estadístico y valor p redondeados o, si la t no está definida, por qué."""
    if t_stat is None or p_valor is None:
        return "no definida (todas las diferencias son iguales)"
    return f"{prefijo}{t_stat:.2f} (p = {p_valor:.2e})"


def _celdas_comparacion(comparacion: dict[str, Any]) -> list[str]:
    """Celdas de diferencia, intervalo, días ganados y valores p contra el champion."""
    comp = comparacion["comparativa"]
    dia = comparacion["analisis_por_dia"]
    fila = comparacion["referencia_por_fila"]
    ic_inf, ic_sup = dia["ic_95_bootstrap"]
    return [
        f"{comp['delta_mae']:+.4f} ({comp['delta_mae_pct']:+.2f}%)",
        f"{dia['diferencia_mae_media']:+.4f} [{ic_inf:+.4f}, {ic_sup:+.4f}]",
        f"{dia['dias_ganados_challenger']} de {dia['n_dias']}",
        f"{dia['proporcion_remuestreos_gana_challenger'] * 100:.1f}%",
        texto_t(dia["t_stat_dia"], dia["p_valor_t_dia"]),
        texto_t(fila["t_stat"], fila["p_valor_t"]),
        (
            f"W = {fila['wilcoxon_stat']:,.0f} (p = {fila['p_valor_wilcoxon']:.2e})"
            if fila["wilcoxon_stat"] is not None and fila["p_valor_wilcoxon"] is not None
            else "no definida (todas las diferencias son cero)"
        ),
    ]


def formatear_tabla_markdown(resultados: dict[str, Any]) -> str:
    """Genera la tabla comparativa de métricas en formato Markdown limpio.

    Las columnas salen de la etiqueta de cada modelo (``metadata.json`` de su carpeta); si
    el resultado incluye el challenger de abril a agosto, aparece como tercera columna.
    """
    modelos = [resultados["modelo_a"], resultados["modelo_b"]]
    comparaciones = [resultados]
    if REFERENCIA_ABR_AGO in resultados:
        modelos.append(resultados[REFERENCIA_ABR_AGO]["modelo"])
        comparaciones.append(resultados[REFERENCIA_ABR_AGO])
    etiquetas = [m.get("etiqueta", m.get("nombre", "")) for m in modelos]

    def fila(nombre: str, celdas: list[str]) -> str:
        return f"| {nombre} | " + " | ".join(celdas) + " |"

    metricas = [
        ("**MAE**", lambda m: f"{m['mae']:.4f}"),
        ("**RMSE**", lambda m: f"{m['rmse']:.4f}"),
        ("**WAPE**", lambda m: f"{m['wape_pct']:.2f}%"),
        ("**R²**", lambda m: f"{m['r2']:.4f}"),
        ("**Sesgo**", lambda m: f"{m['sesgo']:+.4f}" if "sesgo" in m else "-"),
        ("**Latencia (ms/muestra)**", lambda m: f"{m['latencia_ms_muestra']:.4f}"),
    ]
    filas_comparacion = [
        "Diferencia de MAE (modelo menos champion)",
        "Diferencia media diaria de MAE (champion menos modelo) [IC 95% bootstrap]",
        "Días ganados al champion",
        "Remuestreos bootstrap en que gana al champion",
        "t pareada por día",
        "t pareada por fila (referencia)",
        "Wilcoxon por fila (referencia)",
    ]
    celdas = [_celdas_comparacion(c) for c in comparaciones]

    lineas = [
        fila("Métrica", etiquetas),
        "| :--- | " + " | ".join(":---:" for _ in etiquetas) + " |",
        *(fila(nombre, [formato(m) for m in modelos]) for nombre, formato in metricas),
        *(
            fila(nombre, ["-"] + [c[i] for c in celdas])
            for i, nombre in enumerate(filas_comparacion)
        ),
        "",
        (
            "La decisión usa el análisis por día; las pruebas por fila suponen filas "
            "independientes y sólo se reportan como referencia."
        ),
        (
            f"**Veredicto:** `{resultados['decision']['recomendacion'].upper()}` - "
            f"{resultados['decision']['justificacion']}"
        ),
    ]
    return "\n".join(lineas)


@app.command()
def main(
    modelo_a: str = typer.Option(
        str(MODELO_CHAMPION_DEFAULT),
        help="Ruta local, carpeta o URI de MLflow del modelo Champion (A).",
    ),
    modelo_b: str = typer.Option(
        str(MODELO_CHALLENGER_DEFAULT),
        help=(
            "Ruta o URI del modelo Challenger (B). Por omisión, el challenger registrado y "
            "exportado. Si es 'auto', entrena el Challenger de referencia de abril a agosto."
        ),
    ),
    incluir_referencia: bool = typer.Option(
        False,
        "--incluir-referencia/--sin-referencia",
        help="Evalúa también, sobre las mismas filas, el challenger de abril a agosto.",
    ),
    periodo: str = typer.Option(
        PERIODO_AB,
        help="Periodo de evaluación (por omisión, del 16 al 30 de septiembre de 2026).",
    ),
    max_filas: int = typer.Option(
        0,
        help="Filas máximas de evaluación; 0 = todas las filas disponibles.",
    ),
    alpha: float = typer.Option(
        0.05,
        help="Nivel de significancia estadística.",
    ),
    n_bootstrap: int = typer.Option(
        N_BOOTSTRAP_AB, min=100, help="Remuestreos bootstrap de días."
    ),
    salida: Annotated[
        Path, typer.Option(help="Carpeta para guardar el reporte JSON.")
    ] = DIR_DRIFT,
) -> None:
    """Ejecuta el A/B Testing entre Champion y Challenger sobre el periodo de evaluación."""
    logger.info("Iniciando pipeline de A/B Testing para P1...")
    logger.info(f"Modelo Champion: {modelo_a}")
    logger.info(f"Modelo Challenger: {modelo_b}")
    logger.info(f"Periodo de evaluación: {periodo}")

    # 1. Cargar datos de evaluación, con la fecha de cada fila para el análisis por día
    X, y, fechas = preparar_datos_evaluacion(periodo=periodo, max_filas=max_filas)
    logger.info(f"Evaluando sobre {len(X):,} observaciones de {periodo}.")

    # 2. Verificar, antes de cargar y predecir, que ningún modelo se entrenó con el periodo
    b_es_auto = modelo_b.lower() == "auto"
    meta_a = describir_modelo(modelo_a, "champion")
    meta_b = describir_modelo(
        modelo_b, REFERENCIA_ABR_AGO if b_es_auto else "challenger_registrado"
    )
    con_referencia = incluir_referencia and not b_es_auto
    meta_ref = describir_modelo("auto", REFERENCIA_ABR_AGO) if con_referencia else None
    try:
        verificacion = verificar_sin_traslape(
            fechas, [m for m in (meta_a, meta_b, meta_ref) if m is not None]
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    # 3. Cargar el modelo A (Champion) y cargar o entrenar el B (Challenger)
    mod_a = cargar_modelo(modelo_a)
    mod_b = entrenar_challenger_referencia() if b_es_auto else cargar_modelo(modelo_b)

    # 4. Inferencia con medición de latencia
    y_pred_a, lat_a = predecir_con_latencia(mod_a, X)
    y_pred_b, lat_b = predecir_con_latencia(mod_b, X)

    # 5. Evaluar desempeño individual
    eval_a = evaluar_desempeno(y, y_pred_a, lat_a, nombre=meta_a["nombre"], metadatos=meta_a)
    eval_b = evaluar_desempeno(y, y_pred_b, lat_b, nombre=meta_b["nombre"], metadatos=meta_b)

    # 6. Comparación estadística pareada: el análisis por día decide
    opciones = {"alpha": alpha, "fechas": fechas, "n_bootstrap": n_bootstrap}
    resultados = ejecutar_ab_testing(eval_a, eval_b, **opciones)

    # 7. Challenger de abril a agosto como referencia, sobre las mismas filas
    origen_referencia = None
    if incluir_referencia and b_es_auto:
        logger.warning("El modelo B ya es el challenger de abril a agosto; no se repite.")
    elif con_referencia:
        origen_referencia = "auto"
        y_pred_ref, lat_ref = predecir_con_latencia(entrenar_challenger_referencia(), X)
        eval_ref = evaluar_desempeno(
            y, y_pred_ref, lat_ref, nombre=meta_ref["nombre"], metadatos=meta_ref
        )
        agregar_referencia(resultados, ejecutar_ab_testing(eval_a, eval_ref, **opciones))

    resultados["modelos"] = resumen_modelos(resultados)
    resultados["metadatos"] = {
        "periodo_evaluacion": periodo,
        "n_observaciones": len(X),
        "n_dias": resultados["analisis_por_dia"]["n_dias"],
        "version_datos": leer_version_datos(),
        "modelo_a_origen": str(modelo_a),
        "modelo_b_origen": str(modelo_b),
        "referencia_origen": origen_referencia,
        "verificacion_temporal": verificacion,
        "n_bootstrap": n_bootstrap,
        "semilla": SEMILLA,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 8. Guardar artefacto JSON
    guardar_resultados_ab(resultados, carpeta_salida=salida)

    # 9. Imprimir tabla formateada
    print("\n" + "=" * 70)
    print(f"REPORTE DE A/B TESTING: P1 DEMANDA ({periodo})")
    print("=" * 70)
    print(formatear_tabla_markdown(resultados))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    app()
