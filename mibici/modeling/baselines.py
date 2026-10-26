"""Baselines y métricas para las tres preguntas del proyecto MiBici."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import N_CLUSTERS_P3, N_INIT_KMEANS, SEMILLA, UMBRAL_RIESGO_P2


def baseline_p1_rezago_semanal(df_test: pd.DataFrame) -> np.ndarray:
    """Predice la demanda con el valor observado en la misma hora de la semana previa."""
    columnas = {"demanda", "rezago_168h"}
    faltantes = columnas.difference(df_test.columns)
    if faltantes:
        raise ValueError(f"P1 no contiene las columnas requeridas: {sorted(faltantes)}")
    return df_test["rezago_168h"].to_numpy(dtype="float64")


def metricas_p1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión interpretables para la planeación de demanda."""
    denominador = max(float(np.abs(y_true).sum()), 1.0)
    return {
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(root_mean_squared_error(y_true, y_pred)),
        "test_wape": float(np.abs(y_true - y_pred).sum() / denominador),
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_bias": float(np.mean(y_pred - y_true)),
    }


def renombrar_metricas(metricas: dict[str, float], prefijo: str) -> dict[str, float]:
    """Cambia el prefijo ``test_`` de las métricas por otro, por ejemplo ``val``."""
    return {
        (f"{prefijo}_{nombre[5:]}" if nombre.startswith("test_") else f"{prefijo}_{nombre}"): valor
        for nombre, valor in metricas.items()
    }


def baseline_p2_persistencia(
    df_test: pd.DataFrame, umbral_capacidad: float = UMBRAL_RIESGO_P2
) -> tuple[np.ndarray, np.ndarray]:
    """Clasifica riesgo suponiendo que el desequilibrio de la hora previa persistirá.

    El score es el desequilibrio absoluto previo dividido entre la capacidad. La regla
    no consulta el saldo futuro y, por tanto, puede utilizarse en el momento de predicción.
    """
    columnas = {"saldo_lag1", "dpcapacity"}
    faltantes = columnas.difference(df_test.columns)
    if faltantes:
        raise ValueError(f"P2 no contiene las columnas requeridas: {sorted(faltantes)}")
    if not 0 < umbral_capacidad <= 1:
        raise ValueError("El umbral de capacidad debe estar en el intervalo (0, 1].")

    score = (df_test["saldo_lag1"].abs() / df_test["dpcapacity"].replace(0, np.nan)).fillna(0.0)
    prediccion = (score >= umbral_capacidad).astype("int8")
    return prediccion.to_numpy(), score.to_numpy(dtype="float64")


def metricas_p2(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Calcula métricas robustas ante el desbalance de la clase de riesgo."""
    return {
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_average_precision": float(average_precision_score(y_true, score)),
        "test_prevalencia_riesgo": float(np.mean(y_true)),
    }


def columnas_p3(df_perfiles: pd.DataFrame) -> list[str]:
    """Selecciona variables de patrón de uso, no identificadores ni ubicación."""
    horas = sorted(c for c in df_perfiles.columns if c.startswith("prop_hora_"))
    requeridas = horas + ["prop_fin_de_semana", "ratio_salidas"]
    faltantes = set(requeridas).difference(df_perfiles.columns)
    if faltantes or len(horas) != 24:
        raise ValueError(
            "P3 requiere las 24 proporciones horarias, prop_fin_de_semana y ratio_salidas."
        )
    return requeridas


def entrenar_baseline_p3(
    df_perfiles: pd.DataFrame,
    n_clusters: int = N_CLUSTERS_P3,
    semilla: int = SEMILLA,
) -> tuple[Pipeline, np.ndarray, list[str], dict[str, float]]:
    """Estandariza los perfiles y entrena el baseline K-Means."""
    if not 2 <= n_clusters < len(df_perfiles):
        raise ValueError("n_clusters debe ser al menos 2 y menor al número de estaciones.")

    variables = columnas_p3(df_perfiles)
    x = df_perfiles[variables].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pipeline = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, random_state=semilla, n_init=N_INIT_KMEANS),
            ),
        ]
    )
    etiquetas = pipeline.fit_predict(x)
    x_escalado = pipeline.named_steps["escalador"].transform(x)
    modelo = pipeline.named_steps["kmeans"]
    metricas = {
        "train_silhouette": float(silhouette_score(x_escalado, etiquetas)),
        "train_davies_bouldin": float(davies_bouldin_score(x_escalado, etiquetas)),
        "train_calinski_harabasz": float(calinski_harabasz_score(x_escalado, etiquetas)),
        "train_inercia": float(modelo.inertia_),
    }
    return pipeline, etiquetas, variables, metricas
