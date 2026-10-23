"""Creación de artefactos visuales para los runs de MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
)


def grafica_p1(
    fechas: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, destino: Path
) -> pd.DataFrame:
    """Compara demanda real y baseline, agregadas para todo el sistema por hora."""
    resumen = (
        pd.DataFrame({"fecha_hora": fechas.to_numpy(), "real": y_true, "prediccion": y_pred})
        .groupby("fecha_hora", as_index=False)
        .sum(numeric_only=True)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(resumen["fecha_hora"], resumen["real"], label="Real", linewidth=1.2)
    ax.plot(
        resumen["fecha_hora"],
        resumen["prediccion"],
        label="Rezago semanal",
        linewidth=1.1,
        alpha=0.85,
    )
    ax.set(
        title="P1 · Demanda horaria real frente al baseline",
        xlabel="Fecha",
        ylabel="Viajes iniciados",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return resumen


def grafica_p2(y_true: np.ndarray, y_pred: np.ndarray, destino: Path) -> None:
    """Guarda la matriz de confusión del riesgo aproximado de rebalanceo."""
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matriz, display_labels=["Sin riesgo", "Con riesgo"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("P2 · Matriz de confusión del baseline")
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grafica_residuales_p1(y_true: np.ndarray, y_pred: np.ndarray, destino: Path) -> None:
    """Guarda una muestra de residuales para diagnosticar sesgo y heterocedasticidad."""
    residuales = y_true - y_pred
    paso = max(len(y_true) // 20_000, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred[::paso], residuales[::paso], s=8, alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="P1 · Residuales del modelo candidato",
        xlabel="Demanda predicha",
        ylabel="Real - predicción",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grafica_precision_recall_p2(y_true: np.ndarray, score: np.ndarray, destino: Path) -> None:
    """Guarda la curva precisión-recall, apropiada para una clase desbalanceada."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=1.8)
    ax.axhline(float(np.mean(y_true)), color="gray", linestyle="--", label="Prevalencia")
    ax.set(
        title="P2 · Curva precisión-recall",
        xlabel="Recall",
        ylabel="Precisión",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grafica_mapa_p3(df: pd.DataFrame, destino: Path) -> None:
    """Ubica las estaciones y colorea el cluster asignado."""
    fig, ax = plt.subplots(figsize=(8, 7))
    puntos = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["cluster"],
        cmap="tab10",
        s=24 + df["dpcapacity"].clip(lower=0) * 0.7,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set(
        title="P3 · Grupos de estaciones por patrón de uso",
        xlabel="Longitud",
        ylabel="Latitud",
    )
    ax.grid(alpha=0.15)
    ax.legend(*puntos.legend_elements(), title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(fig)


def grafica_perfiles_p3(
    df: pd.DataFrame, variables_horarias: list[str], destino: Path
) -> pd.DataFrame:
    """Compara la curva horaria promedio de cada cluster."""
    perfiles = df.groupby("cluster")[variables_horarias].mean()
    horas = np.arange(len(variables_horarias))
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster, fila in perfiles.iterrows():
        ax.plot(horas, fila.to_numpy(), marker="o", markersize=3, label=f"Cluster {cluster}")
    ax.set(
        title="P3 · Perfil horario promedio por cluster",
        xlabel="Hora del día",
        ylabel="Proporción de salidas",
        xticks=horas,
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return perfiles
