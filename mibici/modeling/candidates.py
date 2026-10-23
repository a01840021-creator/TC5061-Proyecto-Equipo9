"""Entrena modelos candidatos y búsquedas anidadas para P1, P2 y P3.

El módulo complementa los baselines de ``train.py``. Cada ejecución registra
parámetros, métricas de validación/prueba, modelo con firma, versión de datos,
configuración y visualizaciones en el mismo experimento de MLflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import typer

from mibici.config import (
    BATCH_SIZE_DEFAULT,
    FECHA_FIN,
    FECHA_INICIO,
    L2_REGULARIZATION_DEFAULT,
    LEARNING_RATE_DEFAULT,
    MANIFIESTO_DATOS,
    MAX_ITER_DEFAULT,
    MAX_LEAF_NODES_DEFAULT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_CLUSTERS_DEFAULT,
    P1_MODELO_DEFAULT,
    P2_MODELO_DEFAULT,
    P3_MODELO_DEFAULT,
    PROJ_ROOT,
    SEMILLA,
)
from mibici.dataset import leer_version_datos
from mibici.features import (
    cargar_datos_procesados,
    generar_dataset_p1,
    generar_dataset_p2,
    generar_dataset_p3,
    particionar_temporalmente,
)
from mibici.modeling.artifacts import (
    grafica_mapa_p3,
    grafica_p1,
    grafica_p2,
    grafica_perfiles_p3,
    grafica_precision_recall_p2,
    grafica_residuales_p1,
)
from mibici.modeling.baselines import columnas_p3, metricas_p1, metricas_p2

app = typer.Typer(add_completion=False)
FECHA_VALIDACION = "2026-08-01"
FECHA_PRUEBA = "2026-09-01"
AUTOR = "Luis Fernando Barranco Cruz (A01797868)"

FEATURES_P1 = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "mes",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
]
FEATURES_P2 = [
    "estacion_id",
    "dpcapacity",
    "salidas_lag1",
    "llegadas_lag1",
    "saldo_lag1",
    "saldo_media_3h",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
]


def obtener_commit_git() -> str:
    """Devuelve el SHA del commit que produjo la corrida."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def etiquetas(pregunta: str, familia: str, etapa: str) -> dict[str, str]:
    """Construye etiquetas homogéneas para filtrar corridas en la interfaz."""
    return {
        "team": "09",
        "question": pregunta.upper(),
        "model_family": familia,
        "stage_of_work": etapa,
        "git_commit": obtener_commit_git(),
        "author": AUTOR,
    }


def parametros_comunes(version_datos: str) -> dict[str, str | int]:
    return {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "periodo_train": "2026-04-01/2026-07-31",
        "periodo_validation": "2026-08-01/2026-08-31",
        "periodo_test": "2026-09-01/2026-09-30",
        "version_datos": version_datos,
        "semilla": SEMILLA,
    }


def renombrar_metricas(metricas: dict[str, float], prefijo: str) -> dict[str, float]:
    """Cambia el prefijo ``test_`` producido por las funciones compartidas."""
    return {
        (f"{prefijo}_{nombre[5:]}" if nombre.startswith("test_") else f"{prefijo}_{nombre}"): valor
        for nombre, valor in metricas.items()
    }


def dividir_train_validacion(
    train_completo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserva agosto como validación sin utilizar información de septiembre."""
    corte = pd.Timestamp(FECHA_VALIDACION)
    train = train_completo[train_completo["fecha_hora"] < corte].copy()
    validacion = train_completo[train_completo["fecha_hora"] >= corte].copy()
    return train, validacion


def limpiar_x(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Selecciona variables numéricas y reemplaza valores no finitos."""
    return df[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")


def registrar_contexto_datos(
    train: pd.DataFrame,
    features: list[str],
    objetivo: str | None,
    pregunta: str,
) -> None:
    """Registra el manifiesto y metadatos de una muestra no personal del entrenamiento."""
    if MANIFIESTO_DATOS.exists():
        mlflow.log_artifact(str(MANIFIESTO_DATOS), artifact_path="data")

    columnas = features + ([objetivo] if objetivo else [])
    muestra = train[columnas].sample(n=min(5_000, len(train)), random_state=SEMILLA)
    conjunto = mlflow.data.from_pandas(
        muestra,
        targets=objetivo,
        name=f"mibici_{pregunta.lower()}_training_sample",
    )
    mlflow.log_input(conjunto, context="training")


def registrar_configuracion(configuracion: dict[str, Any], directorio: Path) -> None:
    """Guarda los parámetros efectivos y el archivo central de configuración."""
    ruta = directorio / "run_config.json"
    ruta.write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(ruta), artifact_path="config")
    mlflow.log_artifact(str(PROJ_ROOT / "mibici" / "config.py"), artifact_path="config")


def construir_hgb_regresion(configuracion: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(configuracion["learning_rate"]),
        max_iter=int(configuracion["max_iter"]),
        max_leaf_nodes=int(configuracion["max_leaf_nodes"]),
        l2_regularization=float(configuracion["l2_regularization"]),
        random_state=SEMILLA,
    )


def construir_hgb_clasificacion(
    configuracion: dict[str, Any],
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(configuracion["learning_rate"]),
        max_iter=int(configuracion["max_iter"]),
        max_leaf_nodes=int(configuracion["max_leaf_nodes"]),
        l2_regularization=float(configuracion["l2_regularization"]),
        random_state=SEMILLA,
    )


def ejecutar_p1(
    datos: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de regresión para demanda horaria."""
    train_completo, test = particionar_temporalmente(datos, fecha_corte=FECHA_PRUEBA)
    train, validacion = dividir_train_validacion(train_completo)
    x_train, y_train = limpiar_x(train, FEATURES_P1), train["demanda"].to_numpy()
    x_val, y_val = limpiar_x(validacion, FEATURES_P1), validacion["demanda"].to_numpy()
    x_test, y_test = limpiar_x(test, FEATURES_P1), test["demanda"].to_numpy()
    modelo = construir_hgb_regresion(configuracion)

    nombre = configuracion.get("run_name", "P1_candidate_hgb")
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(etiquetas("P1", "hist_gradient_boosting", "candidate"))
        parametros = {
            **parametros_comunes(version),
            **configuracion,
            "n_train": len(train),
            "n_validation": len(validacion),
            "n_test": len(test),
        }
        mlflow.log_params({k: v for k, v in parametros.items() if k != "run_name"})
        modelo.fit(x_train, y_train)
        pred_val = modelo.predict(x_val)
        pred_test = modelo.predict(x_test)
        metricas = {
            **renombrar_metricas(metricas_p1(y_val, pred_val), "val"),
            **metricas_p1(y_test, pred_test),
        }
        mlflow.log_metrics(metricas)

        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_p1(test["fecha_hora"], y_test, pred_test, temporal / "real_vs_predicho_p1.png")
            grafica_residuales_p1(y_test, pred_test, temporal / "residuales_p1.png")
            mlflow.log_artifacts(temporal, artifact_path="plots")
            registrar_configuracion(parametros, temporal)

        registrar_contexto_datos(train, FEATURES_P1, "demanda", "P1")
        ejemplo = x_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
        )
        return run.info.run_id


def ejecutar_p2(
    datos: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de clasificación para riesgo operativo."""
    train_completo, test = particionar_temporalmente(datos, fecha_corte=FECHA_PRUEBA)
    train, validacion = dividir_train_validacion(train_completo)
    x_train, y_train = limpiar_x(train, FEATURES_P2), train["riesgo"].to_numpy()
    x_val, y_val = limpiar_x(validacion, FEATURES_P2), validacion["riesgo"].to_numpy()
    x_test, y_test = limpiar_x(test, FEATURES_P2), test["riesgo"].to_numpy()
    modelo = construir_hgb_clasificacion(configuracion)
    pesos = compute_sample_weight(class_weight="balanced", y=y_train)

    nombre = configuracion.get("run_name", "P2_candidate_hgb")
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(etiquetas("P2", "hist_gradient_boosting", "candidate"))
        parametros = {
            **parametros_comunes(version),
            **configuracion,
            "class_weight": "balanced_sample_weight",
            "decision_threshold": 0.5,
            "n_train": len(train),
            "n_validation": len(validacion),
            "n_test": len(test),
        }
        mlflow.log_params({k: v for k, v in parametros.items() if k != "run_name"})
        modelo.fit(x_train, y_train, sample_weight=pesos)
        score_val = modelo.predict_proba(x_val)[:, 1]
        pred_val = (score_val >= 0.5).astype("int8")
        score_test = modelo.predict_proba(x_test)[:, 1]
        pred_test = (score_test >= 0.5).astype("int8")
        metricas = {
            **renombrar_metricas(metricas_p2(y_val, pred_val, score_val), "val"),
            **metricas_p2(y_test, pred_test, score_test),
        }
        mlflow.log_metrics(metricas)

        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_p2(y_test, pred_test, temporal / "matriz_confusion_p2.png")
            grafica_precision_recall_p2(y_test, score_test, temporal / "precision_recall_p2.png")
            mlflow.log_artifacts(temporal, artifact_path="plots")
            registrar_configuracion(parametros, temporal)

        registrar_contexto_datos(train, FEATURES_P2, "riesgo", "P2")
        ejemplo = x_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict_proba(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
        )
        return run.info.run_id


def construir_cluster(configuracion: dict[str, Any]) -> Pipeline:
    """Construye K-Means o MiniBatch K-Means detrás del mismo preprocesamiento."""
    modelo = str(configuracion["modelo"]).lower()
    comunes = {
        "n_clusters": int(configuracion["n_clusters"]),
        "random_state": SEMILLA,
        "n_init": 20,
    }
    if modelo == "kmeans":
        estimador = KMeans(**comunes)
    elif modelo == "minibatch_kmeans":
        estimador = MiniBatchKMeans(
            **comunes,
            batch_size=int(configuracion.get("batch_size", BATCH_SIZE_DEFAULT)),
        )
    else:
        raise ValueError("P3 admite 'kmeans' o 'minibatch_kmeans'.")
    return Pipeline([("scaler", StandardScaler()), ("cluster", estimador)])


def ejecutar_p3(
    perfiles: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de clustering de estaciones."""
    variables = columnas_p3(perfiles)
    x = limpiar_x(perfiles, variables)
    modelo = construir_cluster(configuracion)
    nombre = configuracion.get("run_name", "P3_candidate_cluster")

    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        familia = str(configuracion["modelo"]).lower()
        mlflow.set_tags(etiquetas("P3", familia, "candidate"))
        parametros = {
            **parametros_comunes(version),
            **configuracion,
            "periodo_clustering": "2026-04-01/2026-08-31",
            "n_estaciones": len(perfiles),
            "n_variables": len(variables),
        }
        mlflow.log_params({k: v for k, v in parametros.items() if k != "run_name"})
        grupos = modelo.fit_predict(x)
        x_escalado = modelo.named_steps["scaler"].transform(x)
        estimador = modelo.named_steps["cluster"]
        metricas = {
            "train_silhouette": float(silhouette_score(x_escalado, grupos)),
            "train_davies_bouldin": float(davies_bouldin_score(x_escalado, grupos)),
            "train_calinski_harabasz": float(calinski_harabasz_score(x_escalado, grupos)),
            "train_inercia": float(estimador.inertia_),
        }
        mlflow.log_metrics(metricas)

        resultado = perfiles.copy()
        resultado["cluster"] = grupos
        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_mapa_p3(resultado, temporal / "mapa_clusters_p3.png")
            horas = [c for c in variables if c.startswith("prop_hora_")]
            perfiles_medios = grafica_perfiles_p3(
                resultado, horas, temporal / "perfiles_clusters_p3.png"
            )
            perfiles_medios.to_csv(temporal / "perfiles_promedio_p3.csv")
            mlflow.log_artifacts(temporal, artifact_path="plots")
            registrar_configuracion(parametros, temporal)

        registrar_contexto_datos(perfiles, variables, None, "P3")
        ejemplo = x.head(5)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
        )
        return run.info.run_id


def configuraciones_busqueda(pregunta: str) -> list[dict[str, Any]]:
    """Define tres candidatos reproducibles por pregunta."""
    pregunta = pregunta.upper()
    if pregunta in {"P1", "P2"}:
        prefijo = pregunta
        return [
            {
                "modelo": "hist_gradient_boosting",
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.0,
                "run_name": f"{prefijo}_hgb_lr005_iter100_leaf31",
            },
            {
                "modelo": "hist_gradient_boosting",
                "learning_rate": 0.10,
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.1,
                "run_name": f"{prefijo}_hgb_lr010_iter150_leaf31",
            },
            {
                "modelo": "hist_gradient_boosting",
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 63,
                "l2_regularization": 1.0,
                "run_name": f"{prefijo}_hgb_lr005_iter200_leaf63",
            },
        ]
    if pregunta == "P3":
        return [
            {
                "modelo": "kmeans",
                "n_clusters": 5,
                "batch_size": BATCH_SIZE_DEFAULT,
                "run_name": "P3_kmeans_k5",
            },
            {
                "modelo": "kmeans",
                "n_clusters": 6,
                "batch_size": BATCH_SIZE_DEFAULT,
                "run_name": "P3_kmeans_k6",
            },
            {
                "modelo": "minibatch_kmeans",
                "n_clusters": 5,
                "batch_size": 128,
                "run_name": "P3_minibatch_kmeans_k5",
            },
        ]
    raise ValueError("La pregunta debe ser P1, P2 o P3.")


def preparar_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Configura el cliente para comunicarse con el servidor local de MLflow."""
    uri = os.getenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


@app.command("train")
def train(
    pregunta: str = typer.Option(..., "--pregunta", "-p"),
    modelo: str = typer.Option("auto", "--modelo"),
    learning_rate: float = typer.Option(LEARNING_RATE_DEFAULT, min=0.001),
    max_iter: int = typer.Option(MAX_ITER_DEFAULT, min=1),
    max_leaf_nodes: int = typer.Option(MAX_LEAF_NODES_DEFAULT, min=2),
    l2_regularization: float = typer.Option(L2_REGULARIZATION_DEFAULT, min=0.0),
    n_clusters: int = typer.Option(N_CLUSTERS_DEFAULT, min=2),
    batch_size: int = typer.Option(BATCH_SIZE_DEFAULT, min=10),
    umbral_riesgo: float = typer.Option(0.30, min=0.01, max=1.0),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Ejecuta una sola corrida candidata desde MLflow Projects."""
    pregunta = pregunta.upper()
    if pregunta not in {"P1", "P2", "P3"}:
        raise typer.BadParameter("Usa P1, P2 o P3.")
    preparar_tracking(tracking_uri, experiment_name)
    version = leer_version_datos()
    viajes, estaciones = cargar_datos_procesados()

    if modelo == "auto":
        modelo = {"P1": P1_MODELO_DEFAULT, "P2": P2_MODELO_DEFAULT, "P3": P3_MODELO_DEFAULT}[
            pregunta
        ]
    config = {
        "modelo": modelo,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
        "umbral_riesgo": umbral_riesgo,
        "run_name": f"{pregunta}_{modelo}_candidate",
    }
    if pregunta == "P1":
        ejecutar_p1(generar_dataset_p1(viajes, estaciones), version, config)
    elif pregunta == "P2":
        datos = generar_dataset_p2(viajes, estaciones, umbral_capacidad=umbral_riesgo)
        ejecutar_p2(datos, version, config)
    else:
        perfiles = generar_dataset_p3(viajes, estaciones, solo_train=True)
        ejecutar_p3(perfiles, version, config)


@app.command("search")
def search(
    pregunta: str = typer.Option(..., "--pregunta", "-p"),
    umbral_riesgo: float = typer.Option(0.30, min=0.01, max=1.0),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Ejecuta una búsqueda padre con tres corridas hijas para la pregunta indicada."""
    pregunta = pregunta.upper()
    if pregunta not in {"P1", "P2", "P3"}:
        raise typer.BadParameter("Usa P1, P2 o P3.")
    preparar_tracking(tracking_uri, experiment_name)
    version = leer_version_datos()
    viajes, estaciones = cargar_datos_procesados()
    if pregunta == "P1":
        datos = generar_dataset_p1(viajes, estaciones)
    elif pregunta == "P2":
        datos = generar_dataset_p2(viajes, estaciones, umbral_capacidad=umbral_riesgo)
    else:
        datos = generar_dataset_p3(viajes, estaciones, solo_train=True)

    with mlflow.start_run(run_name=f"{pregunta}_nested_search"):
        mlflow.set_tags(etiquetas(pregunta, "multiple", "search"))
        mlflow.log_params(parametros_comunes(version))
        mlflow.log_param("n_candidates", 3)
        for config in configuraciones_busqueda(pregunta):
            if pregunta == "P1":
                ejecutar_p1(datos, version, config, nested=True)
            elif pregunta == "P2":
                ejecutar_p2(datos, version, config, nested=True)
            else:
                ejecutar_p3(datos, version, config, nested=True)


if __name__ == "__main__":
    app()
