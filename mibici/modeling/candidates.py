"""Entrena modelos candidatos y búsquedas anidadas para P1, P2 y P3.

El módulo complementa los baselines de ``train.py``. Cada ejecución registra
parámetros, métricas de validación/prueba, modelo con firma, versión de datos,
configuración y visualizaciones en el mismo experimento de MLflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
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
    ALPHA_DEFAULT,
    BATCH_SIZE_DEFAULT,
    C_DEFAULT,
    ESTACIONES_ACTIVAS,
    ESTACIONES_DEFAULT,
    ESTACIONES_TODAS,
    FECHA_CORTE,
    FECHA_VALIDACION,
    HORIZONTE_P2_HORAS,
    L2_REGULARIZATION_DEFAULT,
    LEARNING_RATE_DEFAULT,
    MANIFIESTO_DATOS,
    MAX_ITER_DEFAULT,
    MAX_ITER_LOGISTICA,
    MAX_LEAF_NODES_DEFAULT,
    MIN_SALIDAS_ESTACION_ACTIVA,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_CLUSTERS_DEFAULT,
    N_INIT_KMEANS,
    P1_MODELO_DEFAULT,
    P2_MODELO_DEFAULT,
    P3_MODELO_DEFAULT,
    PERIODO_CLUSTERING,
    PERIODO_TRAIN_CANDIDATOS,
    RUTA_CONFIG,
    SEMILLA,
    UMBRAL_DECISION_P2,
    UMBRAL_RIESGO_P2,
)
from mibici.dataset import leer_version_datos
from mibici.features import (
    cargar_datos_procesados,
    generar_dataset_p1,
    generar_dataset_p2,
    generar_dataset_p3,
    particionar_temporalmente,
    purgar_ventana_futura,
)
from mibici.modeling.artifacts import (
    grafica_mapa_p3,
    grafica_p1,
    grafica_p2,
    grafica_perfiles_p3,
    grafica_precision_recall_p2,
    grafica_residuales_p1,
)
from mibici.modeling.baselines import columnas_p3, metricas_p1, metricas_p2, renombrar_metricas
from mibici.modeling.clustering import WardCentroides
from mibici.modeling.train import etiquetas
from mibici.modeling.train import parametros_comunes as parametros_baseline

app = typer.Typer(add_completion=False)

FEATURES_P1 = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
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

# Modelos que admite cada pregunta y los hiperparámetros que se registran para cada uno
MODELOS = {
    "P1": ("hist_gradient_boosting", "ridge"),
    "P2": ("hist_gradient_boosting", "logistic_regression"),
    "P3": ("kmeans", "minibatch_kmeans", "ward"),
}
HIPERPARAMETROS = {
    "hist_gradient_boosting": ("learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"),
    "ridge": ("alpha",),
    "logistic_regression": ("C",),
    "kmeans": ("n_clusters", "estaciones"),
    "minibatch_kmeans": ("n_clusters", "batch_size", "estaciones"),
    "ward": ("n_clusters", "estaciones"),
}
NOMBRES_MODELO = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "ridge": "Ridge",
    "logistic_regression": "Regresión logística",
    "kmeans": "K-Means",
    "minibatch_kmeans": "MiniBatch K-Means",
    "ward": "Ward",
}


def parametros_comunes(version_datos: str) -> dict[str, str | int]:
    """Parámetros de los baselines, con el entrenamiento de los candidatos hasta julio."""
    return {**parametros_baseline(version_datos), "periodo_train": PERIODO_TRAIN_CANDIDATOS}


def registrar_parametros(parametros: dict[str, Any]) -> None:
    """Registra los parámetros que la corrida todavía no tiene.

    Con `mlflow run`, MLflow registra primero los parámetros del punto de entrada con el
    texto que recibió (por ejemplo, `alpha=1`) y no permite cambiar un valor ya registrado;
    volver a registrarlo como `1.0` haría fallar la corrida.
    """
    existentes = mlflow.get_run(mlflow.active_run().info.run_id).data.params
    nuevos = {k: v for k, v in parametros.items() if k != "run_name" and k not in existentes}
    mlflow.log_params(nuevos)


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
    mlflow.log_artifact(str(RUTA_CONFIG), artifact_path="config")


def construir_regresor(configuracion: dict[str, Any]):
    """HistGradientBoosting o Ridge estandarizado para la demanda de P1."""
    modelo = configuracion["modelo"]
    if modelo == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=float(configuracion["learning_rate"]),
            max_iter=int(configuracion["max_iter"]),
            max_leaf_nodes=int(configuracion["max_leaf_nodes"]),
            l2_regularization=float(configuracion["l2_regularization"]),
            random_state=SEMILLA,
        )
    if modelo == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(configuracion["alpha"]))),
            ]
        )
    raise ValueError(f"P1 admite {MODELOS['P1']}; se recibió {modelo!r}.")


def construir_clasificador(configuracion: dict[str, Any]):
    """HistGradientBoosting o regresión logística balanceada para el riesgo de P2."""
    modelo = configuracion["modelo"]
    if modelo == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(configuracion["learning_rate"]),
            max_iter=int(configuracion["max_iter"]),
            max_leaf_nodes=int(configuracion["max_leaf_nodes"]),
            l2_regularization=float(configuracion["l2_regularization"]),
            random_state=SEMILLA,
        )
    if modelo == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=float(configuracion["C"]),
                        class_weight="balanced",
                        max_iter=MAX_ITER_LOGISTICA,
                        random_state=SEMILLA,
                    ),
                ),
            ]
        )
    raise ValueError(f"P2 admite {MODELOS['P2']}; se recibió {modelo!r}.")


def ejecutar_p1(
    datos: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de regresión para demanda horaria."""
    train_completo, test = particionar_temporalmente(datos, fecha_corte=FECHA_CORTE)
    train, validacion = particionar_temporalmente(train_completo, fecha_corte=FECHA_VALIDACION)
    x_train, y_train = limpiar_x(train, FEATURES_P1), train["demanda"].to_numpy()
    x_val, y_val = limpiar_x(validacion, FEATURES_P1), validacion["demanda"].to_numpy()
    x_test, y_test = limpiar_x(test, FEATURES_P1), test["demanda"].to_numpy()
    familia = configuracion["modelo"]
    modelo = construir_regresor(configuracion)

    nombre = configuracion.get("run_name", f"P1_candidate_{familia}")
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(etiquetas("P1", familia, "candidate"))
        parametros = {
            **parametros_comunes(version),
            **configuracion,
            "n_train": len(train),
            "n_validation": len(validacion),
            "n_test": len(test),
        }
        registrar_parametros(parametros)
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
            etiqueta = NOMBRES_MODELO[familia]
            grafica_p1(
                test["fecha_hora"],
                y_test,
                pred_test,
                temporal / "real_vs_predicho_p1.png",
                etiqueta=etiqueta,
                titulo=f"P1 · Demanda horaria real frente a {etiqueta}",
            )
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


def particionar_p2(
    datos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento, validación y prueba de P2 sin fuga en ninguno de los cortes.

    El tablón ya viene purgado antes de septiembre. Aquí se purgan además las últimas
    horas de entrenamiento antes de agosto, cuya etiqueta usaría horas de validación.
    """
    train_completo, test = particionar_temporalmente(datos, fecha_corte=FECHA_CORTE)
    train, validacion = particionar_temporalmente(train_completo, fecha_corte=FECHA_VALIDACION)
    train = purgar_ventana_futura(train, FECHA_VALIDACION, HORIZONTE_P2_HORAS)
    return train, validacion, test


def ejecutar_p2(
    datos: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de clasificación para riesgo operativo."""
    train, validacion, test = particionar_p2(datos)
    x_train, y_train = limpiar_x(train, FEATURES_P2), train["riesgo"].to_numpy()
    x_val, y_val = limpiar_x(validacion, FEATURES_P2), validacion["riesgo"].to_numpy()
    x_test, y_test = limpiar_x(test, FEATURES_P2), test["riesgo"].to_numpy()
    familia = configuracion["modelo"]
    modelo = construir_clasificador(configuracion)

    nombre = configuracion.get("run_name", f"P2_candidate_{familia}")
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(etiquetas("P2", familia, "candidate"))
        parametros = {
            **parametros_comunes(version),
            **configuracion,
            "umbral_capacidad": UMBRAL_RIESGO_P2,
            "horizonte_horas": HORIZONTE_P2_HORAS,
            "class_weight": (
                "balanced_sample_weight" if familia == "hist_gradient_boosting" else "balanced"
            ),
            "decision_threshold": UMBRAL_DECISION_P2,
            "n_train": len(train),
            "n_validation": len(validacion),
            "n_test": len(test),
        }
        registrar_parametros(parametros)
        if familia == "hist_gradient_boosting":
            pesos = compute_sample_weight(class_weight="balanced", y=y_train)
            modelo.fit(x_train, y_train, sample_weight=pesos)
        else:
            modelo.fit(x_train, y_train)
        score_val = modelo.predict_proba(x_val)[:, 1]
        pred_val = (score_val >= UMBRAL_DECISION_P2).astype("int8")
        score_test = modelo.predict_proba(x_test)[:, 1]
        pred_test = (score_test >= UMBRAL_DECISION_P2).astype("int8")
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
    """Construye K-Means, MiniBatch K-Means o Ward detrás del mismo preprocesamiento."""
    modelo = str(configuracion["modelo"]).lower()
    n_clusters = int(configuracion["n_clusters"])
    if modelo == "kmeans":
        estimador = KMeans(n_clusters=n_clusters, random_state=SEMILLA, n_init=N_INIT_KMEANS)
    elif modelo == "minibatch_kmeans":
        estimador = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEMILLA,
            n_init=N_INIT_KMEANS,
            batch_size=int(configuracion.get("batch_size", BATCH_SIZE_DEFAULT)),
        )
    elif modelo == "ward":
        estimador = WardCentroides(n_clusters=n_clusters)
    else:
        raise ValueError(f"P3 admite {MODELOS['P3']}; se recibió {modelo!r}.")
    return Pipeline([("scaler", StandardScaler()), ("cluster", estimador)])


def separar_estaciones_inactivas(perfiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide los perfiles en estaciones activas e inactivas antes del corte.

    Una estación sin salidas antes del corte tiene el mismo perfil de ceros que todas las
    demás en su situación, y el agrupamiento la aparta en un grupo propio que no es un
    patrón de uso.
    """
    activa = perfiles["n_salidas"] >= MIN_SALIDAS_ESTACION_ACTIVA
    return perfiles[activa].reset_index(drop=True), perfiles[~activa].reset_index(drop=True)


def ejecutar_p3(
    perfiles: pd.DataFrame,
    version: str,
    configuracion: dict[str, Any],
    nested: bool = False,
) -> str:
    """Entrena y registra un candidato de clustering de estaciones.

    Con `estaciones=activas` se excluyen las estaciones sin salidas antes del corte. Sus
    métricas se calculan sobre otro conjunto de estaciones y no son comparables con las de
    las corridas con todas las estaciones.
    """
    estaciones = configuracion.get("estaciones", ESTACIONES_TODAS)
    if estaciones not in (ESTACIONES_TODAS, ESTACIONES_ACTIVAS):
        raise ValueError(f"estaciones debe ser {ESTACIONES_TODAS!r} o {ESTACIONES_ACTIVAS!r}.")
    excluidas = perfiles.iloc[0:0]
    if estaciones == ESTACIONES_ACTIVAS:
        perfiles, excluidas = separar_estaciones_inactivas(perfiles)

    variables = columnas_p3(perfiles)
    x = limpiar_x(perfiles, variables)
    modelo = construir_cluster(configuracion)
    familia = str(configuracion["modelo"]).lower()
    nombre = configuracion.get("run_name", f"P3_candidate_{familia}")

    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(etiquetas("P3", familia, "candidate"))
        parametros = {
            **parametros_baseline(version, con_validacion=False),
            **configuracion,
            "estaciones": estaciones,
            "periodo_clustering": PERIODO_CLUSTERING,
            "n_estaciones": len(perfiles),
            "n_estaciones_excluidas": len(excluidas),
            "estaciones_excluidas": ",".join(str(i) for i in excluidas["id"]) or "ninguna",
            "n_variables": len(variables),
        }
        registrar_parametros(parametros)
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

        if len(excluidas):
            with tempfile.TemporaryDirectory() as temporal:
                ruta = Path(temporal) / "estaciones_excluidas_p3.csv"
                excluidas[["id", "name", "dpcapacity", "n_salidas"]].to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta), artifact_path="data")

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
    """Define los candidatos reproducibles de la búsqueda de cada pregunta."""
    pregunta = pregunta.upper()
    if pregunta in {"P1", "P2"}:
        prefijo = pregunta
        hgb = [
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
        if pregunta == "P1":
            lineal = {"modelo": "ridge", "alpha": ALPHA_DEFAULT, "run_name": "P1_ridge"}
        else:
            lineal = {"modelo": "logistic_regression", "C": C_DEFAULT, "run_name": "P2_logistica"}
        return hgb + [lineal]
    if pregunta == "P3":
        return [
            {
                "modelo": "kmeans",
                "n_clusters": 5,
                "estaciones": ESTACIONES_TODAS,
                "run_name": "P3_kmeans_k5",
            },
            {
                "modelo": "kmeans",
                "n_clusters": 6,
                "estaciones": ESTACIONES_TODAS,
                "run_name": "P3_kmeans_k6",
            },
            {
                "modelo": "minibatch_kmeans",
                "n_clusters": 5,
                "batch_size": BATCH_SIZE_DEFAULT,
                "estaciones": ESTACIONES_TODAS,
                "run_name": "P3_minibatch_kmeans_k5",
            },
            {
                "modelo": "ward",
                "n_clusters": 5,
                "estaciones": ESTACIONES_TODAS,
                "run_name": "P3_ward_k5",
            },
            {
                "modelo": "kmeans",
                "n_clusters": 5,
                "estaciones": ESTACIONES_ACTIVAS,
                "run_name": "P3_kmeans_k5_activas",
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
    alpha: float = typer.Option(ALPHA_DEFAULT, "--alpha", min=0.0),
    c: float = typer.Option(C_DEFAULT, "--C", min=1e-6),
    n_clusters: int = typer.Option(N_CLUSTERS_DEFAULT, min=2),
    batch_size: int = typer.Option(BATCH_SIZE_DEFAULT, min=10),
    estaciones: str = typer.Option(ESTACIONES_DEFAULT, "--estaciones"),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Ejecuta una sola corrida candidata desde MLflow Projects."""
    pregunta = pregunta.upper()
    if pregunta not in MODELOS:
        raise typer.BadParameter("Usa P1, P2 o P3.")
    modelo = modelo.lower()
    if modelo == "auto":
        modelo = {"P1": P1_MODELO_DEFAULT, "P2": P2_MODELO_DEFAULT, "P3": P3_MODELO_DEFAULT}[
            pregunta
        ]
    if modelo not in MODELOS[pregunta]:
        raise typer.BadParameter(f"{pregunta} admite {', '.join(MODELOS[pregunta])}.")
    estaciones = estaciones.lower()
    if estaciones not in (ESTACIONES_TODAS, ESTACIONES_ACTIVAS):
        raise typer.BadParameter(f"Usa {ESTACIONES_TODAS} o {ESTACIONES_ACTIVAS}.")

    preparar_tracking(tracking_uri, experiment_name)
    version = leer_version_datos()
    viajes, estaciones_df = cargar_datos_procesados()

    valores = {
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
        "alpha": alpha,
        "C": c,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
        "estaciones": estaciones,
    }
    config = {
        "modelo": modelo,
        **{clave: valores[clave] for clave in HIPERPARAMETROS[modelo]},
        "run_name": f"{pregunta}_{modelo}_candidate",
    }
    if pregunta == "P1":
        ejecutar_p1(generar_dataset_p1(viajes, estaciones_df), version, config)
    elif pregunta == "P2":
        ejecutar_p2(generar_dataset_p2(viajes, estaciones_df), version, config)
    else:
        perfiles = generar_dataset_p3(viajes, estaciones_df, solo_train=True)
        ejecutar_p3(perfiles, version, config)


@app.command("search")
def search(
    pregunta: str = typer.Option(..., "--pregunta", "-p"),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Ejecuta una búsqueda padre con una corrida hija por candidato de la pregunta."""
    pregunta = pregunta.upper()
    if pregunta not in MODELOS:
        raise typer.BadParameter("Usa P1, P2 o P3.")
    preparar_tracking(tracking_uri, experiment_name)
    version = leer_version_datos()
    viajes, estaciones = cargar_datos_procesados()
    if pregunta == "P1":
        datos = generar_dataset_p1(viajes, estaciones)
    elif pregunta == "P2":
        datos = generar_dataset_p2(viajes, estaciones)
    else:
        datos = generar_dataset_p3(viajes, estaciones, solo_train=True)

    configuraciones = configuraciones_busqueda(pregunta)
    with mlflow.start_run(run_name=f"{pregunta}_nested_search"):
        mlflow.set_tags(etiquetas(pregunta, "multiple", "search"))
        if pregunta == "P3":
            registrar_parametros(parametros_baseline(version, con_validacion=False))
        else:
            registrar_parametros(parametros_comunes(version))
        registrar_parametros({"n_candidates": len(configuraciones)})
        for config in configuraciones:
            if pregunta == "P1":
                ejecutar_p1(datos, version, config, nested=True)
            elif pregunta == "P2":
                ejecutar_p2(datos, version, config, nested=True)
            else:
                ejecutar_p3(datos, version, config, nested=True)


if __name__ == "__main__":
    app()
