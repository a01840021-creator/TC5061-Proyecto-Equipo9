"""Ejecuta los baselines P1, P2 y P3 y registra cada corrida en MLflow."""

from __future__ import annotations

import gc
import getpass
import json
from pathlib import Path
import subprocess
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from mibici.config import (
    EXPERIMENTO,
    FECHA_CORTE,
    FECHA_FIN,
    FECHA_INICIO,
    HORIZONTE_P2_HORAS,
    N_CLUSTERS_P3,
    N_INIT_KMEANS,
    PERIODO_TEST,
    PERIODO_TRAIN,
    PROJ_ROOT,
    SEMILLA,
    TRACKING_URI_DEFAULT,
    UMBRAL_RIESGO_P2,
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
)
from mibici.modeling.baselines import (
    baseline_p1_rezago_semanal,
    baseline_p2_persistencia,
    entrenar_baseline_p3,
    metricas_p1,
    metricas_p2,
)

app = typer.Typer(add_completion=False)


def obtener_autor() -> str:
    """Nombre de Git (`user.name`) de quien ejecuta; si no hay, el usuario del sistema."""
    try:
        nombre = subprocess.check_output(
            ["git", "config", "user.name"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        nombre = ""
    return nombre or getpass.getuser()


def obtener_commit_git() -> str:
    """Devuelve el SHA del commit que produjo la corrida."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def parametros_comunes(version_datos: str) -> dict[str, str | int]:
    return {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "periodo_train": PERIODO_TRAIN,
        "periodo_test": PERIODO_TEST,
        "version_datos": version_datos,
        "semilla": SEMILLA,
    }


def etiquetas_comunes(pregunta: str) -> dict[str, str]:
    return {
        "equipo": "09",
        "pregunta": pregunta,
        "autor": obtener_autor(),
        "git_commit": obtener_commit_git(),
        "tipo_modelo": "baseline",
    }


def guardar_json(contenido: dict[str, float], destino: Path) -> None:
    destino.write_text(json.dumps(contenido, indent=2), encoding="utf-8")


def ejecutar_p1(viajes: pd.DataFrame, estaciones: pd.DataFrame, version: str) -> None:
    datos = generar_dataset_p1(viajes, estaciones)
    train, test = particionar_temporalmente(datos, fecha_corte=FECHA_CORTE)
    y_true = test["demanda"].to_numpy(dtype="float64")
    y_pred = baseline_p1_rezago_semanal(test)
    metricas = metricas_p1(y_true, y_pred)

    with mlflow.start_run(run_name="P1_baseline_rezago_semanal"):
        mlflow.set_tags(etiquetas_comunes("P1"))
        mlflow.log_params(parametros_comunes(version))
        mlflow.log_params(
            {
                "modelo": "naive_rezago_168h",
                "granularidad": "estacion-hora",
                "horizonte": "24_horas",
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(metricas)
        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            resumen = grafica_p1(test["fecha_hora"], y_true, y_pred, temporal / "demanda_p1.png")
            resumen.to_csv(temporal / "demanda_horaria_p1.csv", index=False)
            guardar_json(metricas, temporal / "metricas_p1.json")
            mlflow.log_artifacts(temporal, artifact_path="evidencias_p1")


def ejecutar_p2(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    version: str,
    umbral: float,
) -> None:
    datos = generar_dataset_p2(viajes, estaciones, umbral_capacidad=umbral)
    train, test = particionar_temporalmente(datos, fecha_corte=FECHA_CORTE)
    y_true = test["riesgo"].to_numpy(dtype="int8")
    y_pred, score = baseline_p2_persistencia(test, umbral_capacidad=umbral)
    metricas = metricas_p2(y_true, y_pred, score)

    with mlflow.start_run(run_name="P2_baseline_persistencia_saldo"):
        mlflow.set_tags(etiquetas_comunes("P2"))
        mlflow.log_params(parametros_comunes(version))
        mlflow.log_params(
            {
                "modelo": "regla_persistencia_saldo_lag1",
                "granularidad": "estacion-hora",
                "horizonte_horas": HORIZONTE_P2_HORAS,
                "umbral_capacidad": umbral,
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(metricas)
        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_p2(y_true, y_pred, temporal / "matriz_confusion_p2.png")
            resumen = test[["fecha_hora", "riesgo"]].copy()
            resumen["prediccion"] = y_pred
            resumen["fecha"] = resumen["fecha_hora"].dt.date
            resumen.groupby("fecha")[["riesgo", "prediccion"]].mean().to_csv(
                temporal / "riesgo_diario_p2.csv"
            )
            guardar_json(metricas, temporal / "metricas_p2.json")
            mlflow.log_artifacts(temporal, artifact_path="evidencias_p2")


def ejecutar_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    version: str,
    n_clusters: int,
) -> None:
    perfiles = generar_dataset_p3(viajes, estaciones, solo_train=True)
    modelo, etiquetas, variables, metricas = entrenar_baseline_p3(
        perfiles, n_clusters=n_clusters, semilla=SEMILLA
    )
    perfiles = perfiles.copy()
    perfiles["cluster"] = etiquetas

    with mlflow.start_run(run_name=f"P3_baseline_kmeans_k{n_clusters}"):
        mlflow.set_tags(etiquetas_comunes("P3"))
        mlflow.log_params(parametros_comunes(version))
        mlflow.log_params(
            {
                "modelo": "KMeans",
                "n_clusters": n_clusters,
                "n_init": N_INIT_KMEANS,
                "n_estaciones": len(perfiles),
                "n_variables": len(variables),
                "periodo_clustering": PERIODO_TRAIN,
            }
        )
        mlflow.log_metrics(metricas)
        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_mapa_p3(perfiles, temporal / "mapa_clusters_p3.png")
            perfiles_hora = [c for c in variables if c.startswith("prop_hora_")]
            centros = grafica_perfiles_p3(
                perfiles, perfiles_hora, temporal / "perfiles_clusters_p3.png"
            )
            perfiles.to_csv(temporal / "estaciones_clusters_p3.csv", index=False)
            centros.to_csv(temporal / "perfiles_promedio_p3.csv")
            guardar_json(metricas, temporal / "metricas_p3.json")
            mlflow.log_artifacts(temporal, artifact_path="evidencias_p3")

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo_p3",
            serialization_format="cloudpickle",
            input_example=perfiles[variables].head(5),
        )


@app.command()
def main(
    pregunta: str = typer.Option(
        "all", "--pregunta", "-p", help="Baseline a ejecutar: P1, P2, P3 o all."
    ),
    tracking_uri: str = typer.Option(
        TRACKING_URI_DEFAULT, help="URI del backend SQLite de MLflow."
    ),
    experiment_name: str = typer.Option(EXPERIMENTO, help="Nombre del experimento."),
    umbral_riesgo: float = typer.Option(UMBRAL_RIESGO_P2, min=0.01, max=1.0),
    n_clusters: int = typer.Option(N_CLUSTERS_P3, min=2),
) -> None:
    """Ejecuta uno o todos los baselines dentro del mismo experimento de MLflow."""
    pregunta = pregunta.upper()
    if pregunta not in {"P1", "P2", "P3", "ALL"}:
        raise typer.BadParameter("Usa P1, P2, P3 o all.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    version = leer_version_datos()
    viajes, estaciones = cargar_datos_procesados()

    if pregunta in {"P1", "ALL"}:
        typer.echo("Ejecutando P1: regresión de demanda...")
        ejecutar_p1(viajes, estaciones, version)
        gc.collect()
    if pregunta in {"P2", "ALL"}:
        typer.echo("Ejecutando P2: clasificación de riesgo...")
        ejecutar_p2(viajes, estaciones, version, umbral_riesgo)
        gc.collect()
    if pregunta in {"P3", "ALL"}:
        typer.echo("Ejecutando P3: clustering de estaciones...")
        ejecutar_p3(viajes, estaciones, version, n_clusters)

    typer.echo(f"Runs registrados en '{experiment_name}'.")
    typer.echo(f"Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    app()
