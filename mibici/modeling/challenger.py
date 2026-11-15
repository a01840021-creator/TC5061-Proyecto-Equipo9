"""Entrena y registra el challenger reentrenado de P1 con datos que ya contienen el drift.

El challenger hereda la familia, los hiperparámetros exactos y la semilla de la corrida de
origen del ``champion`` de P1 en el Model Registry; sólo cambia el periodo de entrenamiento,
que se extiende hasta ``FECHA_FIN_TRAIN_CHALLENGER`` (15 de septiembre). Se evalúa en
``PERIODO_AB`` (16 al 30 de septiembre), posterior a todo su entrenamiento y al del
champion, y se registra como versión nueva de ``mibici-equipo09-p1-demanda`` con el alias
``challenger``. El alias ``champion`` no se toca: promoverlo es una decisión aparte, que
se toma con el A/B de ``mibici/monitoring/ab_test.py``.

Con el servidor de tracking encendido, desde la raíz del repositorio::

    make modelo-challenger

Cada ejecución registra una versión nueva; para volver a exportar la misma versión basta
``make exportar-challenger``. Cada corrida guarda si el árbol de git tenía cambios sin
commit y una instantánea de este módulo con su SHA-256, para reconstruir el código exacto
aunque ``git_commit`` no lo contenga.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess  # nosec B404  # solo para git status --porcelain
import tempfile
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.models import get_model_info, infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import typer

from mibici.config import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    FECHA_FIN,
    FECHA_INICIO,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRO,
    PERIODO_AB,
    PERIODO_TRAIN_CHALLENGER,
    PROJ_ROOT,
    REPORTS_DIR,
)
from mibici.dataset import leer_version_datos
from mibici.features import cargar_datos_procesados, generar_dataset_p1
from mibici.modeling.artifacts import grafica_p1
from mibici.modeling.baselines import metricas_p1, renombrar_metricas
from mibici.modeling.candidates import (
    HIPERPARAMETROS,
    construir_regresor,
    limpiar_x,
    registrar_configuracion,
    registrar_contexto_datos,
)
from mibici.modeling.registry import registrar_version
from mibici.modeling.train import etiquetas
from mibici.monitoring.drift import FEATURES, OBJETIVO, cortar

app = typer.Typer(add_completion=False)

FAMILIA = "hist_gradient_boosting"
ETAPA = "challenger"
PREFIJO_METRICAS = "ab"
# Métrica con la que se registra la versión, en el mismo formato que registrar_version
METRICA_REGISTRO = f"{PREFIJO_METRICAS}_mae"
RUTA_INTERPRETACION = REPORTS_DIR / "drift" / "interpretacion_drift.json"
MOTIVO_POR_OMISION = (
    "Drift en septiembre en rezago_24h, rezago_48h, rezago_168h y demanda "
    "(Wasserstein normalizada > 0.1)"
)
# Carpeta de artefactos con la instantánea del código que entrena la corrida
CARPETA_CODIGO = "codigo"
# random_state de scikit-learn admite enteros de 0 a 2**32 - 1
SEMILLA_MAXIMA = 2**32 - 1


def resolver_champion(client: MlflowClient, nombre: str) -> tuple[Any, Run]:
    """Versión con alias ``champion`` y la corrida de la que proviene."""
    version = client.get_model_version_by_alias(name=nombre, alias=ALIAS_CHAMPION)
    return version, client.get_run(version.run_id)


def semilla_heredada(run: Run) -> int:
    """Semilla con que se entrenó la corrida de origen del champion, validada."""
    texto = run.data.params.get("semilla")
    if texto is None:
        raise ValueError(f"La corrida {run.info.run_id} del champion no registró semilla.")
    try:
        semilla = int(texto)
    except ValueError:
        raise ValueError(
            f"La semilla {texto!r} de la corrida {run.info.run_id} no es un entero."
        ) from None
    if not 0 <= semilla <= SEMILLA_MAXIMA:
        raise ValueError(
            f"La semilla {semilla} de la corrida {run.info.run_id} está fuera de "
            f"0 a {SEMILLA_MAXIMA}."
        )
    return semilla


def configuracion_heredada(run: Run) -> dict[str, Any]:
    """Familia, hiperparámetros exactos y semilla de la corrida de origen del champion.

    Sólo se admite HistGradientBoosting: el reentrenamiento conserva la arquitectura y
    la semilla y cambia únicamente el periodo de entrenamiento.
    """
    parametros = run.data.params
    familia = parametros.get("modelo")
    if familia != FAMILIA:
        raise ValueError(
            f"El champion proviene de la corrida {run.info.run_id} con modelo {familia!r}; "
            f"el challenger reentrenado sólo admite {FAMILIA!r}."
        )
    faltantes = [p for p in HIPERPARAMETROS[FAMILIA] if p not in parametros]
    if faltantes:
        raise ValueError(
            f"La corrida {run.info.run_id} del champion no registró {', '.join(faltantes)}."
        )
    return {
        "modelo": FAMILIA,
        **{p: parametros[p] for p in HIPERPARAMETROS[FAMILIA]},
        "semilla": semilla_heredada(run),
    }


def columnas_champion(nombre: str, version: str) -> list[str]:
    """Entradas que declara la firma del champion, en su orden."""
    firma = get_model_info(f"models:/{nombre}/{version}").signature
    if firma is None:
        raise ValueError(f"La versión {version} de {nombre} no tiene firma.")
    columnas = firma.inputs.input_names()
    desconocidas = sorted(set(columnas) - set(FEATURES))
    if desconocidas:
        raise ValueError(f"La firma del champion declara entradas desconocidas: {desconocidas}")
    return columnas


def separar_periodos(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corta entrenamiento y evaluación del challenger y verifica que no se traslapen."""
    train = cortar(datos, PERIODO_TRAIN_CHALLENGER)
    evaluacion = cortar(datos, PERIODO_AB)
    if train.empty or evaluacion.empty:
        raise ValueError(
            f"No hay filas en {PERIODO_TRAIN_CHALLENGER} o en {PERIODO_AB} para el challenger."
        )
    if train["fecha_hora"].max() >= evaluacion["fecha_hora"].min():
        raise ValueError("El entrenamiento del challenger se traslapa con la evaluación.")
    return train, evaluacion


def motivo_reentrenamiento(ruta: Path = RUTA_INTERPRETACION) -> str:
    """Resume el drift que motiva el reentrenamiento a partir de la interpretación.

    Si el archivo no existe, usa el resumen del drift de septiembre que motivó esta versión.
    """
    referencia = ruta.relative_to(PROJ_ROOT) if ruta.is_relative_to(PROJ_ROOT) else ruta
    if not ruta.is_file():
        return f"{MOTIVO_POR_OMISION}; ver {referencia.as_posix()}"
    informe = json.loads(ruta.read_text(encoding="utf-8"))
    actual = informe["interpretaciones"]["test"]
    variables = [c for c in FEATURES if c in actual["entradas_criticas_con_drift"]]
    if actual["objetivo_con_drift"]:
        variables.append(OBJETIVO)
    umbrales = {float(d["umbral"]) for d in actual["detalle_drift"].values()}
    umbral = f" (Wasserstein normalizada > {max(umbrales):g})" if umbrales else ""
    if len(variables) > 1:
        nombres = f"{', '.join(variables[:-1])} y {variables[-1]}"
    else:
        nombres = variables[0] if variables else "ninguna variable"
    return (
        f"Drift en {actual['periodo_actual']} en {nombres}{umbral}; "
        f"acción de continuous training {informe['continuous_training']['accion']}; "
        f"ver {referencia.as_posix()}"
    )


def arbol_git_con_cambios() -> str:
    """``true`` si el árbol de trabajo tiene cambios sin commit, ``false`` si no.

    Devuelve ``desconocido`` si git no está disponible o la carpeta no es un repositorio.
    """
    try:
        # Lista fija de argumentos, sin shell y sin datos de usuario; sin bloqueos
        # opcionales, para que git status no reescriba el índice
        salida = subprocess.check_output(  # nosec B603, B607  # solo lee git status
            ["git", "--no-optional-locks", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"
    return "true" if salida.strip() else "false"


def registrar_instantanea_codigo(archivo: Path = Path(__file__)) -> dict[str, str]:
    """Adjunta a la corrida activa el código que la entrena y devuelve sus etiquetas.

    Las etiquetas indican si el árbol de git tenía cambios sin commit, la ruta del
    artefacto y su SHA-256, para verificar después que el archivo no cambió.
    """
    sha256 = hashlib.sha256(archivo.read_bytes()).hexdigest()
    mlflow.log_artifact(str(archivo), artifact_path=CARPETA_CODIGO)
    return {
        "git_dirty": arbol_git_con_cambios(),
        "code_snapshot_path": f"{CARPETA_CODIGO}/{archivo.name}",
        "code_snapshot_sha256": sha256,
    }


def etiquetas_linaje(nombre: str, version_padre: Any, run_padre: Run, motivo: str) -> dict:
    """Etiquetas que ligan la corrida del challenger con el champion del que proviene."""
    return {
        "parent_model": nombre,
        "parent_version": str(version_padre.version),
        "parent_alias": ALIAS_CHAMPION,
        "parent_run_id": run_padre.info.run_id,
        "retrain_reason": motivo,
    }


def nombre_corrida(run_padre: Run) -> str:
    """Por ejemplo, ``P1_challenger_hgb_lr005_iter200_leaf63_hasta_20260915``."""
    base = run_padre.info.run_name.removeprefix("P1_")
    fin = PERIODO_TRAIN_CHALLENGER.split("/")[1].replace("-", "")
    return f"P1_{ETAPA}_{base}_hasta_{fin}"


def entrenar_y_registrar(
    datos: pd.DataFrame,
    client: MlflowClient,
    version_datos: str,
    nombre: str = MODELOS_REGISTRO["P1"],
    ruta_interpretacion: Path = RUTA_INTERPRETACION,
) -> dict[str, Any]:
    """Entrena el challenger, registra su corrida y su versión y le mueve el alias.

    Devuelve la versión creada, su corrida y sus métricas en el periodo del A/B.
    """
    version_padre, run_padre = resolver_champion(client, nombre)
    configuracion = configuracion_heredada(run_padre)
    columnas = columnas_champion(nombre, version_padre.version)
    train, evaluacion = separar_periodos(datos)
    x_train, y_train = limpiar_x(train, columnas), train[OBJETIVO].to_numpy()
    x_ab, y_ab = limpiar_x(evaluacion, columnas), evaluacion[OBJETIVO].to_numpy()
    modelo = construir_regresor(configuracion, semilla=configuracion["semilla"])
    motivo = motivo_reentrenamiento(ruta_interpretacion)

    with mlflow.start_run(run_name=nombre_corrida(run_padre)) as run:
        mlflow.set_tags(
            {
                **etiquetas("P1", FAMILIA, ETAPA),
                **etiquetas_linaje(nombre, version_padre, run_padre, motivo),
            }
        )
        parametros = {
            **configuracion,
            "fecha_inicio": FECHA_INICIO,
            "fecha_fin": FECHA_FIN,
            "periodo_train": PERIODO_TRAIN_CHALLENGER,
            "periodo_evaluacion": PERIODO_AB,
            "version_datos": version_datos,
            "n_train": len(train),
            "n_evaluacion": len(evaluacion),
        }
        mlflow.log_params(parametros)
        instantanea = registrar_instantanea_codigo()
        mlflow.set_tags(instantanea)
        modelo.fit(x_train, y_train)
        prediccion = modelo.predict(x_ab)
        metricas = renombrar_metricas(metricas_p1(y_ab, prediccion), PREFIJO_METRICAS)
        mlflow.log_metrics(metricas)

        with tempfile.TemporaryDirectory() as temporal:
            temporal = Path(temporal)
            grafica_p1(
                evaluacion["fecha_hora"],
                y_ab,
                prediccion,
                temporal / "real_vs_predicho_p1.png",
                etiqueta="Challenger",
                titulo="P1 · Demanda horaria real frente al challenger reentrenado",
            )
            mlflow.log_artifacts(temporal, artifact_path="plots")
            registrar_configuracion(parametros, temporal)
        if ruta_interpretacion.is_file():
            mlflow.log_artifact(str(ruta_interpretacion), artifact_path="drift")

        registrar_contexto_datos(train, columnas, OBJETIVO, "P1")
        ejemplo = x_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id

    version = registrar_version(client, nombre, client.get_run(run_id), "P1", METRICA_REGISTRO)
    extra = {
        "parent_model": nombre,
        "parent_version": str(version_padre.version),
        "train_period": PERIODO_TRAIN_CHALLENGER,
        "evaluation_period": PERIODO_AB,
        **instantanea,
    }
    for clave, valor in extra.items():
        client.set_model_version_tag(nombre, version, clave, valor)
    client.update_model_version(
        name=nombre,
        version=version,
        description=(
            f"Challenger reentrenado de P1 con {FAMILIA} y los hiperparámetros de la versión "
            f"{version_padre.version} (champion, corrida {run_padre.info.run_id}), entrenado "
            f"de {PERIODO_TRAIN_CHALLENGER.replace('/', ' a ')} con datos que ya contienen el "
            f"drift. Corrida {nombre_corrida(run_padre)} ({run_id}); "
            f"{METRICA_REGISTRO} = {metricas[METRICA_REGISTRO]:.6f} en "
            f"{PERIODO_AB.replace('/', ' a ')}. Versión de datos {version_datos}. Motivo: {motivo}."
        ),
    )

    champion_antes = str(version_padre.version)
    client.set_registered_model_alias(name=nombre, alias=ALIAS_CHALLENGER, version=version)
    champion_despues = str(client.get_model_version_by_alias(nombre, ALIAS_CHAMPION).version)
    if champion_despues != champion_antes:
        raise RuntimeError(
            f"El alias {ALIAS_CHAMPION} cambió de la versión {champion_antes} a la "
            f"{champion_despues} durante el registro del challenger."
        )
    return {"version": version, "run_id": run_id, "metricas": metricas}


@app.command()
def main(
    modelo: str = typer.Option(MODELOS_REGISTRO["P1"], help="Nombre en el Model Registry."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_NAME),
) -> None:
    """Entrena, registra y asigna el alias challenger al reentrenamiento de P1."""
    uri = os.getenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    viajes, estaciones = cargar_datos_procesados()
    datos = generar_dataset_p1(viajes, estaciones)

    resultado = entrenar_y_registrar(datos, MlflowClient(uri), leer_version_datos(), modelo)
    typer.echo(
        f"{modelo}@{ALIAS_CHALLENGER}: versión {resultado['version']}, "
        f"corrida {resultado['run_id']}"
    )
    for metrica, valor in resultado["metricas"].items():
        typer.echo(f"  {metrica} = {valor:.6f}")


if __name__ == "__main__":
    app()
