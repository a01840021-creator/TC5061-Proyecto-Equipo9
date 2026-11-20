"""Resume resultados de Locust y genera gráficas para el reporte de operación."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OBJETIVO = "POST /predict/p1"


def _numero(valor, predeterminado=0.0) -> float:
    """Convierte valores numéricos de CSV y tolera celdas vacías."""
    convertido = pd.to_numeric(valor, errors="coerce")
    return float(predeterminado if pd.isna(convertido) else convertido)


def _porcentaje(serie: pd.Series) -> pd.Series:
    """Convierte porcentajes de Docker, como ``18.2%``, a números."""
    return pd.to_numeric(serie.astype(str).str.rstrip("%"), errors="coerce")


def _fila_objetivo(frame: pd.DataFrame, nombre: str = OBJETIVO) -> pd.Series:
    """Selecciona la operación de predicción y usa el agregado como respaldo."""
    if "Name" not in frame.columns:
        raise ValueError("El CSV de Locust no contiene la columna 'Name'.")
    coincidencias = frame.loc[frame["Name"] == nombre]
    if coincidencias.empty:
        coincidencias = frame.loc[frame["Name"] == "Aggregated"]
    if coincidencias.empty:
        raise ValueError(f"No se encontró {nombre!r} ni la fila 'Aggregated'.")
    return coincidencias.iloc[-1]


def resumir_ejecucion(
    estadisticas: Path,
    historial: Path,
    recursos: Path | None = None,
) -> dict:
    """Calcula las métricas que necesita el análisis de desempeño y FinOps."""
    stats = pd.read_csv(estadisticas)
    history = pd.read_csv(historial)
    fila = _fila_objetivo(stats)

    solicitudes = int(_numero(fila.get("Request Count")))
    fallos = int(_numero(fila.get("Failure Count")))
    resumen = {
        "endpoint": OBJETIVO,
        "solicitudes_totales": solicitudes,
        "solicitudes_exitosas": solicitudes - fallos,
        "solicitudes_fallidas": fallos,
        "tasa_error_pct": round(100 * fallos / solicitudes, 4) if solicitudes else 0.0,
        "rps_promedio": round(_numero(fila.get("Requests/s")), 4),
        "latencia_promedio_ms": round(_numero(fila.get("Average Response Time")), 3),
        "latencia_p50_ms": round(_numero(fila.get("50%")), 3),
        "latencia_p95_ms": round(_numero(fila.get("95%")), 3),
        "latencia_p99_ms": round(_numero(fila.get("99%")), 3),
    }

    if "Timestamp" in history.columns and not history.empty:
        marcas = pd.to_numeric(history["Timestamp"], errors="coerce").dropna()
        if not marcas.empty:
            resumen["duracion_segundos"] = round(float(marcas.max() - marcas.min()), 3)

    if recursos is not None and recursos.is_file():
        resource_frame = pd.read_csv(recursos)
        resource_frame["cpu_percent"] = _porcentaje(resource_frame["cpu_percent"])
        resource_frame["memory_percent"] = _porcentaje(resource_frame["memory_percent"])
        por_contenedor = {}
        for contenedor, grupo in resource_frame.groupby("container"):
            por_contenedor[str(contenedor)] = {
                "muestras": len(grupo),
                "cpu_promedio_pct": round(float(grupo["cpu_percent"].mean()), 3),
                "cpu_maximo_pct": round(float(grupo["cpu_percent"].max()), 3),
                "memoria_promedio_pct": round(float(grupo["memory_percent"].mean()), 3),
                "memoria_maxima_pct": round(float(grupo["memory_percent"].max()), 3),
            }
        resumen["recursos_por_contenedor"] = por_contenedor

    return resumen


def _historial_objetivo(frame: pd.DataFrame) -> pd.DataFrame:
    """Filtra el historial a la operación principal o al agregado disponible."""
    if "Name" not in frame.columns:
        return frame.copy()
    objetivo = frame.loc[frame["Name"] == OBJETIVO]
    if objetivo.empty:
        objetivo = frame.loc[frame["Name"] == "Aggregated"]
    return objetivo.copy()


def graficar_locust(historial: Path, destino: Path) -> Path:
    """Grafica throughput, latencias, fallos y concurrencia."""
    frame = _historial_objetivo(pd.read_csv(historial))
    if frame.empty:
        raise ValueError("El historial de Locust no contiene observaciones del endpoint.")

    tiempo = pd.to_numeric(frame["Timestamp"], errors="coerce")
    frame["segundos"] = tiempo - tiempo.min()
    figura, ejes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ejes[0, 0].plot(frame["segundos"], frame["Requests/s"], color="#0057A6")
    ejes[0, 0].set(title="Throughput", ylabel="Peticiones por segundo")

    for columna, etiqueta, color in (
        ("50%", "p50", "#6A737D"),
        ("95%", "p95", "#F28E2B"),
        ("99%", "p99", "#D62728"),
    ):
        if columna in frame:
            ejes[0, 1].plot(frame["segundos"], frame[columna], label=etiqueta, color=color)
    ejes[0, 1].set(title="Latencia por percentil", ylabel="Milisegundos")
    ejes[0, 1].legend()

    ejes[1, 0].plot(frame["segundos"], frame["Failures/s"], color="#D62728")
    ejes[1, 0].set(title="Errores", ylabel="Fallos por segundo")

    if "User Count" in frame:
        ejes[1, 1].step(frame["segundos"], frame["User Count"], color="#2CA02C")
    ejes[1, 1].set(title="Concurrencia", ylabel="Usuarios simulados")

    for eje in ejes.flat:
        eje.set_xlabel("Tiempo transcurrido (s)")
        eje.grid(alpha=0.25)
        eje.spines[["top", "right"]].set_visible(False)

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.suptitle("Prueba de carga de la API MiBici", fontsize=14, fontweight="bold")
    figura.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(figura)
    return destino


def graficar_recursos(recursos: Path, destino: Path) -> Path:
    """Grafica CPU y memoria de cada contenedor durante la carga."""
    frame = pd.read_csv(recursos)
    if frame.empty:
        raise ValueError("El archivo de recursos no contiene observaciones.")
    frame["cpu_percent"] = _porcentaje(frame["cpu_percent"])
    frame["memory_percent"] = _porcentaje(frame["memory_percent"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["segundos"] = (frame["timestamp"] - frame["timestamp"].min()).dt.total_seconds()

    figura, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for contenedor, grupo in frame.groupby("container"):
        ejes[0].plot(grupo["segundos"], grupo["cpu_percent"], label=contenedor)
        ejes[1].plot(grupo["segundos"], grupo["memory_percent"], label=contenedor)

    ejes[0].set(title="Uso de CPU", ylabel="CPU (%)")
    ejes[1].set(title="Uso de memoria", ylabel="Memoria (%)", xlabel="Tiempo (s)")
    for eje in ejes:
        eje.grid(alpha=0.25)
        eje.legend()
        eje.spines[["top", "right"]].set_visible(False)

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.suptitle(
        "Recursos de los contenedores durante la carga", fontsize=14, fontweight="bold"
    )
    figura.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(figura)
    return destino


def main() -> None:
    """CLI para producir artefactos listos para el reporte y para FinOps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("reports/load_testing/locust"))
    parser.add_argument(
        "--recursos", type=Path, default=Path("reports/load_testing/resources.csv")
    )
    parser.add_argument("--salida", type=Path, default=Path("reports/load_testing"))
    args = parser.parse_args()

    stats = args.prefix.with_name(f"{args.prefix.name}_stats.csv")
    history = args.prefix.with_name(f"{args.prefix.name}_stats_history.csv")
    args.salida.mkdir(parents=True, exist_ok=True)

    resumen = resumir_ejecucion(stats, history, args.recursos)
    resumen_path = args.salida / "load_test_summary.json"
    resumen_path.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    graficar_locust(history, args.salida / "locust_performance.png")
    if args.recursos.is_file():
        graficar_recursos(args.recursos, args.salida / "resource_usage.png")

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Artefactos guardados en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
