from datetime import date, timedelta
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
# Nivel INFO salvo que LOGURU_LEVEL diga otro, para que la ruta local de PROJ_ROOT,
# que se registra en DEBUG, no aparezca en las salidas de los notebooks.
# Con color salvo que LOGURU_COLORIZE lo desactive, para guardar notebooks sin códigos ANSI.
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=os.getenv("LOGURU_COLORIZE", "True").strip().lower()
        not in {"0", "false", "no", "off"},
        level=os.getenv("LOGURU_LEVEL", "INFO"),
    )
except ModuleNotFoundError:
    pass

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Datos limpios que deja la ingesta y que leen P1, P2 y P3
RUTA_VIAJES = PROCESSED_DATA_DIR / "viajes.parquet"
RUTA_ESTACIONES = PROCESSED_DATA_DIR / "estaciones.parquet"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# El manifiesto va fuera de data/ porque data/ está en el .gitignore y el manifiesto sí se versiona
MANIFIESTO_DATOS = PROJ_ROOT / "references" / "manifiesto_datos.json"

# Fuentes de datos abiertos de MiBici. Las URLs no siguen un patrón (el número de
# carpeta cambia por mes), por eso van escritas una por una.
URLS_VIAJES = {
    "2026_04": "https://www.mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026_05": "https://www.mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026_06": "https://www.mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "2026_07": "https://www.mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "2026_08": "https://www.mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "2026_09": "https://www.mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
}
URL_NOMENCLATURA = "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"

# Ventana de análisis y semilla global
FECHA_INICIO = "2026-04-01"
FECHA_FIN = "2026-09-30"
SEMILLA = 42

# Un viaje más corto que esto que termina donde empezó es un préstamo fallido, no demanda
DURACION_MINIMA_VIAJE_S = 60

# Partición temporal: entrenamiento hasta el día previo al corte, prueba desde el corte
FECHA_CORTE = "2026-09-01"
FECHA_FIN_TRAIN = (date.fromisoformat(FECHA_CORTE) - timedelta(days=1)).isoformat()
PERIODO_TRAIN = f"{FECHA_INICIO}/{FECHA_FIN_TRAIN}"
PERIODO_TEST = f"{FECHA_CORTE}/{FECHA_FIN}"

# P2: hay riesgo si el saldo neto de las próximas horas alcanza esta fracción de la capacidad
UMBRAL_RIESGO_P2 = 0.30
HORIZONTE_P2_HORAS = 2

# P3: agrupamiento K-Means
N_CLUSTERS_P3 = 4
N_INIT_KMEANS = 20

# MLflow
EXPERIMENTO = "mibici-equipo09"
TRACKING_URI_DEFAULT = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
