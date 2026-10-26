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
RUTA_CONFIG = Path(__file__).resolve()
PROJ_ROOT = RUTA_CONFIG.parents[1]
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

# Cada predicción se emite al cierre de su ventana más este retraso. Un viaje que empezó
# antes del cierre y sigue abierto a la emisión ya dura al menos esto, así que no es espurio
RETRASO_DISPONIBILIDAD_SEG = 60

# Un viaje más corto que esto que termina donde empezó es un préstamo fallido, no demanda.
# Es el mismo retraso: al emitir, el desenlace de cualquier viaje espurio ya es observable
DURACION_MINIMA_VIAJE_S = RETRASO_DISPONIBILIDAD_SEG

# Partición temporal: entrenamiento hasta el día previo al corte, prueba desde el corte
FECHA_CORTE = "2026-09-01"
FECHA_FIN_TRAIN = (date.fromisoformat(FECHA_CORTE) - timedelta(days=1)).isoformat()
PERIODO_TRAIN = f"{FECHA_INICIO}/{FECHA_FIN_TRAIN}"
PERIODO_TEST = f"{FECHA_CORTE}/{FECHA_FIN}"

# Validación: el último mes antes del corte. Los candidatos se ajustan antes de esta fecha,
# se comparan en la ventana de validación y se prueban desde el corte; los baselines de P1
# y P2, que no se entrenan, se evalúan en la misma ventana.
FECHA_VALIDACION = "2026-08-01"
FECHA_FIN_TRAIN_CANDIDATOS = (date.fromisoformat(FECHA_VALIDACION) - timedelta(days=1)).isoformat()
PERIODO_TRAIN_CANDIDATOS = f"{FECHA_INICIO}/{FECHA_FIN_TRAIN_CANDIDATOS}"
PERIODO_VALIDACION = f"{FECHA_VALIDACION}/{FECHA_FIN_TRAIN}"
# P3 agrupa con todo el periodo previo al corte; no tiene ventana de validación
PERIODO_CLUSTERING = PERIODO_TRAIN

# P2: hay riesgo si el saldo neto de las próximas horas alcanza esta fracción de la capacidad
UMBRAL_RIESGO_P2 = 0.30
HORIZONTE_P2_HORAS = 2
# Probabilidad a partir de la cual un candidato de P2 emite alerta
UMBRAL_DECISION_P2 = 0.5

# P3: agrupamiento K-Means
N_CLUSTERS_P3 = 4
N_INIT_KMEANS = 20
# Una estación es activa si tiene al menos estas salidas antes del corte; las demás quedan
# con un perfil idéntico de ceros y pueden excluirse del agrupamiento
MIN_SALIDAS_ESTACION_ACTIVA = 1
ESTACIONES_TODAS = "todas"
ESTACIONES_ACTIVAS = "activas"

# MLflow
EQUIPO = "09"
EXPERIMENTO = "mibici-equipo09"
MLFLOW_EXPERIMENT_NAME = EXPERIMENTO
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
TRACKING_URI_DEFAULT = MLFLOW_TRACKING_URI

# Hiperparámetros predeterminados de los modelos candidatos
P1_MODELO_DEFAULT = "hist_gradient_boosting"
P2_MODELO_DEFAULT = "hist_gradient_boosting"
P3_MODELO_DEFAULT = "kmeans"
LEARNING_RATE_DEFAULT = 0.05
MAX_ITER_DEFAULT = 150
MAX_LEAF_NODES_DEFAULT = 31
L2_REGULARIZATION_DEFAULT = 0.0
N_CLUSTERS_DEFAULT = 5
BATCH_SIZE_DEFAULT = 128
ALPHA_DEFAULT = 1.0
C_DEFAULT = 1.0
MAX_ITER_LOGISTICA = 1000
ESTACIONES_DEFAULT = ESTACIONES_TODAS

# Model Registry: nombre de cada modelo, métrica con la que se elige y si gana el mayor
# valor. La versión con mejor métrica recibe ALIAS_CHAMPION y la segunda ALIAS_CHALLENGER.
MODELOS_REGISTRO = {
    "P1": "mibici-equipo09-p1-demanda",
    "P2": "mibici-equipo09-p2-riesgo",
    "P3": "mibici-equipo09-p3-grupos",
}
METRICAS_SELECCION = {
    "P1": ("val_mae", False),
    "P2": ("val_average_precision", True),
    "P3": ("train_silhouette", True),
}
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"
