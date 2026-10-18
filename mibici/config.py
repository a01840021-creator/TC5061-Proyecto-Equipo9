from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

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

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
