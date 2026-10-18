from mibici.config import FECHA_FIN, FECHA_INICIO, URLS_VIAJES
from mibici.dataset import calcular_version_datos


def test_configuracion_incluye_seis_meses():
    assert FECHA_INICIO == "2026-04-01"
    assert FECHA_FIN == "2026-09-30"
    assert list(URLS_VIAJES) == [
        "2026_04",
        "2026_05",
        "2026_06",
        "2026_07",
        "2026_08",
        "2026_09",
    ]


def test_version_datos_es_determinista():
    hashes = ["hash-abril", "hash-mayo"]
    assert calcular_version_datos(hashes) == calcular_version_datos(hashes)
    assert len(calcular_version_datos(hashes)) == 12
