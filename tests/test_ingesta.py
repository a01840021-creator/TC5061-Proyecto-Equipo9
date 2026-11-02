"""Pruebas de la ingesta (mibici/dataset.py) con CSV sintéticos en tmp_path.

``ejecutar`` se corre completa: las rutas, las URLs y el manifiesto del módulo apuntan a
``tmp_path`` y ``requests.get`` falla si se llama, así que la prueba no descarga nada ni
toca ``data/`` ni ``references/manifiesto_datos.json``.
"""

import csv
import hashlib
import json

from loguru import logger
import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from mibici import dataset
from mibici.config import FECHA_FIN, FECHA_INICIO, MANIFIESTO_DATOS
from mibici.dataset import (
    COLUMNAS_PERSONALES,
    ErrorIngesta,
    a_entero,
    calcular_version_datos,
    descargar,
    escribir_manifiesto,
    leer_csv,
    leer_version_datos,
    normalizar_nombre,
    procesar_estaciones,
    procesar_viajes,
    quitar_sin_estacion,
    sha256_archivo,
    validar_columnas,
)

ENCABEZADO_VIAJES = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]
ENCABEZADO_ESTACIONES = ["id", "name", "latitude", "longitude", "dpcapacity"]
ESTACIONES = [
    [1, "Estación 1", 20.67, -103.35, 10],
    [2, "Estación 2", 20.68, -103.36, 12],
    [3, "Estación 3", 20.69, -103.37, 0],
]
COLUMNAS_PROCESADAS = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]


def viaje(viaje_id, inicio="2026-04-10 08:00:00", origen=1, destino=2, usuario=100):
    fin = (pd.Timestamp(inicio) + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    return [viaje_id, usuario, "M", 1990, inicio, fin, origen, destino]


def escribir_csv(ruta, encabezado, filas, encoding="utf-8"):
    with open(ruta, "w", newline="", encoding=encoding) as f:
        escritor = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        escritor.writerow(encabezado)
        escritor.writerows(filas)
    return ruta


@pytest.fixture
def ingesta(tmp_path, monkeypatch):
    """Prepara un entorno aislado para ``dataset.ejecutar`` con dos meses de viajes."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    meses = {
        "2026_04": "https://datos.invalid/datos_abiertos_2026_04.csv",
        "2026_05": "https://datos.invalid/datos_abiertos_2026_05.csv",
    }
    manifiesto = tmp_path / "manifiesto_datos.json"
    escribir_manifiesto_original = dataset.escribir_manifiesto

    def no_descargar(*args, **kwargs):
        raise AssertionError("La prueba no debe descargar archivos.")

    monkeypatch.setattr(dataset, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(dataset, "RUTA_VIAJES", processed / "viajes.parquet")
    monkeypatch.setattr(dataset, "RUTA_ESTACIONES", processed / "estaciones.parquet")
    monkeypatch.setattr(dataset, "URLS_VIAJES", meses)
    monkeypatch.setattr(
        dataset, "URL_NOMENCLATURA", "https://datos.invalid/nomenclatura_prueba.csv"
    )
    monkeypatch.setattr(
        dataset,
        "escribir_manifiesto",
        lambda archivos: escribir_manifiesto_original(archivos, destino=manifiesto),
    )
    monkeypatch.setattr(dataset.requests, "get", no_descargar)
    escribir_csv(raw / "nomenclatura_prueba.csv", ENCABEZADO_ESTACIONES, ESTACIONES)

    class Entorno:
        def __init__(self):
            self.raw, self.processed, self.manifiesto = raw, processed, manifiesto

        def meses(self, abril, mayo=()):
            escribir_csv(raw / "datos_abiertos_2026_04.csv", ENCABEZADO_VIAJES, abril)
            escribir_csv(raw / "datos_abiertos_2026_05.csv", ENCABEZADO_VIAJES, mayo)

        def ejecutar(self):
            dataset.ejecutar()
            return (
                pd.read_parquet(processed / "viajes.parquet"),
                pd.read_parquet(processed / "estaciones.parquet"),
            )

    return Entorno()


@pytest.fixture
def advertencias():
    mensajes = []
    identificador = logger.add(lambda m: mensajes.append(m.record["message"]), level="WARNING")
    yield mensajes
    logger.remove(identificador)


# Normalización y lectura --------------------------------------------------------------


@pytest.mark.parametrize(
    ("original", "esperado"),
    [
        ("Año_de_nacimiento", "anio_de_nacimiento"),
        ("Viaje_Id", "viaje_id"),
        ("\ufeffViaje_Id", "viaje_id"),
        ("Inicio del viaje", "inicio_del_viaje"),
        ("  Género  ", "genero"),
        ("AÑO", "anio"),
        ("Origen-Id (estación)", "origen_id_estacion"),
        ("___", ""),
    ],
)
def test_normalizar_nombre_a_snake_case_ascii(original, esperado):
    assert normalizar_nombre(original) == esperado


def test_leer_csv_normaliza_columnas_y_lee_todo_como_texto(tmp_path):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [viaje(1)])
    df = leer_csv(ruta)
    assert list(df.columns) == dataset.COLUMNAS_VIAJES
    assert (df.dtypes == object).all()


def test_leer_csv_prueba_latin1_si_utf8_falla(tmp_path, advertencias):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [viaje(1)], encoding="latin-1")
    assert "anio_de_nacimiento" in leer_csv(ruta).columns
    assert any("utf-8" in m for m in advertencias)


def test_leer_csv_con_bom(tmp_path):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [viaje(1)], encoding="utf-8-sig")
    assert leer_csv(ruta).columns[0] == "viaje_id"


# Validación y tipos -------------------------------------------------------------------


def test_validar_columnas_acepta_columnas_extra():
    validar_columnas(pd.DataFrame(columns=["a", "b", "c"]), ["a", "b"], "x.csv")


def test_validar_columnas_falla_si_falta_una_obligatoria():
    with pytest.raises(ErrorIngesta, match=r"x\.csv: faltan las columnas \['b'\]"):
        validar_columnas(pd.DataFrame(columns=["a", "c"]), ["a", "b"], "x.csv")


def test_a_entero_convierte_texto_a_int64():
    df = pd.DataFrame({"id": ["1", "20", "-3"]})
    a_entero(df, ["id"], "x.csv")
    assert df["id"].dtype == "int64"
    assert df["id"].tolist() == [1, 20, -3]


@pytest.mark.parametrize(
    ("valores", "mensaje"),
    [
        (["1", None, "3"], "tiene 1 nulos"),
        (["1", "2.5", "3"], "no es entera"),
        (["1", "abc", "3"], "no es entera"),
    ],
)
def test_a_entero_falla_con_nulos_o_no_enteros(valores, mensaje):
    with pytest.raises(ErrorIngesta, match=mensaje):
        a_entero(pd.DataFrame({"id": valores}), ["id"], "x.csv")


def test_procesar_viajes_tipa_y_conserva_las_ocho_columnas(tmp_path):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [viaje(1), viaje(2)])
    df = procesar_viajes(ruta)
    assert list(df.columns) == dataset.COLUMNAS_VIAJES
    for col in dataset.COLUMNAS_ID_VIAJES:
        assert df[col].dtype == "int64"
    for col in dataset.COLUMNAS_FECHA:
        assert df[col].dtype == "datetime64[ns]"


def test_procesar_viajes_sin_filas_devuelve_tabla_vacia_tipada(tmp_path):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [])
    df = procesar_viajes(ruta)
    assert len(df) == 0
    assert df["viaje_id"].dtype == "int64"
    assert df["inicio_del_viaje"].dtype == "datetime64[ns]"


def test_procesar_viajes_falla_si_falta_una_columna(tmp_path):
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES[:-1], [viaje(1)[:-1]])
    with pytest.raises(ErrorIngesta, match="destino_id"):
        procesar_viajes(ruta)


def test_procesar_viajes_falla_con_identificador_nulo(tmp_path):
    fila = viaje(1)
    fila[6] = ""
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [fila])
    with pytest.raises(ErrorIngesta, match="origen_id tiene 1 nulos"):
        procesar_viajes(ruta)


def test_procesar_viajes_falla_con_identificador_no_entero(tmp_path):
    fila = viaje(1)
    fila[7] = "2.5"
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [fila])
    with pytest.raises(ErrorIngesta, match="destino_id no es entera"):
        procesar_viajes(ruta)


@pytest.mark.parametrize("fecha", ["10/04/2026 08:00:00", "2026-04-10", "2026-04-10T08:00:00"])
def test_procesar_viajes_falla_con_fecha_de_otro_formato(tmp_path, fecha):
    """La fecha inválida detiene la ingesta con el ValueError de pandas, no con ErrorIngesta."""
    fila = viaje(1)
    fila[4] = fecha
    ruta = escribir_csv(tmp_path / "v.csv", ENCABEZADO_VIAJES, [fila])
    with pytest.raises(ValueError, match="doesn't match format"):
        procesar_viajes(ruta)


def test_procesar_estaciones_tipa_columnas(tmp_path):
    ruta = escribir_csv(tmp_path / "e.csv", ENCABEZADO_ESTACIONES + ["extra"], [*ESTACIONES])
    df = procesar_estaciones(ruta)
    assert list(df.columns) == ENCABEZADO_ESTACIONES
    assert df["id"].dtype == "int64" and df["dpcapacity"].dtype == "int64"
    assert df["latitude"].dtype == "float64" and df["longitude"].dtype == "float64"


def test_procesar_estaciones_falla_con_capacidad_nula(tmp_path):
    filas = [list(e) for e in ESTACIONES]
    filas[0][4] = ""
    ruta = escribir_csv(tmp_path / "e.csv", ENCABEZADO_ESTACIONES, filas)
    with pytest.raises(ErrorIngesta, match="dpcapacity tiene 1 nulos"):
        procesar_estaciones(ruta)


# Estaciones fuera de la nomenclatura --------------------------------------------------


def test_quitar_sin_estacion_descarta_origen_o_destino_ausente():
    viajes = pd.DataFrame(
        {"viaje_id": [1, 2, 3, 4], "origen_id": [1, 9, 1, 9], "destino_id": [2, 1, 8, 8]}
    )
    conservados, por_estacion = quitar_sin_estacion(viajes, pd.DataFrame({"id": [1, 2]}))
    assert conservados["viaje_id"].tolist() == [1]
    # El viaje 4 tiene las dos puntas ausentes y cuenta en cada estación
    assert por_estacion == {8: 2, 9: 2}


def test_quitar_sin_estacion_sin_ausentes_no_descarta_nada():
    viajes = pd.DataFrame({"origen_id": [1, 2], "destino_id": [2, 1]})
    conservados, por_estacion = quitar_sin_estacion(viajes, pd.DataFrame({"id": [1, 2]}))
    assert len(conservados) == 2
    assert por_estacion == {}


# Hashes, versión de datos y manifiesto ------------------------------------------------


def test_sha256_archivo_coincide_con_hashlib_en_varios_bloques(tmp_path):
    contenido = bytes(range(256)) * 10_000  # 2.4 MiB, más de un bloque de lectura
    ruta = tmp_path / "f.bin"
    ruta.write_bytes(contenido)
    assert sha256_archivo(ruta) == hashlib.sha256(contenido).hexdigest()


def test_sha256_de_archivo_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    assert sha256_archivo(ruta) == hashlib.sha256(b"").hexdigest()


def test_version_datos_depende_del_contenido_y_del_orden():
    assert calcular_version_datos(["a", "b"]) == hashlib.sha256(b"ab").hexdigest()[:12]
    assert calcular_version_datos(["a", "b"]) != calcular_version_datos(["b", "a"])
    assert calcular_version_datos([]) == hashlib.sha256(b"").hexdigest()[:12]


def test_escribir_y_leer_manifiesto(tmp_path):
    archivos = [
        {"nombre": "a.csv", "url": "https://datos.invalid/a.csv", "registros": 2, "sha256": "x"},
        {"nombre": "b.csv", "url": "https://datos.invalid/b.csv", "registros": 0, "sha256": "y"},
    ]
    destino = tmp_path / "nuevo" / "manifiesto.json"
    version = escribir_manifiesto(archivos, destino=destino)
    assert version == calcular_version_datos(["x", "y"])
    assert json.loads(destino.read_text(encoding="utf-8")) == {
        "version_datos": version,
        "archivos": archivos,
    }
    assert leer_version_datos(destino) == version


def test_leer_version_datos_sin_manifiesto_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_version_datos(tmp_path / "no_existe.json")


# Descarga -----------------------------------------------------------------------------


class RespuestaFalsa:
    def __init__(self, bloques, error=None):
        self.bloques, self.error, self.headers = bloques, error, {"Content-Length": "6"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_content(self, chunk_size):
        for bloque in self.bloques:
            if isinstance(bloque, Exception):
                raise bloque
            yield bloque


def test_descargar_no_vuelve_a_descargar_un_archivo_existente(tmp_path, monkeypatch):
    destino = tmp_path / "a.csv"
    destino.write_text("ya estaba", encoding="utf-8")
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: pytest.fail("descargó"))
    descargar("https://datos.invalid/a.csv", destino)
    assert destino.read_text(encoding="utf-8") == "ya estaba"


def test_descargar_escribe_por_bloques_y_renombra_el_parcial(tmp_path, monkeypatch):
    llamadas = []

    def get(url, **kwargs):
        llamadas.append(kwargs)
        return RespuestaFalsa([b"abc", b"def"])

    monkeypatch.setattr(dataset.requests, "get", get)
    destino = tmp_path / "sub" / "a.csv"
    descargar("https://datos.invalid/a.csv", destino)
    assert destino.read_bytes() == b"abcdef"
    assert not (tmp_path / "sub" / "a.csv.part").exists()
    assert llamadas[0]["timeout"] == dataset.TIMEOUT_DESCARGA
    assert llamadas[0]["headers"] == dataset.CABECERAS


@pytest.mark.parametrize(
    "respuesta",
    [
        RespuestaFalsa([], error=requests.HTTPError("404")),
        RespuestaFalsa([b"abc", requests.ConnectionError("corte")]),
    ],
)
def test_descargar_falla_sin_dejar_archivos(tmp_path, monkeypatch, respuesta):
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: respuesta)
    destino = tmp_path / "a.csv"
    with pytest.raises(ErrorIngesta, match="No se pudo descargar"):
        descargar("https://datos.invalid/a.csv", destino)
    assert list(tmp_path.iterdir()) == []


# Ingesta completa ---------------------------------------------------------------------


def test_ingesta_quita_datos_personales(ingesta):
    ingesta.meses([viaje(1), viaje(2, usuario=200)])
    viajes, _ = ingesta.ejecutar()
    assert list(viajes.columns) == COLUMNAS_PROCESADAS
    assert not set(COLUMNAS_PERSONALES) & set(viajes.columns)


def test_ingesta_elimina_duplicados_exactos_con_las_ocho_columnas(ingesta, capsys):
    ingesta.meses(
        abril=[
            viaje(1),
            viaje(1),  # duplicado exacto en el mismo archivo
            viaje(2, usuario=100),
            viaje(2, usuario=101),  # difiere sólo en usuario_id: no es duplicado exacto
            viaje(3),
        ],
        mayo=[viaje(3)],  # duplicado exacto entre archivos
    )
    viajes, _ = ingesta.ejecutar()
    assert viajes["viaje_id"].tolist() == [1, 2, 2, 3]
    assert "Duplicados exactos eliminados: 2" in capsys.readouterr().out


def test_ingesta_descarta_viajes_con_estacion_fuera_de_nomenclatura(ingesta, capsys):
    ingesta.meses(
        [viaje(1), viaje(2, origen=99), viaje(3, destino=98), viaje(4, origen=99, destino=98)]
    )
    viajes, estaciones = ingesta.ejecutar()
    assert viajes["viaje_id"].tolist() == [1]
    salida = capsys.readouterr().out
    assert "Viajes descartados por estación fuera de la nomenclatura: 3" in salida
    assert "Estación 98: 2 viajes" in salida
    assert "Estación 99: 2 viajes" in salida
    assert estaciones["id"].tolist() == [1, 2, 3]


def test_ingesta_tipa_viajes_y_estaciones(ingesta):
    ingesta.meses([viaje(1)])
    viajes, estaciones = ingesta.ejecutar()
    assert dict(viajes.dtypes.astype(str)) == {
        "viaje_id": "int64",
        "inicio_del_viaje": "datetime64[ns]",
        "fin_del_viaje": "datetime64[ns]",
        "origen_id": "int64",
        "destino_id": "int64",
    }
    assert dict(estaciones.dtypes.astype(str)) == {
        "id": "int64",
        "name": "object",
        "latitude": "float64",
        "longitude": "float64",
        "dpcapacity": "int64",
    }


def test_ingesta_con_un_mes_sin_filas(ingesta, capsys):
    ingesta.meses(abril=[viaje(1)], mayo=[])
    viajes, _ = ingesta.ejecutar()
    assert len(viajes) == 1
    assert "Registros crudos: 1" in capsys.readouterr().out
    manifiesto = json.loads(ingesta.manifiesto.read_text(encoding="utf-8"))
    assert [a["registros"] for a in manifiesto["archivos"]] == [1, 0, 3]


def test_ingesta_con_todos_los_archivos_sin_filas(ingesta):
    ingesta.meses(abril=[], mayo=[])
    viajes, _ = ingesta.ejecutar()
    assert len(viajes) == 0
    assert list(viajes.columns) == COLUMNAS_PROCESADAS


@pytest.mark.parametrize(
    ("inicio", "fuera"),
    [
        (f"{FECHA_INICIO} 00:00:00", False),
        (f"{FECHA_FIN} 23:59:59", False),
        ("2026-03-31 23:59:59", True),
        ("2026-10-01 00:00:00", True),
    ],
)
def test_ingesta_advierte_viajes_en_el_borde_de_la_ventana_sin_filtrarlos(
    ingesta, advertencias, inicio, fuera
):
    ingesta.meses([viaje(1), viaje(2, inicio=inicio)])
    viajes, _ = ingesta.ejecutar()
    assert len(viajes) == 2
    esperado = f"1 viajes inician fuera de {FECHA_INICIO} a {FECHA_FIN}; no se filtran."
    assert (esperado in advertencias) is fuera


def test_ingesta_escribe_manifiesto_con_hashes_de_los_crudos(ingesta):
    antes = MANIFIESTO_DATOS.read_bytes()
    ingesta.meses([viaje(1), viaje(2)], [viaje(3)])
    ingesta.ejecutar()
    manifiesto = json.loads(ingesta.manifiesto.read_text(encoding="utf-8"))
    nombres = [
        "datos_abiertos_2026_04.csv",
        "datos_abiertos_2026_05.csv",
        "nomenclatura_prueba.csv",
    ]
    assert [a["nombre"] for a in manifiesto["archivos"]] == nombres
    assert [a["registros"] for a in manifiesto["archivos"]] == [2, 1, 3]
    hashes = [sha256_archivo(ingesta.raw / n) for n in nombres]
    assert [a["sha256"] for a in manifiesto["archivos"]] == hashes
    assert manifiesto["version_datos"] == calcular_version_datos(hashes)
    # El manifiesto versionado del repositorio no cambia
    assert MANIFIESTO_DATOS.read_bytes() == antes


def test_main_termina_con_codigo_1_ante_un_error_de_ingesta(ingesta):
    escribir_csv(ingesta.raw / "datos_abiertos_2026_04.csv", ENCABEZADO_VIAJES[:-1], [])
    escribir_csv(ingesta.raw / "datos_abiertos_2026_05.csv", ENCABEZADO_VIAJES, [])
    resultado = CliRunner().invoke(dataset.app, [])
    assert resultado.exit_code == 1
    assert not (ingesta.processed / "viajes.parquet").exists()
