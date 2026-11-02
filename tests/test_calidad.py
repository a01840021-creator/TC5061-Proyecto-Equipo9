"""Chequeos de calidad de P3 y del dataset base con datos sintéticos.

Cada verificación se prueba con datos limpios, donde no debe reportar nada, y con datos
que tienen duplicados, nulos o estaciones fuera de la nomenclatura. Las corridas
completas escriben sus reportes en tmp_path y comprueban que ningún dato personal llega
a DeepChecks ni al resumen JSON.
"""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from mibici import dataset, quality
from mibici.dataset import COLUMNAS_PERSONALES, quitar_sin_estacion
from mibici.quality import (
    componentes_deepchecks,
    contar_no_convertibles,
    contar_nulos,
    diagnosticar_estaciones,
    diagnosticar_perfiles_p3,
    diagnosticar_viajes,
    ejecutar_validacion,
    ejecutar_validacion_base,
    seleccionar_preguntas,
    suite_integridad_con_nulos,
    tipar_viajes_crudos,
    validar_dataset_base,
    validar_perfiles_p3,
    verificar_estaciones_de_viajes,
)

# Valores que no deben aparecer en ningún reporte: hacen las veces de datos personales
USUARIO_SENTINELA = "usuario-sentinela-7731"
GENERO_SENTINELA = "genero-sentinela"
ANIO_SENTINELA = "1899"


def viajes_crudos(n=6, **cambios) -> pd.DataFrame:
    """Viajes como los lee ``leer_csv``: ocho columnas de texto."""
    inicio = pd.date_range("2026-04-02 08:00:00", periods=n, freq="37min")
    datos = pd.DataFrame(
        {
            "viaje_id": [str(i) for i in range(1, n + 1)],
            "usuario_id": [f"{USUARIO_SENTINELA}-{i}" for i in range(n)],
            "genero": GENERO_SENTINELA,
            "anio_de_nacimiento": ANIO_SENTINELA,
            "inicio_del_viaje": inicio.strftime("%Y-%m-%d %H:%M:%S"),
            "fin_del_viaje": (inicio + pd.Timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M:%S"),
            "origen_id": [str(1 + i % 3) for i in range(n)],
            "destino_id": [str(1 + (i + 1) % 3) for i in range(n)],
        }
    )
    for columna, valores in cambios.items():
        datos[columna] = valores
    return datos


ESTACIONES = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Estación 1", "Estación 2", "Estación 3"],
        "latitude": [20.67, 20.68, 20.69],
        "longitude": [-103.35, -103.36, -103.37],
        "dpcapacity": [10, 12, 14],
    }
)


# Relación entre viajes y nomenclatura -------------------------------------------------


def test_relacion_limpia_no_reporta_estaciones_fuera():
    resultado = verificar_estaciones_de_viajes(tipar_viajes_crudos(viajes_crudos()), ESTACIONES)
    assert resultado == {
        "viajes_fuera_de_nomenclatura": 0,
        "viajes_con_origen_fuera": 0,
        "viajes_con_destino_fuera": 0,
        "estaciones_ausentes": {},
    }


def test_relacion_detecta_estaciones_fuera_con_la_regla_de_la_ingesta():
    viajes = tipar_viajes_crudos(
        viajes_crudos(
            origen_id=["1", "9", "1", "9", "2", "3"], destino_id=["2", "1", "8", "8", "3", "1"]
        )
    )
    resultado = verificar_estaciones_de_viajes(viajes, ESTACIONES)
    assert resultado["viajes_fuera_de_nomenclatura"] == 3
    assert resultado["viajes_con_origen_fuera"] == 2
    assert resultado["viajes_con_destino_fuera"] == 2
    assert resultado["estaciones_ausentes"] == {"8": 2, "9": 2}

    conservados, por_estacion = quitar_sin_estacion(viajes, ESTACIONES)
    assert resultado["viajes_fuera_de_nomenclatura"] == len(viajes) - len(conservados)
    assert resultado["estaciones_ausentes"] == {str(k): v for k, v in por_estacion.items()}


def test_relacion_no_cuenta_identificadores_nulos_como_fuera():
    viajes = tipar_viajes_crudos(viajes_crudos(origen_id=[None, "1", "2", "3", "1", "2"]))
    assert verificar_estaciones_de_viajes(viajes, ESTACIONES)["viajes_fuera_de_nomenclatura"] == 0


# Viajes: duplicados, nulos y tipos ----------------------------------------------------


def test_viajes_limpios_no_reportan_problemas():
    crudo = viajes_crudos()
    diagnostico = diagnosticar_viajes(tipar_viajes_crudos(crudo), ESTACIONES)
    assert diagnostico["duplicados_exactos"] == 0
    assert diagnostico["viaje_id_repetidos"] == 0
    assert diagnostico["estaciones"]["viajes_fuera_de_nomenclatura"] == 0
    assert contar_nulos(crudo) == {
        "por_columna": dict.fromkeys(quality.COLUMNAS_VIAJES_SIN_PERSONALES, 0),
        "en_columnas_personales": 0,
    }
    assert set(contar_no_convertibles(crudo).values()) == {0}


def test_duplicados_exactos_usan_las_ocho_columnas():
    crudo = viajes_crudos()
    duplicado = crudo.iloc[[0]]
    solo_cambia_usuario = crudo.iloc[[1]].assign(usuario_id="otro-usuario")
    crudo = pd.concat([crudo, duplicado, solo_cambia_usuario], ignore_index=True)
    diagnostico = diagnosticar_viajes(tipar_viajes_crudos(crudo), ESTACIONES)
    assert diagnostico["columnas_regla_duplicados"] == dataset.COLUMNAS_VIAJES
    assert diagnostico["duplicados_exactos"] == 1
    assert diagnostico["viaje_id_repetidos"] == 2


def test_duplicados_en_procesados_usan_las_cinco_columnas_que_quedan():
    procesados = tipar_viajes_crudos(viajes_crudos()).drop(columns=COLUMNAS_PERSONALES)
    procesados = pd.concat([procesados, procesados.iloc[[2]]], ignore_index=True)
    diagnostico = diagnosticar_viajes(procesados, ESTACIONES)
    assert diagnostico["columnas_regla_duplicados"] == quality.COLUMNAS_VIAJES_SIN_PERSONALES
    assert diagnostico["duplicados_exactos"] == 1


def test_contar_nulos_da_solo_un_total_de_las_columnas_personales():
    crudo = viajes_crudos(
        genero=[None, "x", None, "x", "x", "x"], destino_id=["1", None, "2", "3", "1", "2"]
    )
    conteo = contar_nulos(crudo)
    assert conteo["por_columna"]["destino_id"] == 1
    assert conteo["en_columnas_personales"] == 2
    assert not set(COLUMNAS_PERSONALES) & set(conteo["por_columna"])


def test_contar_nulos_sin_columnas_personales_omite_su_total():
    assert "en_columnas_personales" not in contar_nulos(ESTACIONES)


def test_contar_no_convertibles_detecta_lo_que_la_ingesta_rechazaria():
    crudo = viajes_crudos(
        viaje_id=["1", "2.5", "x", "4", None, "6"],
        inicio_del_viaje=[
            "2026-04-02 08:00:00",
            "02/04/2026 08:00:00",
            "2026-04-02",
            None,
            "2026-04-02 09:00:00",
            "2026-04-02 10:00:00",
        ],
    )
    conteo = contar_no_convertibles(crudo)
    assert conteo["viaje_id"] == 2
    assert conteo["inicio_del_viaje"] == 2
    assert conteo["origen_id"] == 0


def test_estaciones_limpias_y_con_problemas():
    assert diagnosticar_estaciones(ESTACIONES) == {
        "filas": 3,
        "duplicados_exactos": 0,
        "ids_repetidos": 0,
        "capacidad_cero": 0,
    }
    sucias = pd.concat([ESTACIONES, ESTACIONES.iloc[[0]]], ignore_index=True)
    sucias.loc[1, "dpcapacity"] = 0
    diagnostico = diagnosticar_estaciones(sucias)
    assert diagnostico["duplicados_exactos"] == 1
    assert diagnostico["ids_repetidos"] == 1
    assert diagnostico["capacidad_cero"] == 1


# Perfiles de P3 -----------------------------------------------------------------------


def test_perfiles_p3_limpios_no_reportan_problemas(perfiles_p3, datos_p3):
    activos = perfiles_p3[perfiles_p3["n_salidas"] > 0]
    estaciones = datos_p3[1][datos_p3[1]["id"].isin(activos["id"])]
    assert diagnosticar_perfiles_p3(activos, estaciones) == {
        "filas": 30,
        "ids_repetidos": 0,
        "nulos": 0,
        "proporciones_fuera_de_rango": 0,
        "perfiles_repetidos": 0,
        "estaciones_sin_salidas": 0,
        "perfiles_repetidos_con_salidas": 0,
        "estaciones_fuera_de_nomenclatura": 0,
        "estaciones_sin_perfil": 0,
    }


def test_perfiles_p3_reporta_estaciones_sin_salidas_como_perfiles_repetidos(perfiles_p3, datos_p3):
    diagnostico = diagnosticar_perfiles_p3(perfiles_p3, datos_p3[1])
    assert diagnostico["estaciones_sin_salidas"] == 2
    assert diagnostico["perfiles_repetidos"] == 1
    assert diagnostico["perfiles_repetidos_con_salidas"] == 0


def test_perfiles_p3_detecta_duplicados_nulos_rangos_y_estaciones_fuera(perfiles_p3, datos_p3):
    activos = perfiles_p3[perfiles_p3["n_salidas"] > 0].reset_index(drop=True)
    estaciones = datos_p3[1][datos_p3[1]["id"].isin(activos["id"])]
    sucios = pd.concat([activos, activos.iloc[[0]]], ignore_index=True)
    sucios.loc[1, "prop_hora_08"] = np.nan
    sucios.loc[2, "ratio_salidas"] = 1.5
    sucios.loc[3, "id"] = 999
    diagnostico = diagnosticar_perfiles_p3(sucios, estaciones.iloc[1:])
    assert diagnostico["ids_repetidos"] == 1
    assert diagnostico["perfiles_repetidos"] == 1
    assert diagnostico["perfiles_repetidos_con_salidas"] == 1
    assert diagnostico["nulos"] == 1
    assert diagnostico["proporciones_fuera_de_rango"] == 1
    # El 999 no está en la nomenclatura; la estación 1 se quitó de la nomenclatura
    assert diagnostico["estaciones_fuera_de_nomenclatura"] == 2
    assert diagnostico["estaciones_sin_perfil"] == 1


def test_validar_perfiles_p3_con_deepchecks(perfiles_p3, datos_p3, tmp_path):
    activos = perfiles_p3[perfiles_p3["n_salidas"] > 0].reset_index(drop=True)
    limpio = validar_perfiles_p3(activos, datos_p3[1], directorio_salida=tmp_path / "limpio")
    reporte = limpio["reportes"]["integridad_perfiles"]
    assert reporte["archivo"] == "p3/integridad_perfiles.html"
    assert (tmp_path / "limpio" / "p3" / "integridad_perfiles.html").exists()
    assert {"Percent Of Nulls", "Data Duplicates"} <= set(reporte["chequeos_aprobados"])
    assert "Feature Feature Correlation" not in reporte["chequeos_no_ejecutados"]

    sucios = pd.concat([activos, activos.iloc[:5]], ignore_index=True)
    sucios.loc[0, "prop_hora_08"] = np.nan
    sucio = validar_perfiles_p3(sucios, datos_p3[1], directorio_salida=tmp_path / "sucio")
    no_aprobados = sucio["reportes"]["integridad_perfiles"]["chequeos_no_aprobados"]
    assert {"Percent Of Nulls", "Data Duplicates"} <= set(no_aprobados)
    escrito = json.loads((tmp_path / "sucio" / "p3" / "resumen.json").read_text("utf-8"))
    assert escrito["diagnostico"]["nulos"] == 1
    assert escrito["diagnostico"]["ids_repetidos"] == 5


# Dataset base completo ----------------------------------------------------------------


def escribir_crudos(carpeta, viajes: pd.DataFrame, estaciones: pd.DataFrame):
    """Escribe los CSV con los encabezados originales de MiBici."""
    carpeta.mkdir(parents=True, exist_ok=True)
    encabezado = [
        "Viaje_Id",
        "Usuario_Id",
        "Genero",
        "Año_de_nacimiento",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    mitad = len(viajes) // 2
    rutas = []
    for mes, parte in (("2026_04", viajes.iloc[:mitad]), ("2026_05", viajes.iloc[mitad:])):
        ruta = carpeta / f"datos_abiertos_{mes}.csv"
        parte.set_axis(encabezado, axis=1).to_csv(ruta, index=False, quoting=csv.QUOTE_ALL)
        rutas.append(ruta)
    nomenclatura = carpeta / "nomenclatura_prueba.csv"
    estaciones.to_csv(nomenclatura, index=False)
    return rutas, nomenclatura


@pytest.fixture
def base_sintetica(tmp_path, monkeypatch):
    """Crudos con 2 duplicados exactos y 3 viajes fuera de la nomenclatura, y su ingesta."""
    n = 400
    crudo = viajes_crudos(n)
    crudo.loc[n - 3 :, "origen_id"] = "77"
    crudo = pd.concat([crudo, crudo.iloc[[0, 10]]], ignore_index=True)
    # Difiere sólo en usuario_id: la ingesta lo conserva
    crudo = pd.concat([crudo, crudo.iloc[[20]].assign(usuario_id="otro")], ignore_index=True)
    rutas, nomenclatura = escribir_crudos(tmp_path / "raw", crudo, ESTACIONES)

    processed = tmp_path / "processed"
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(dataset, "RUTA_VIAJES", processed / "viajes.parquet")
    monkeypatch.setattr(dataset, "RUTA_ESTACIONES", processed / "estaciones.parquet")
    monkeypatch.setattr(dataset, "URLS_VIAJES", {"2026_04": "x", "2026_05": "y"})
    monkeypatch.setattr(dataset, "URL_NOMENCLATURA", "https://datos.invalid/" + nomenclatura.name)
    escribir = dataset.escribir_manifiesto
    monkeypatch.setattr(
        dataset, "escribir_manifiesto", lambda a: escribir(a, destino=tmp_path / "m.json")
    )
    dataset.ejecutar()
    return rutas, nomenclatura, processed / "viajes.parquet", processed / "estaciones.parquet"


def test_validacion_base_detecta_en_crudos_lo_que_la_ingesta_corrige(base_sintetica, tmp_path):
    salida = tmp_path / "reportes"
    resumen = validar_dataset_base(*base_sintetica, max_filas=100, directorio_salida=salida)

    crudos = resumen["diagnostico"]["crudos"]["viajes"]
    assert crudos["filas"] == 403
    assert crudos["duplicados_exactos"] == 2
    assert crudos["estaciones"]["viajes_fuera_de_nomenclatura"] == 3
    assert crudos["estaciones"]["estaciones_ausentes"] == {"77": 3}
    assert set(crudos["valores_no_convertibles"].values()) == {0}

    procesados = resumen["diagnostico"]["procesados"]["viajes"]
    assert procesados["filas"] == 403 - 2 - 3
    # La fila que sólo difería en usuario_id se conserva; sin las columnas personales queda
    # igual a su par en las cinco columnas restantes, y el chequeo de procesados lo muestra
    assert procesados["duplicados_exactos"] == 1
    assert procesados["viaje_id_repetidos"] == 1
    assert procesados["estaciones"]["viajes_fuera_de_nomenclatura"] == 0
    assert set(procesados["nulos"]["por_columna"].values()) == {0}
    assert procesados["tipos"]["origen_id"] == "int64"

    assert resumen["filas_evaluadas"] == {
        "crudos_viajes": 100,
        "crudos_estaciones": 3,
        "procesados_viajes": 100,
        "procesados_estaciones": 3,
    }
    for nombre, reporte in resumen["reportes"].items():
        assert (salida / reporte["archivo"]).exists(), nombre
        assert "Percent Of Nulls" in reporte["chequeos_aprobados"]


def test_validacion_base_no_entrega_datos_personales_a_deepchecks_ni_al_json(
    base_sintetica, tmp_path
):
    salida = tmp_path / "reportes"
    validar_dataset_base(*base_sintetica, max_filas=500, directorio_salida=salida)
    resumen = (salida / "base" / "resumen.json").read_text("utf-8")
    for sentinela in (USUARIO_SENTINELA, GENERO_SENTINELA, ANIO_SENTINELA, "otro"):
        assert sentinela not in resumen
    for html in (salida / "base").glob("*.html"):
        contenido = html.read_text("utf-8")
        for texto in (USUARIO_SENTINELA, GENERO_SENTINELA, *COLUMNAS_PERSONALES):
            assert texto not in contenido, (html.name, texto)


def test_validacion_base_falla_si_faltan_archivos(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "RAW_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="python -m mibici.dataset"):
        ejecutar_validacion_base(directorio_salida=tmp_path / "reportes")


# P1 y P2 con DeepChecks ---------------------------------------------------------------


def test_validacion_p1_con_salida_fuera_de_reports(datos_sinteticos, tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "cargar_datos_procesados", lambda: datos_sinteticos)
    resumen = ejecutar_validacion("P1", max_filas=300, directorio_salida=tmp_path)
    assert resumen["filas_train_evaluadas"] == resumen["filas_test_evaluadas"] == 300
    assert set(resumen["reportes"]) == {
        "integridad_train",
        "integridad_test",
        "validacion_train_test",
    }
    for reporte in resumen["reportes"].values():
        assert (tmp_path / reporte["archivo"]).exists()
    # Con np.NaN repuesto, el chequeo de correlación entre variables sí se ejecuta
    integridad = resumen["reportes"]["integridad_train"]
    assert "Feature Feature Correlation" not in integridad["chequeos_no_ejecutados"]
    assert json.loads((tmp_path / "p1" / "resumen.json").read_text("utf-8")) == resumen


# Configuración y CLI ------------------------------------------------------------------


def test_componentes_deepchecks_reponen_interfaces_de_numpy():
    componentes_deepchecks()
    assert np.Inf == np.inf
    assert np.isnan(np.NaN)


def test_suite_integridad_agrega_percent_of_nulls():
    nombres = [check.name() for check in suite_integridad_con_nulos().checks.values()]
    assert "Percent Of Nulls" in nombres
    assert "Data Duplicates" in nombres


@pytest.mark.parametrize(
    ("opcion", "esperado"),
    [
        ("all", ["P1", "P2", "P3", "BASE"]),
        ("ALL", ["P1", "P2", "P3", "BASE"]),
        ("p3", ["P3"]),
        ("base", ["BASE"]),
        ("P1", ["P1"]),
    ],
)
def test_seleccionar_preguntas(opcion, esperado):
    assert seleccionar_preguntas(opcion) == esperado


def test_seleccionar_pregunta_invalida():
    with pytest.raises(typer.BadParameter):
        seleccionar_preguntas("p4")


def test_cli_rechaza_pregunta_invalida():
    resultado = CliRunner().invoke(quality.app, ["--pregunta", "p4"])
    assert resultado.exit_code == 2


def test_muestreo_devuelve_copia_completa_si_hay_menos_filas():
    datos = pd.DataFrame({"valor": range(5)}, index=range(10, 15))
    muestra = quality.muestrear_deterministicamente(datos, max_filas=10)
    assert muestra["valor"].tolist() == [0, 1, 2, 3, 4]
    assert muestra.index.tolist() == [0, 1, 2, 3, 4]
    assert datos.index.tolist() == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("max_filas", [0, -1])
def test_muestreo_rechaza_max_filas_menor_a_uno(max_filas):
    with pytest.raises(ValueError, match="max_filas"):
        quality.muestrear_deterministicamente(pd.DataFrame({"x": [1]}), max_filas)
