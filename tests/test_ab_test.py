"""Pruebas unitarias para el módulo de A/B testing de P1."""

from __future__ import annotations

from datetime import date, timedelta
import itertools
import json
from pathlib import Path
import pickle  # nosec B403

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from typer.testing import CliRunner

from mibici.config import (
    PERIODO_TRAIN,
    PERIODO_TRAIN_CANDIDATOS,
    PERIODO_TRAIN_CHALLENGER,
    SEMILLA,
)
from mibici.monitoring import ab_test


def fechas_por_dia(n_filas: int, n_dias: int, inicio: date = date(2026, 9, 16)) -> np.ndarray:
    """Asigna las filas a días consecutivos desde ``inicio``, en bloques iguales."""
    return np.array([inicio + timedelta(days=int(i * n_dias / n_filas)) for i in range(n_filas)])


def carpeta_exportada(carpeta: Path, alias: str, version: str, periodo_train: str | None) -> Path:
    """Carpeta con el metadata.json que escribe mibici.modeling.serving."""
    carpeta.mkdir(parents=True)
    metadatos = {
        "model_name": "mibici-equipo09-p1-demanda",
        "alias": alias,
        "version": version,
        "model_uri": f"models:/mibici-equipo09-p1-demanda@{alias}",
        "run_id": "abc123",
    }
    if periodo_train is not None:
        metadatos["periodo_train"] = periodo_train
    (carpeta / "metadata.json").write_text(json.dumps(metadatos), encoding="utf-8")
    return carpeta


def leer_json_estricto(ruta: Path) -> dict:
    """Lee un JSON y falla si contiene Infinity, -Infinity o NaN."""

    def rechazar(token: str):
        raise ValueError(f"JSON no estricto: {token}")

    return json.loads(ruta.read_text(encoding="utf-8"), parse_constant=rechazar)


class ModeloDummy:
    """Modelo simple para pruebas unitarias sin dependencia de archivos pesados."""

    def __init__(self, sesgo: float = 0.0):
        self.sesgo = sesgo

    def predict(self, X):
        # Predicción determinista basada en el número de filas
        n = len(X)
        return np.ones(n, dtype=float) * 2.0 + self.sesgo


@pytest.fixture
def datos_sinteticos():
    """Genera datos sintéticos para pruebas deterministas."""
    y_real = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    # Modelo A con predicción perfecta
    y_pred_a = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    # Modelo B con error de 1.0 en cada fila
    y_pred_b = np.array([3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    return y_real, y_pred_a, y_pred_b


def test_evaluar_desempeno_perfecto(datos_sinteticos):
    """Verifica el cálculo de métricas cuando la predicción es exacta."""
    y_real, y_pred_a, _ = datos_sinteticos
    res = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.5, nombre="modelo_perfecto")

    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["wape_pct"] == 0.0
    assert res["r2"] == 1.0
    assert res["latencia_ms_muestra"] == 0.5
    assert len(res["_errores_absolutos"]) == len(y_real)


def test_evaluar_desempeno_con_error(datos_sinteticos):
    """Verifica el cálculo de métricas con errores conocidos."""
    y_real, _, y_pred_b = datos_sinteticos
    res = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=1.2, nombre="modelo_error")

    assert res["mae"] == 1.0
    assert res["rmse"] == 1.0
    # suma errores = 5, suma real = 20 -> WAPE = 25%
    assert res["wape_pct"] == 25.0
    # el modelo B sobreestima cada fila en 1.0
    assert res["sesgo"] == 1.0
    assert res["latencia_ms_muestra"] == 1.2


def test_ejecutar_ab_testing_challenger_gana():
    """Verifica la recomendación cuando el Challenger supera al Champion con significancia."""
    # 100 muestras con error consistente del Champion (2.0) y menor del Challenger (1.0)
    n = 100
    y_real = np.zeros(n, dtype=float)
    y_pred_a = np.ones(n, dtype=float) * 2.0
    y_pred_b = np.ones(n, dtype=float) * 1.0

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.05, fechas=fechas_por_dia(n, 10))

    assert res["decision"]["recomendacion"] == "promover_challenger"
    assert res["decision"]["criterios_cumplidos"]["mejora_mae"] is True
    assert res["decision"]["criterios_cumplidos"]["ic_bootstrap_favorable"] is True
    assert res["decision"]["criterios_cumplidos"]["t_por_dia_significativa"] is True
    assert res["comparativa"]["delta_mae"] == -1.0
    # Errores constantes: la t por fila no está definida y cuenta como significativa por la
    # convención de diferencias constantes, en lugar de guardar t = inf y p = 0
    assert res["referencia_por_fila"]["es_estadisticamente_significativo"] is True
    assert res["referencia_por_fila"]["caso_degenerado_t"] == ab_test.DIFERENCIAS_CONSTANTES
    assert res["analisis_por_dia"]["dias_ganados_challenger"] == 10


def test_ejecutar_ab_testing_no_significativo():
    """Verifica que no se promueva si la diferencia no es estadísticamente significativa."""
    # Diferencia casi nula y con alta variabilidad
    rng = np.random.default_rng(42)
    n = 30
    y_real = rng.normal(10, 2, n)
    y_pred_a = y_real + rng.normal(0, 1.0, n)
    y_pred_b = y_real + rng.normal(0, 0.99, n)

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.01, fechas=fechas_por_dia(n, 5))

    # Si p > alpha, la recomendación no debe ser promover
    assert res["decision"]["recomendacion"] in {"mantener_champion", "descartar_challenger"}


def test_ejecutar_ab_testing_challenger_peor():
    """Verifica el descarte si el Challenger comete mayor error que el Champion."""
    n = 50
    y_real = np.zeros(n, dtype=float)
    y_pred_a = np.ones(n, dtype=float) * 1.0
    y_pred_b = np.ones(n, dtype=float) * 3.0

    eval_a = ab_test.evaluar_desempeno(y_real, y_pred_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, y_pred_b, latencia_ms=0.1, nombre="chall")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.05, fechas=fechas_por_dia(n, 5))

    assert res["decision"]["recomendacion"] == "descartar_challenger"
    assert res["comparativa"]["delta_mae"] > 0


def test_guardar_resultados_ab(tmp_path: Path):
    """Verifica que el artefacto JSON se guarde con todas las secciones requeridas."""
    datos = {
        "modelo_a": {"mae": 0.8},
        "modelo_b": {"mae": 0.7},
        "comparativa": {"delta_mae": -0.1},
        "analisis_por_dia": {"ic_95_bootstrap": [0.01, 0.2], "p_valor_t_dia": 0.01},
        "referencia_por_fila": {"t_stat": 2.5, "p_valor_t": 0.01},
        "decision": {"recomendacion": "promover_challenger"},
    }

    ruta = ab_test.guardar_resultados_ab(datos, carpeta_salida=tmp_path)
    assert ruta.exists()

    with open(ruta, "r", encoding="utf-8") as f:
        contenido = json.load(f)

    assert contenido["decision"]["recomendacion"] == "promover_challenger"
    assert "analisis_por_dia" in contenido
    assert "referencia_por_fila" in contenido


def resultado_tabla() -> dict:
    """Resultado mínimo de dos modelos con las secciones que muestra la tabla."""
    return {
        "modelo_a": {
            "etiqueta": "Champion (v1)",
            "mae": 0.83,
            "rmse": 1.60,
            "wape_pct": 61.4,
            "r2": 0.63,
            "latencia_ms_muestra": 0.001,
        },
        "modelo_b": {
            "etiqueta": "Challenger (v2)",
            "mae": 0.82,
            "rmse": 1.59,
            "wape_pct": 61.2,
            "r2": 0.64,
            "latencia_ms_muestra": 0.001,
        },
        "comparativa": {
            "delta_mae": -0.01,
            "delta_mae_pct": -1.2,
            "delta_rmse": -0.01,
            "delta_rmse_pct": -0.6,
            "delta_wape_pct": -0.2,
            "delta_r2": 0.01,
            "delta_latencia_ms": 0.0,
        },
        "analisis_por_dia": {
            "n_dias": 15,
            "dias_ganados_challenger": 13,
            "diferencia_mae_media": 0.01,
            "ic_95_bootstrap": [0.005, 0.015],
            "proporcion_remuestreos_gana_challenger": 0.999,
            "t_stat_dia": 4.1,
            "p_valor_t_dia": 1e-3,
        },
        "referencia_por_fila": {
            "t_stat": 4.5,
            "p_valor_t": 1e-5,
            "wilcoxon_stat": 1234.0,
            "p_valor_wilcoxon": 2e-4,
            "alpha": 0.05,
            "ic_95_diferencia": [0.005, 0.015],
        },
        "decision": {
            "recomendacion": "promover_challenger",
            "justificacion": "Mejora demostrada con significancia.",
        },
    }


def test_formatear_tabla_markdown():
    """Verifica que la salida en Markdown contenga los encabezados y métricas clave."""
    tabla = ab_test.formatear_tabla_markdown(resultado_tabla())
    assert "| Métrica | Champion (v1) | Challenger (v2) |" in tabla
    assert "PROMOVER_CHALLENGER" in tabla
    assert "MAE" in tabla
    assert "RMSE" in tabla
    assert "+0.0100 [+0.0050, +0.0150]" in tabla
    assert "13 de 15" in tabla
    assert "t = 4.10 (p = 1.00e-03)" in tabla
    assert "W = 1,234 (p = 2.00e-04)" in tabla


def test_formatear_tabla_markdown_con_tres_modelos():
    """La referencia de abril a agosto aparece como tercera columna con su comparación."""
    res = resultado_tabla()
    referencia = {k: res[k] for k in ("comparativa", "analisis_por_dia", "referencia_por_fila")}
    referencia["modelo"] = {**res["modelo_b"], "etiqueta": ab_test.ETIQUETA_REFERENCIA}
    res[ab_test.REFERENCIA_ABR_AGO] = referencia

    tabla = ab_test.formatear_tabla_markdown(res)

    encabezado = f"| Métrica | Champion (v1) | Challenger (v2) | {ab_test.ETIQUETA_REFERENCIA} |"
    assert encabezado in tabla
    assert "| Días ganados al champion | - | 13 de 15 | 13 de 15 |" in tabla


def test_cargar_modelo_archivo_y_directorio(tmp_path: Path):
    """Verifica la carga de modelos desde archivo .pkl y desde directorio con model.pkl."""
    modelo_original = ModeloDummy(sesgo=0.5)

    # 1. Archivo pickle directo
    ruta_pkl = tmp_path / "test_model.pkl"
    with open(ruta_pkl, "wb") as f:
        pickle.dump(modelo_original, f)

    m1 = ab_test.cargar_modelo(ruta_pkl)
    assert isinstance(m1, ModeloDummy)

    # 2. Directorio con model.pkl
    dir_modelo = tmp_path / "modelo_dir"
    dir_modelo.mkdir()
    with open(dir_modelo / "model.pkl", "wb") as f:
        pickle.dump(modelo_original, f)

    m2 = ab_test.cargar_modelo(dir_modelo)
    assert isinstance(m2, ModeloDummy)


def test_cargar_modelo_no_existente():
    """Verifica que se lance FileNotFoundError ante una ruta inexistente."""
    with pytest.raises(FileNotFoundError):
        ab_test.cargar_modelo("ruta/inexistente/a_modelo")


def test_predecir_con_latencia():
    """Verifica que la predicción devuelva un array float y latencia no negativa."""
    import pandas as pd

    mod = ModeloDummy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    preds, lat = ab_test.predecir_con_latencia(mod, X)

    assert len(preds) == 3
    assert lat >= 0.0


def test_ab_test_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifica que la CLI de ab_test se ejecute correctamente en entorno aislado/CI."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n_samples = 30
    X_dummy = pd.DataFrame({col: rng.standard_normal(n_samples) for col in ab_test.FEATURES_P1})
    y_dummy = np.linspace(1.0, 10.0, n_samples)

    class ModeloMock:
        def __init__(self, offset: float, modulo: int = 5):
            self.offset = offset
            self.modulo = modulo

        def predict(self, X):
            n = len(X)
            return np.linspace(1.0, 10.0, n) + self.offset + (np.arange(n) % self.modulo) * 0.03

    fechas_dummy = fechas_por_dia(n_samples, 5)
    monkeypatch.setattr(
        ab_test,
        "preparar_datos_evaluacion",
        lambda **kwargs: (X_dummy, y_dummy, fechas_dummy),
    )
    monkeypatch.setattr(
        ab_test,
        "cargar_modelo",
        lambda *args, **kwargs: ModeloMock(offset=1.0, modulo=7),
    )
    monkeypatch.setattr(
        ab_test,
        "entrenar_challenger_referencia",
        lambda **kwargs: ModeloMock(offset=0.2, modulo=3),
    )

    # El modelo A declara su entrenamiento en metadata.json, como una carpeta exportada
    modelo_a = carpeta_exportada(tmp_path / "dummy_a", "champion", "1", PERIODO_TRAIN_CANDIDATOS)

    runner = CliRunner()
    resultado = runner.invoke(
        ab_test.app,
        [
            "--modelo-a",
            str(modelo_a),
            "--modelo-b",
            "auto",
            "--max-filas",
            "30",
            "--salida",
            str(tmp_path),
        ],
    )
    assert resultado.exit_code == 0, f"Error en CLI: {resultado.output}"
    assert "REPORTE DE A/B TESTING" in resultado.output
    assert (tmp_path / "ab_test_results.json").exists()
    contenido = json.loads((tmp_path / "ab_test_results.json").read_text(encoding="utf-8"))
    assert contenido["modelo_b"]["etiqueta"] == ab_test.ETIQUETA_REFERENCIA
    assert contenido["analisis_por_dia"]["n_dias"] == 5


# Análisis por día ----------------------------------------------------------------------


def test_bootstrap_challenger_gana_todos_los_dias_da_intervalo_positivo():
    """Si el challenger gana todos los días, el intervalo bootstrap queda arriba de cero."""
    diferencias = np.random.default_rng(0).uniform(0.01, 0.05, 15)

    res = ab_test.bootstrap_dias(diferencias, n_bootstrap=5000, semilla=42)

    assert res["ic_95_bootstrap"][0] > 0
    assert res["proporcion_remuestreos_gana_challenger"] == 1.0
    assert res["diferencia_mae_media"] == pytest.approx(diferencias.mean())


def test_bootstrap_con_puro_ruido_incluye_cero():
    """Con diferencias de puro ruido, sin ganador, el intervalo contiene el cero."""
    ruido = np.random.default_rng(1).normal(0.0, 0.05, 8)
    diferencias = np.concatenate([ruido, -ruido])

    res = ab_test.bootstrap_dias(diferencias, n_bootstrap=5000, semilla=42)

    ic_inf, ic_sup = res["ic_95_bootstrap"]
    assert ic_inf < 0 < ic_sup
    assert 0.2 < res["proporcion_remuestreos_gana_challenger"] < 0.8


def test_bootstrap_se_repite_con_la_misma_semilla():
    diferencias = np.random.default_rng(2).normal(0.01, 0.05, 15)

    primero = ab_test.bootstrap_dias(diferencias, n_bootstrap=5000, semilla=42)
    segundo = ab_test.bootstrap_dias(diferencias, n_bootstrap=5000, semilla=42)
    otra_semilla = ab_test.bootstrap_dias(diferencias, n_bootstrap=5000, semilla=7)

    assert primero == segundo
    assert otra_semilla["ic_95_bootstrap"] != primero["ic_95_bootstrap"]


def test_analisis_por_dia_en_un_caso_conocido():
    """Cuatro días con diferencias de MAE conocidas: días ganados, media y t pareada."""
    filas_por_dia = 3
    diferencias = np.array([0.1, 0.2, -0.05, 0.3])
    fechas = fechas_por_dia(len(diferencias) * filas_por_dia, len(diferencias))
    err_b = np.ones(len(fechas))
    err_a = err_b + np.repeat(diferencias, filas_por_dia)

    res = ab_test.analisis_por_dia(err_a, err_b, fechas, alpha=0.05, n_bootstrap=1000)

    t_esperada = diferencias.mean() / (diferencias.std(ddof=1) / np.sqrt(len(diferencias)))
    p_esperado = 2 * stats.t.sf(abs(t_esperada), df=len(diferencias) - 1)
    assert res["n_dias"] == 4
    assert res["dias_ganados_challenger"] == 3
    assert res["diferencia_mae_media"] == pytest.approx(0.1375)
    assert res["t_stat_dia"] == pytest.approx(t_esperada, abs=1e-4)
    assert res["p_valor_t_dia"] == pytest.approx(p_esperado)
    assert [d["fecha"] for d in res["por_dia"]] == [
        "2026-09-16",
        "2026-09-17",
        "2026-09-18",
        "2026-09-19",
    ]
    assert [d["diferencia_mae"] for d in res["por_dia"]] == pytest.approx(diferencias)


def test_t_pareada_con_diferencias_constantes_queda_nula_con_su_caso():
    constantes = ab_test.DIFERENCIAS_CONSTANTES
    assert ab_test.t_pareada(np.full(5, 2.0), np.full(5, 1.0)) == (None, None, constantes)
    assert ab_test.t_pareada(np.full(5, 1.0), np.full(5, 2.0)) == (None, None, constantes)
    nulas = ab_test.DIFERENCIAS_NULAS
    assert ab_test.t_pareada(np.full(5, 1.0), np.full(5, 1.0)) == (None, None, nulas)


def test_convencion_de_significancia_en_casos_degenerados():
    assert ab_test.es_significativa(None, ab_test.DIFERENCIAS_CONSTANTES, 0.05) is True
    assert ab_test.es_significativa(None, ab_test.DIFERENCIAS_NULAS, 0.05) is False
    assert ab_test.es_significativa(0.01, None, 0.05) is True
    assert ab_test.es_significativa(0.2, None, 0.05) is False


def test_decision_usa_el_analisis_por_dia_y_no_el_por_fila():
    """Muchas filas por día dan una t por fila significativa aunque los días no lo sean."""
    rng = np.random.default_rng(3)
    efectos_diarios = np.array([0.30, -0.25, 0.35, -0.20, 0.30, -0.30])
    filas_por_dia = 2000
    n = len(efectos_diarios) * filas_por_dia
    fechas = fechas_por_dia(n, len(efectos_diarios))
    err_b = 1.0 + rng.normal(0.0, 0.01, n)
    err_a = err_b + np.repeat(efectos_diarios, filas_por_dia) + rng.normal(0.0, 0.01, n)
    y_real = np.zeros(n)

    eval_a = ab_test.evaluar_desempeno(y_real, err_a, latencia_ms=0.1, nombre="champ")
    eval_b = ab_test.evaluar_desempeno(y_real, err_b, latencia_ms=0.1, nombre="chall")
    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, alpha=0.05, fechas=fechas)

    assert res["referencia_por_fila"]["p_valor_t"] < 1e-10
    assert res["analisis_por_dia"]["p_valor_t_dia"] > 0.05
    assert res["analisis_por_dia"]["ic_95_bootstrap"][0] < 0
    assert res["decision"]["criterios_cumplidos"]["mejora_mae"] is True
    assert res["decision"]["recomendacion"] == "mantener_champion"
    assert res["decision"]["base"] == "analisis_por_dia"
    assert "independientes" in res["referencia_por_fila"]["nota"]


def test_ejecutar_ab_testing_requiere_la_fecha_de_cada_fila():
    y_real = np.zeros(10)
    eval_a = ab_test.evaluar_desempeno(y_real, np.ones(10), latencia_ms=0.1, nombre="a")
    eval_b = ab_test.evaluar_desempeno(y_real, np.ones(10), latencia_ms=0.1, nombre="b")

    with pytest.raises(ValueError, match="fecha de cada fila"):
        ab_test.ejecutar_ab_testing(eval_a, eval_b)


def test_describir_modelo_lee_metadata_json(tmp_path: Path):
    carpeta = tmp_path / "p1_challenger"
    carpeta.mkdir()
    metadatos = {
        "model_name": "mibici-equipo09-p1-demanda",
        "alias": "challenger",
        "version": "3",
        "run_id": "abc123",
    }
    (carpeta / "metadata.json").write_text(json.dumps(metadatos), encoding="utf-8")

    res = ab_test.describir_modelo(carpeta, "challenger_registrado")

    assert res["etiqueta"] == "Challenger (v3)"
    assert res["nombre"] == "challenger_v3"
    assert res["model_name"] == "mibici-equipo09-p1-demanda"
    assert res["run_id"] == "abc123"
    assert res["rol"] == "challenger_registrado"


def test_describir_modelo_auto_y_sin_metadata():
    auto = ab_test.describir_modelo("auto", ab_test.REFERENCIA_ABR_AGO)
    archivo = ab_test.describir_modelo("modelos/otro.pkl", "challenger_registrado")

    assert auto["etiqueta"] == ab_test.ETIQUETA_REFERENCIA
    assert auto["version"] is None
    assert auto["periodo_train"] == ab_test.PERIODO_TRAIN
    assert archivo["etiqueta"] == "otro"


def test_champion_empaquetado_se_describe_con_su_metadata():
    """La etiqueta del champion sale de models/p1/metadata.json, no del código."""
    res = ab_test.describir_modelo(ab_test.MODELO_CHAMPION_DEFAULT, "champion")

    assert res["etiqueta"] == "Champion (v1)"
    assert res["origen"] == "models/p1"


def test_preparar_datos_evaluacion_conserva_la_fecha(monkeypatch):
    """Corta el periodo del A/B, prepara X con limpiar_x y conserva el día de cada fila."""
    horas = pd.date_range("2026-09-10", "2026-09-30 23:00", freq="h")
    tablon = pd.DataFrame({col: np.arange(len(horas)) for col in ab_test.FEATURES_P1})
    tablon["rezago_24h"] = np.where(np.arange(len(horas)) % 5 == 0, np.inf, 1)
    tablon["fecha_hora"] = horas
    tablon["demanda"] = 1
    monkeypatch.setattr(ab_test, "cargar_tablon_p1", lambda: tablon)

    X, y, fechas = ab_test.preparar_datos_evaluacion()

    assert list(X.columns) == ab_test.FEATURES_P1
    assert set(X.dtypes) == {np.dtype("float64")}
    assert len(X) == len(y) == len(fechas)
    assert min(fechas) == date(2026, 9, 16)
    assert max(fechas) == date(2026, 9, 30)
    assert len(set(fechas)) == 15
    assert np.isfinite(X.to_numpy()).all()


def test_ab_test_cli_con_challenger_registrado_y_referencia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Por omisión compara contra models/p1_challenger y añade la referencia de abril a agosto."""
    rng = np.random.default_rng(4)
    n_samples = 60
    X_dummy = pd.DataFrame({col: rng.standard_normal(n_samples) for col in ab_test.FEATURES_P1})
    y_dummy = np.linspace(1.0, 10.0, n_samples)
    fechas_dummy = fechas_por_dia(n_samples, 6)

    class ModeloMock:
        def __init__(self, offset: float, modulo: int):
            self.offset = offset
            self.modulo = modulo

        def predict(self, X):
            n = len(X)
            return np.linspace(1.0, 10.0, n) + self.offset + (np.arange(n) % self.modulo) * 0.03

    cargados = []

    def cargar(origen):
        cargados.append(str(origen))
        return ModeloMock(offset=1.0 if str(origen).endswith("p1") else 0.5, modulo=7)

    monkeypatch.setattr(
        ab_test, "preparar_datos_evaluacion", lambda **kwargs: (X_dummy, y_dummy, fechas_dummy)
    )
    monkeypatch.setattr(ab_test, "cargar_modelo", cargar)
    monkeypatch.setattr(
        ab_test, "entrenar_challenger_referencia", lambda **kwargs: ModeloMock(0.2, 3)
    )

    resultado = CliRunner().invoke(
        ab_test.app, ["--incluir-referencia", "--n-bootstrap", "500", "--salida", str(tmp_path)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert cargados == ["models/p1", "models/p1_challenger"]
    contenido = json.loads((tmp_path / "ab_test_results.json").read_text(encoding="utf-8"))
    assert list(contenido["modelos"]) == [
        "champion",
        "challenger_registrado",
        ab_test.REFERENCIA_ABR_AGO,
    ]
    assert contenido["modelo_a"]["etiqueta"] == "Champion (v1)"
    assert contenido[ab_test.REFERENCIA_ABR_AGO]["analisis_por_dia"]["n_dias"] == 6
    assert contenido["metadatos"]["periodo_evaluacion"] == ab_test.PERIODO_AB
    assert contenido["metadatos"]["referencia_origen"] == "auto"
    assert contenido["metadatos"]["n_bootstrap"] == 500
    assert ab_test.ETIQUETA_REFERENCIA in resultado.output


# Verificación temporal ------------------------------------------------------------------


def test_rechaza_evaluar_un_modelo_exportado_desde_el_ultimo_dia_de_su_entrenamiento(
    tmp_path: Path,
):
    carpeta = carpeta_exportada(
        tmp_path / "p1_challenger", "challenger", "3", "2026-04-01/2026-09-15"
    )
    modelo = ab_test.describir_modelo(carpeta, "challenger_registrado")

    with pytest.raises(ValueError, match="evaluaría sobre sus datos de entrenamiento"):
        ab_test.verificar_sin_traslape(fechas_por_dia(30, 15, date(2026, 9, 15)), [modelo])


def test_acepta_un_modelo_exportado_desde_el_dia_siguiente_a_su_entrenamiento(tmp_path: Path):
    carpeta = carpeta_exportada(
        tmp_path / "p1_challenger", "challenger", "3", "2026-04-01/2026-09-15"
    )
    modelo = ab_test.describir_modelo(carpeta, "challenger_registrado")

    verificacion = ab_test.verificar_sin_traslape(fechas_por_dia(30, 15), [modelo])

    assert verificacion == {
        "primera_fecha_evaluada": "2026-09-16",
        "periodos_train": {"challenger_v3": "2026-04-01/2026-09-15"},
    }


def test_los_modelos_empaquetados_declaran_su_periodo_y_el_15_se_rechaza():
    """models/p1 (abril a julio) y models/p1_challenger (hasta el 15 de septiembre)."""
    champion = ab_test.describir_modelo(ab_test.MODELO_CHAMPION_DEFAULT, "champion")
    challenger = ab_test.describir_modelo(ab_test.MODELO_CHALLENGER_DEFAULT, "challenger")
    assert champion["periodo_train"] == PERIODO_TRAIN_CANDIDATOS
    assert challenger["periodo_train"] == PERIODO_TRAIN_CHALLENGER

    with pytest.raises(ValueError, match="Challenger \\(v3\\)"):
        ab_test.verificar_sin_traslape(
            fechas_por_dia(10, 2, date(2026, 9, 15)), [champion, challenger]
        )
    ab_test.verificar_sin_traslape(fechas_por_dia(10, 2), [champion, challenger])


def test_auto_se_verifica_con_el_periodo_con_que_se_entrena():
    """auto se entrena de abril a agosto: rechaza el 31 de agosto y acepta desde el 1 de septiembre."""
    auto = ab_test.describir_modelo("auto", ab_test.REFERENCIA_ABR_AGO)
    assert auto["periodo_train"] == ab_test.PERIODO_TRAIN_REFERENCIA == PERIODO_TRAIN
    fin = date.fromisoformat(PERIODO_TRAIN.split("/")[1])

    with pytest.raises(ValueError, match="evaluaría sobre sus datos de entrenamiento"):
        ab_test.verificar_sin_traslape(fechas_por_dia(10, 2, fin), [auto])
    verificacion = ab_test.verificar_sin_traslape(
        fechas_por_dia(10, 2, fin + timedelta(days=1)), [auto]
    )
    assert verificacion["primera_fecha_evaluada"] == "2026-09-01"


@pytest.mark.parametrize("origen", ["modelos/otro.pkl", "models:/modelo/1", "sin_periodo"])
def test_rechaza_un_modelo_que_no_declara_su_periodo(tmp_path: Path, origen: str):
    if origen == "sin_periodo":
        # Una exportación anterior, sin periodo_train en metadata.json
        origen = carpeta_exportada(tmp_path / "viejo", "champion", "1", None)
    modelo = ab_test.describir_modelo(origen, "champion")

    with pytest.raises(ValueError, match="no declara su periodo de entrenamiento"):
        ab_test.verificar_sin_traslape(fechas_por_dia(10, 2), [modelo])


def _cli_con_fechas(monkeypatch, inicio: date, argumentos: list[str]):
    """Ejecuta la CLI con datos simulados desde ``inicio`` y registra los modelos cargados."""
    n = 40
    X = pd.DataFrame({col: np.zeros(n) for col in ab_test.FEATURES_P1})
    y = np.linspace(1.0, 10.0, n)
    cargados = []

    def cargar(origen):
        cargados.append(str(origen))
        return ModeloDummy(sesgo=0.5)

    def entrenar(**kwargs):
        cargados.append("auto")
        return ModeloDummy(sesgo=0.3)

    monkeypatch.setattr(
        ab_test, "preparar_datos_evaluacion", lambda **kwargs: (X, y, fechas_por_dia(n, 4, inicio))
    )
    monkeypatch.setattr(ab_test, "cargar_modelo", cargar)
    monkeypatch.setattr(ab_test, "entrenar_challenger_referencia", entrenar)
    resultado = CliRunner().invoke(ab_test.app, ["--n-bootstrap", "200", *argumentos])
    return resultado, cargados


def test_cli_falla_antes_de_predecir_si_el_periodo_incluye_el_15_de_septiembre(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resultado, cargados = _cli_con_fechas(
        monkeypatch, date(2026, 9, 15), ["--incluir-referencia", "--salida", str(tmp_path)]
    )

    assert resultado.exit_code == 1
    assert "evaluaría sobre sus datos de entrenamiento" in resultado.output
    assert cargados == []
    assert not (tmp_path / "ab_test_results.json").exists()


def test_cli_acepta_desde_el_16_de_septiembre_y_guarda_la_verificacion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resultado, cargados = _cli_con_fechas(
        monkeypatch, date(2026, 9, 16), ["--incluir-referencia", "--salida", str(tmp_path)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert cargados == ["models/p1", "models/p1_challenger", "auto"]
    contenido = leer_json_estricto(tmp_path / "ab_test_results.json")
    assert contenido["metadatos"]["verificacion_temporal"] == {
        "primera_fecha_evaluada": "2026-09-16",
        "periodos_train": {
            "champion_v1": PERIODO_TRAIN_CANDIDATOS,
            "challenger_v3": PERIODO_TRAIN_CHALLENGER,
            ab_test.REFERENCIA_ABR_AGO: PERIODO_TRAIN,
        },
    }


def test_cli_con_auto_falla_sin_entrenarlo_si_el_periodo_toca_su_entrenamiento(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resultado, cargados = _cli_con_fechas(
        monkeypatch, date(2026, 8, 31), ["--modelo-b", "auto", "--salida", str(tmp_path)]
    )

    assert resultado.exit_code == 1
    assert ab_test.ETIQUETA_REFERENCIA in resultado.output
    assert cargados == []


# JSON estricto y casos degenerados --------------------------------------------------------


def test_diferencias_diarias_constantes_se_guardan_sin_infinity(tmp_path: Path):
    """Errores constantes 2 frente a 1: t nula con su caso y JSON de ida y vuelta idéntico."""
    n = 30
    y_real = np.zeros(n)
    eval_a = ab_test.evaluar_desempeno(y_real, np.full(n, 2.0), latencia_ms=0.1, nombre="a")
    eval_b = ab_test.evaluar_desempeno(y_real, np.full(n, 1.0), latencia_ms=0.1, nombre="b")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, fechas=fechas_por_dia(n, 3))
    ruta = ab_test.guardar_resultados_ab(res, carpeta_salida=tmp_path)
    leido = leer_json_estricto(ruta)

    assert leido == json.loads(json.dumps(res, allow_nan=False))
    dia = leido["analisis_por_dia"]
    assert dia["t_stat_dia"] is None and dia["p_valor_t_dia"] is None
    assert dia["caso_degenerado_t_dia"] == ab_test.DIFERENCIAS_CONSTANTES
    assert "por convención" in dia["convencion_t_dia"]
    assert dia["ic_95_bootstrap"] == [1.0, 1.0]
    fila = leido["referencia_por_fila"]
    assert fila["t_stat"] is None
    assert fila["caso_degenerado_t"] == ab_test.DIFERENCIAS_CONSTANTES
    assert fila["wilcoxon_stat"] is not None
    assert leido["decision"]["recomendacion"] == "promover_challenger"
    assert leido["decision"]["convencion"] == ab_test.CONVENCION_DECISION
    assert "no definida" in leido["decision"]["justificacion"]
    assert "no definida" in ab_test.formatear_tabla_markdown(leido)


def test_modelos_identicos_se_guardan_sin_nan(tmp_path: Path):
    n = 30
    y_real = np.linspace(0.0, 5.0, n)
    prediccion = y_real + np.linspace(-1.0, 1.0, n)
    eval_a = ab_test.evaluar_desempeno(y_real, prediccion, latencia_ms=0.1, nombre="a")
    eval_b = ab_test.evaluar_desempeno(y_real, prediccion.copy(), latencia_ms=0.1, nombre="b")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, fechas=fechas_por_dia(n, 3))
    ruta = ab_test.guardar_resultados_ab(res, carpeta_salida=tmp_path)
    leido = leer_json_estricto(ruta)

    assert leido == json.loads(json.dumps(res, allow_nan=False))
    assert leido["analisis_por_dia"]["caso_degenerado_t_dia"] == ab_test.DIFERENCIAS_NULAS
    fila = leido["referencia_por_fila"]
    assert fila["t_stat"] is None and fila["p_valor_t"] is None
    assert fila["wilcoxon_stat"] is None and fila["p_valor_wilcoxon"] is None
    assert fila["caso_degenerado_wilcoxon"] == ab_test.DIFERENCIAS_NULAS
    assert fila["es_estadisticamente_significativo"] is False
    assert leido["decision"]["criterios_cumplidos"]["t_por_dia_significativa"] is False
    assert leido["decision"]["recomendacion"] == "descartar_challenger"
    assert "no definida" in ab_test.formatear_tabla_markdown(leido)


def test_guardar_resultados_rechaza_valores_no_finitos(tmp_path: Path):
    with pytest.raises(ValueError, match="JSON compliant"):
        ab_test.guardar_resultados_ab({"t": float("inf")}, carpeta_salida=tmp_path)


# Precisión completa --------------------------------------------------------------------


def test_decide_con_precision_completa_aunque_las_diferencias_sean_muy_pequenas(tmp_path: Path):
    """Diferencias diarias de 2e-7 a 4e-7: el intervalo excluye el cero sin redondear."""
    diferencias = np.array([2e-7, 3e-7, 4e-7, 3e-7])
    filas_por_dia = 5
    fechas = fechas_por_dia(len(diferencias) * filas_por_dia, len(diferencias))
    err_b = np.ones(len(fechas))
    err_a = err_b + np.repeat(diferencias, filas_por_dia)
    y_real = np.zeros(len(fechas))
    eval_a = ab_test.evaluar_desempeno(y_real, err_a, latencia_ms=0.1, nombre="a")
    eval_b = ab_test.evaluar_desempeno(y_real, err_b, latencia_ms=0.1, nombre="b")

    res = ab_test.ejecutar_ab_testing(eval_a, eval_b, fechas=fechas)

    dia = res["analisis_por_dia"]
    assert dia["ic_95_bootstrap"][0] == pytest.approx(2.25e-7, rel=1e-6)
    assert dia["ic_95_bootstrap"][1] == pytest.approx(3.75e-7, rel=1e-6)
    assert dia["diferencia_mae_media"] == pytest.approx(3e-7, rel=1e-6)
    assert dia["p_valor_t_dia"] == pytest.approx(0.005208, abs=1e-6)
    assert res["decision"]["criterios_cumplidos"]["ic_bootstrap_favorable"] is True
    assert res["decision"]["recomendacion"] == "promover_challenger"
    # El JSON conserva la misma precisión con que se decidió
    leido = leer_json_estricto(ab_test.guardar_resultados_ab(res, carpeta_salida=tmp_path))
    assert leido["analisis_por_dia"]["ic_95_bootstrap"] == dia["ic_95_bootstrap"]
    assert [d["diferencia_mae"] for d in leido["analisis_por_dia"]["por_dia"]] == pytest.approx(
        diferencias, rel=1e-6
    )


def test_las_estadisticas_se_reconstruyen_desde_los_mae_diarios_guardados(tmp_path: Path):
    rng = np.random.default_rng(5)
    n_dias, filas_por_dia = 15, 40
    fechas = fechas_por_dia(n_dias * filas_por_dia, n_dias)
    err_a = rng.gamma(2.0, 0.5, len(fechas))
    err_b = err_a - rng.normal(0.002, 0.01, len(fechas))

    res = ab_test.analisis_por_dia(err_a, err_b, fechas, n_bootstrap=2000)
    leido = json.loads(json.dumps(res, allow_nan=False))

    diferencias = np.array([d["diferencia_mae"] for d in leido["por_dia"]])
    recalculado = ab_test.bootstrap_dias(diferencias, n_bootstrap=2000, semilla=SEMILLA)
    assert recalculado["ic_95_bootstrap"] == leido["ic_95_bootstrap"]
    assert (
        recalculado["proporcion_remuestreos_gana_challenger"]
        == leido["proporcion_remuestreos_gana_challenger"]
    )


# Huecos de cobertura del análisis por día y por fila ----------------------------------------


def test_cada_dia_pesa_lo_mismo_aunque_tenga_distinto_numero_de_filas():
    """Un día de tres filas con diferencia 1 y uno de una fila con diferencia 0: media 0.5."""
    fechas = np.array([date(2026, 9, 16)] * 3 + [date(2026, 9, 17)])
    err_b = np.ones(4)
    err_a = np.array([2.0, 2.0, 2.0, 1.0])

    res = ab_test.analisis_por_dia(err_a, err_b, fechas, n_bootstrap=500)

    assert res["diferencia_mae_media"] == pytest.approx(0.5)
    assert np.mean(err_a - err_b) == pytest.approx(0.75)
    assert [d["diferencia_mae"] for d in res["por_dia"]] == [1.0, 0.0]


def test_percentiles_bootstrap_coinciden_con_la_distribucion_exacta():
    """Con cuatro días se enumeran los 256 remuestreos posibles y se comparan los percentiles."""
    diferencias = np.array([-0.02, 0.01, 0.03, 0.06])
    medias = np.sort([np.mean(c) for c in itertools.product(diferencias, repeat=4)])
    acumulada = np.arange(1, len(medias) + 1) / len(medias)
    exacto_inf = medias[np.searchsorted(acumulada, 0.025)]
    exacto_sup = medias[np.searchsorted(acumulada, 0.975)]
    paso = 0.0025  # separación entre medias posibles (0.01 / 4)

    res = ab_test.bootstrap_dias(diferencias, n_bootstrap=200_000, semilla=SEMILLA)

    ic_inf, ic_sup = res["ic_95_bootstrap"]
    assert abs(ic_inf - exacto_inf) <= paso + 1e-12
    assert abs(ic_sup - exacto_sup) <= paso + 1e-12
    assert res["proporcion_remuestreos_gana_challenger"] == pytest.approx(
        np.mean(medias > 0), abs=0.005
    )


def test_analisis_por_dia_requiere_al_menos_dos_dias():
    fechas = np.array([date(2026, 9, 16)] * 5)
    y_real = np.zeros(5)
    eval_a = ab_test.evaluar_desempeno(y_real, np.full(5, 2.0), latencia_ms=0.1, nombre="a")
    eval_b = ab_test.evaluar_desempeno(y_real, np.full(5, 1.0), latencia_ms=0.1, nombre="b")

    with pytest.raises(ValueError, match="al menos dos días"):
        ab_test.analisis_por_dia(np.full(5, 2.0), np.ones(5), fechas)
    with pytest.raises(ValueError, match="al menos dos días"):
        ab_test.ejecutar_ab_testing(eval_a, eval_b, fechas=fechas)


@pytest.mark.parametrize("n", [500, 60_000])
def test_pruebas_por_fila_conservan_los_calculos_originales(n: int):
    """t, IC normal y Wilcoxon iguales a SciPy; con más de 50,000 filas, la submuestra de 20,000."""
    rng = np.random.default_rng(6)
    err_a = rng.gamma(2.0, 0.5, n)
    err_b = err_a - rng.normal(0.003, 0.2, n)

    res = ab_test.comparar_por_fila(err_a, err_b)

    t_esperada = stats.ttest_rel(err_a, err_b)
    diff = err_a - err_b
    sem = stats.sem(diff)
    if n > 50_000:
        idx = np.random.default_rng(SEMILLA).choice(n, size=20_000, replace=False)
        w_esperada = stats.wilcoxon(err_a[idx], err_b[idx])
    else:
        w_esperada = stats.wilcoxon(err_a, err_b)
    assert res["n_filas"] == n
    assert res["t_stat"] == float(t_esperada.statistic)
    assert res["p_valor_t"] == float(t_esperada.pvalue)
    assert res["wilcoxon_stat"] == float(w_esperada.statistic)
    assert res["p_valor_wilcoxon"] == float(w_esperada.pvalue)
    assert res["diferencia_media_mae"] == float(np.mean(diff))
    assert res["ic_95_diferencia"] == [np.mean(diff) - 1.96 * sem, np.mean(diff) + 1.96 * sem]
    assert res["caso_degenerado_t"] is None and res["caso_degenerado_wilcoxon"] is None
