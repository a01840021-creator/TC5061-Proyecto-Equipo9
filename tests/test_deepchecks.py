"""Pruebas de calidad con casos normales y casos límite sintéticos."""

import numpy as np
import pandas as pd

from mibici.quality import componentes_deepchecks, muestrear_deterministicamente


def test_muestreo_es_reproducible_y_valida_el_limite():
    datos = pd.DataFrame({"valor": range(100)})
    muestra_1 = muestrear_deterministicamente(datos, max_filas=10, semilla=17)
    muestra_2 = muestrear_deterministicamente(datos, max_filas=10, semilla=17)
    pd.testing.assert_frame_equal(muestra_1, muestra_2)
    assert len(muestra_1) == 10


def test_deepchecks_detecta_duplicados():
    Dataset, _, _ = componentes_deepchecks()
    from deepchecks.tabular.checks import DataDuplicates

    limpio = Dataset(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    duplicado = Dataset(pd.DataFrame({"x": [1, 2, 2], "y": [4, 5, 5]}))

    check = DataDuplicates().add_condition_ratio_less_or_equal(0)
    assert check.run(limpio).passed_conditions()
    assert not check.run(duplicado).passed_conditions()


def test_deepchecks_detecta_valores_faltantes():
    Dataset, _, _ = componentes_deepchecks()
    from deepchecks.tabular.checks import PercentOfNulls

    limpio = Dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    con_nulo = Dataset(pd.DataFrame({"x": [1.0, np.nan, 3.0]}))

    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0)
    assert check.run(limpio).passed_conditions()
    assert not check.run(con_nulo).passed_conditions()


def test_deepchecks_detecta_drift_numerico():
    Dataset, _, _ = componentes_deepchecks()
    from deepchecks.tabular.checks import FeatureDrift

    rng = np.random.default_rng(42)
    train = Dataset(pd.DataFrame({"demanda": rng.normal(0, 1, 500)}))
    test = Dataset(pd.DataFrame({"demanda": rng.normal(5, 1, 500)}))
    check = FeatureDrift().add_condition_drift_score_less_than(
        max_allowed_categorical_score=0.2,
        max_allowed_numeric_score=0.2,
    )

    assert not check.run(train, test).passed_conditions()
