"""Pruebas del agrupamiento de Ward con asignación por centroide."""

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.modeling.clustering import WardCentroides


def datos():
    return make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=0)


def test_ward_ajusta_y_expone_atributos():
    x, _ = datos()
    modelo = WardCentroides(n_clusters=3).fit(x)
    assert modelo.labels_.shape == (90,)
    assert modelo.cluster_centers_.shape == (3, 2)
    assert set(modelo.labels_) == {0, 1, 2}
    for grupo in range(3):
        np.testing.assert_allclose(
            modelo.cluster_centers_[grupo], x[modelo.labels_ == grupo].mean(axis=0)
        )
    assert modelo.inertia_ > 0


def test_fit_predict_coincide_con_labels_y_predict():
    x, _ = datos()
    modelo = WardCentroides(n_clusters=3)
    etiquetas = modelo.fit_predict(x)
    np.testing.assert_array_equal(etiquetas, modelo.labels_)
    np.testing.assert_array_equal(modelo.predict(x), etiquetas)


def test_predict_asigna_al_centroide_mas_cercano():
    x, _ = datos()
    modelo = WardCentroides(n_clusters=3).fit(x)
    nuevos = modelo.cluster_centers_ + 0.01
    np.testing.assert_array_equal(modelo.predict(nuevos), np.arange(3))


def test_compatible_con_clone_y_pipeline():
    x, _ = datos()
    copia = clone(WardCentroides(n_clusters=4))
    assert copia.get_params() == {"n_clusters": 4}
    pipeline = Pipeline([("scaler", StandardScaler()), ("cluster", WardCentroides(3))])
    assert pipeline.fit_predict(x).shape == (90,)
    assert pipeline.predict(x[:5]).shape == (5,)
