"""Estimadores de agrupamiento propios del proyecto, compatibles con scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted, validate_data


class WardCentroides(ClusterMixin, BaseEstimator):
    """Agrupamiento jerárquico de Ward que asigna observaciones nuevas por centroide.

    `AgglomerativeClustering` sólo etiqueta las observaciones con que se ajusta y no tiene
    `predict`. Este estimador ajusta Ward, calcula el centroide de cada grupo en el mismo
    espacio de entrada y asigna cualquier observación nueva al centroide más cercano. Así
    el modelo tiene firma, se registra con `log_model` y se carga desde MLflow como
    cualquier otro estimador. En las observaciones de entrenamiento, `predict` puede
    diferir de `labels_` en las pocas que Ward dejó más cerca del centroide de otro grupo.

    Vive en el paquete `mibici` para que el modelo serializado se pueda cargar en cualquier
    entorno que instale el proyecto.
    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="float64")
        ward = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        self.labels_ = ward.fit_predict(X)
        self.cluster_centers_ = np.vstack(
            [X[self.labels_ == grupo].mean(axis=0) for grupo in range(self.n_clusters)]
        )
        self.inertia_ = float(((X - self.cluster_centers_[self.labels_]) ** 2).sum())
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype="float64", reset=False)
        return pairwise_distances_argmin(X, self.cluster_centers_)
