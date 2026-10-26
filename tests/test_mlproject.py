"""El MLproject no puede leer config.py; esta prueba evita que sus valores se desalineen."""

import yaml

from mibici import config
from mibici.config import PROJ_ROOT

ESPERADOS = {
    "learning_rate": config.LEARNING_RATE_DEFAULT,
    "max_iter": config.MAX_ITER_DEFAULT,
    "max_leaf_nodes": config.MAX_LEAF_NODES_DEFAULT,
    "l2_regularization": config.L2_REGULARIZATION_DEFAULT,
    "alpha": config.ALPHA_DEFAULT,
    "C": config.C_DEFAULT,
    "n_clusters": config.N_CLUSTERS_DEFAULT,
    "batch_size": config.BATCH_SIZE_DEFAULT,
    "estaciones": config.ESTACIONES_DEFAULT,
}


def cargar_mlproject() -> dict:
    return yaml.safe_load((PROJ_ROOT / "MLproject").read_text(encoding="utf-8"))


def test_valores_por_defecto_de_train_coinciden_con_config():
    parametros = cargar_mlproject()["entry_points"]["train"]["parameters"]
    defaults = {k: v["default"] for k, v in parametros.items() if "default" in v}
    assert defaults == ESPERADOS


def test_register_no_recibe_parametros():
    assert "parameters" not in cargar_mlproject()["entry_points"]["register"]
