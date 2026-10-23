"""Pruebas unitarias ligeras para la configuración de candidatos."""

import pytest

from mibici.modeling.candidates import configuraciones_busqueda


@pytest.mark.parametrize("pregunta", ["P1", "P2", "P3"])
def test_busqueda_contiene_tres_candidatos(pregunta: str) -> None:
    configuraciones = configuraciones_busqueda(pregunta)
    assert len(configuraciones) == 3
    assert all("modelo" in config for config in configuraciones)
    assert all("run_name" in config for config in configuraciones)


def test_p3_compara_dos_familias() -> None:
    familias = {config["modelo"] for config in configuraciones_busqueda("P3")}
    assert familias == {"kmeans", "minibatch_kmeans"}


def test_pregunta_invalida_falla() -> None:
    with pytest.raises(ValueError):
        configuraciones_busqueda("P4")
