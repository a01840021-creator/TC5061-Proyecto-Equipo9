"""Pruebas de la selección de candidatas para el Model Registry."""

from types import SimpleNamespace

from mibici.config import ESTACIONES_TODAS
from mibici.modeling.registry import filtro_candidatas, ordenar_candidatas


def corrida(run_id: str, **metricas: float):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(metrics=metricas)
    )


def test_menor_es_mejor_para_mae():
    runs = [corrida("a", val_mae=0.9), corrida("b", val_mae=0.7), corrida("c", val_mae=0.8)]
    ordenadas = ordenar_candidatas(runs, "val_mae", mayor_es_mejor=False)
    assert [r.info.run_id for r in ordenadas] == ["b", "c", "a"]


def test_mayor_es_mejor_para_average_precision():
    runs = [
        corrida("a", val_average_precision=0.2),
        corrida("b", val_average_precision=0.4),
        corrida("c", val_average_precision=0.3),
    ]
    ordenadas = ordenar_candidatas(runs, "val_average_precision", mayor_es_mejor=True)
    assert [r.info.run_id for r in ordenadas[:2]] == ["b", "c"]


def test_ignora_corridas_sin_la_metrica():
    runs = [corrida("a", test_mae=0.1), corrida("b", val_mae=0.7)]
    assert [r.info.run_id for r in ordenar_candidatas(runs, "val_mae", False)] == ["b"]


def test_filtro_exige_candidata_terminada_del_commit():
    filtro = filtro_candidatas("P1", "abc123")
    assert "tags.stage_of_work = 'candidate'" in filtro
    assert "tags.question = 'P1'" in filtro
    assert "tags.git_commit = 'abc123'" in filtro
    assert "attributes.status = 'FINISHED'" in filtro
    assert "params.estaciones" not in filtro


def test_p3_solo_compite_con_todas_las_estaciones():
    assert f"params.estaciones = '{ESTACIONES_TODAS}'" in filtro_candidatas("P3", "abc123")
