"""Pruebas de la auditoría reproducible del proyecto final."""

import json
from pathlib import Path

import yaml

from mibici.auditoria_final import (
    COMPONENTES_REQUERIDOS,
    ENTRY_POINTS_ESPERADOS,
    VERSION_PYTHON_ESPERADA,
    ejecutar_auditoria,
    verificar_componentes,
    verificar_dependencias_fijadas,
    verificar_entregables_finales,
    verificar_linaje_champion,
    verificar_mlproject,
    verificar_version_python,
)


def _escribir(ruta: Path, contenido: str = "contenido\n") -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")


def _repositorio_minimo(raiz: Path) -> Path:
    for rutas in COMPONENTES_REQUERIDOS.values():
        for ruta in rutas:
            _escribir(raiz / ruta)

    _escribir(
        raiz / "python_env.yaml",
        yaml.safe_dump({"python": VERSION_PYTHON_ESPERADA}),
    )
    _escribir(raiz / "Dockerfile", f"FROM python:{VERSION_PYTHON_ESPERADA}-slim\n")
    _escribir(
        raiz / ".github/workflows/ci.yml",
        yaml.safe_dump(
            {
                "jobs": {
                    "verificar": {
                        "steps": [
                            {
                                "uses": "actions/setup-python@v5",
                                "with": {"python-version": VERSION_PYTHON_ESPERADA},
                            }
                        ]
                    }
                }
            }
        ),
    )
    _escribir(
        raiz / "MLproject",
        yaml.safe_dump(
            {
                "entry_points": {
                    nombre: {"command": "python -V"} for nombre in ENTRY_POINTS_ESPERADOS
                }
            }
        ),
    )
    _escribir(raiz / "requirements.txt", "pandas==2.3.3\n-e .\n")
    _escribir(raiz / "requirements-serving.txt", "fastapi==0.143.3\n")
    _escribir(
        raiz / "models/p1/metadata.json",
        json.dumps(
            {
                "model_name": "mibici-equipo09-p1-demanda",
                "alias": "champion",
                "version": "1",
                "run_id": "run-123",
                "periodo_train": "2026-04-01/2026-07-31",
            }
        ),
    )
    return raiz


def test_componentes_detectan_archivo_faltante(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    (raiz / "locustfile.py").unlink()

    resultados = verificar_componentes(raiz)
    despliegue = next(r for r in resultados if r.nombre == "componentes_despliegue_y_operacion")

    assert not despliegue.aprobado
    assert "locustfile.py" in despliegue.detalle


def test_version_python_alineada_y_desalineada(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    assert verificar_version_python(raiz).aprobado

    _escribir(raiz / "Dockerfile", "FROM python:3.11.9-slim\n")
    resultado = verificar_version_python(raiz)

    assert not resultado.aprobado
    assert "Dockerfile=3.11.9" in resultado.detalle


def test_dependencias_exigen_version_exacta(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    assert verificar_dependencias_fijadas(raiz).aprobado

    _escribir(raiz / "requirements.txt", "pandas>=2.0\n")
    resultado = verificar_dependencias_fijadas(raiz)

    assert not resultado.aprobado
    assert "pandas>=2.0" in resultado.detalle


def test_mlproject_exige_todos_los_entry_points(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    assert verificar_mlproject(raiz).aprobado

    _escribir(raiz / "MLproject", yaml.safe_dump({"entry_points": {"ingest": {}}}))
    resultado = verificar_mlproject(raiz)

    assert not resultado.aprobado
    assert "register" in resultado.detalle


def test_linaje_exige_alias_champion(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    assert verificar_linaje_champion(raiz).aprobado

    metadatos = json.loads((raiz / "models/p1/metadata.json").read_text(encoding="utf-8"))
    metadatos["alias"] = "challenger"
    _escribir(raiz / "models/p1/metadata.json", json.dumps(metadatos))

    assert not verificar_linaje_champion(raiz).aprobado


def test_entregables_finales_son_pendientes_no_errores(tmp_path):
    resultados = verificar_entregables_finales(tmp_path)

    assert all(not resultado.aprobado for resultado in resultados)
    assert all(resultado.severidad == "pendiente" for resultado in resultados)


def test_auditoria_distingue_base_tecnica_de_entrega_completa(tmp_path):
    raiz = _repositorio_minimo(tmp_path)
    reporte = ejecutar_auditoria(raiz)

    assert reporte["resumen"]["base_tecnica_reproducible"] is True
    assert reporte["resumen"]["entrega_completa"] is False
    assert reporte["resumen"]["pendientes_entrega"] == 3

    _escribir(raiz / "MODEL_CARD.md")
    _escribir(raiz / "reports/final/TC5061_Proyecto_Final_Equipo09.pdf")
    _escribir(raiz / "reports/final/enlace_video.txt", "https://example.com/video\n")
    reporte_completo = ejecutar_auditoria(raiz)

    assert reporte_completo["resumen"]["entrega_completa"] is True
