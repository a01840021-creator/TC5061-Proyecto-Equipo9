# Auditoría de consolidación y reproducibilidad

## Objetivo

Verificar, antes de la entrega final, que el repositorio conserva los contratos técnicos de
las semanas anteriores y que los tres archivos de entrega se encuentran disponibles. La
auditoría no sustituye las pruebas: las complementa con una vista transversal de la rúbrica.

## Ejecución

Desde el entorno Python del proyecto:

```bash
make audit-final
```

El comando escribe `reports/final/auditoria_reproducibilidad.json`. Mientras el equipo
integra la entrega, los faltantes de presentación, video o Model Card se reportan como
pendientes, sin ocultar el resultado de la base técnica.

Para la validación definitiva:

```bash
make audit-final-strict
```

El modo estricto devuelve un código distinto de cero ante cualquier error técnico o
entregable final ausente. El video puede estar versionado como MP4 o representado mediante
`reports/final/enlace_video.txt` con una URL accesible sin solicitar permisos.

## Alcance de los chequeos

1. Componentes de estructura, experimentación, Registry, despliegue, operación, monitoreo y
   FinOps.
2. Python 3.10.20 alineado entre MLflow Projects, Docker y GitHub Actions.
3. Dependencias fijadas con `==` en los entornos de desarrollo y servicio.
4. Entry points `ingest`, `baseline`, `train`, `search` y `register` en `MLproject`.
5. Linaje del champion de P1: nombre, alias, versión, `run_id` y periodo de entrenamiento.
6. Presencia del Model Card, la presentación final y el video o su enlace.

## Validación desde un clon limpio

```bash
git clone https://github.com/a01840021-creator/TC5061-Proyecto-Equipo9.git
cd TC5061-Proyecto-Equipo9
uv venv .venv --python 3.10.20
source .venv/bin/activate
uv pip install -r requirements.txt

python -m ruff format --check .
python -m ruff check .
python -m pytest -q tests
make audit-final
docker compose config -q
```

La construcción y validación integral del contenedor también se ejecuta en GitHub Actions,
por lo que no depende de que la persona auditora tenga Docker disponible localmente.

## Evidencia esperada

- `reports/final/auditoria_reproducibilidad.json` asociado al commit evaluado.
- CI/CD en verde para ese mismo commit.
- Model Card y presentación con los nombres establecidos en la actividad.
- Video accesible y con duración entre dos y cinco minutos.
