# MiBici: demanda, riesgo de rebalanceo y patrones de uso

Proyecto integrador del equipo 9. Propone y construye una solución de Machine Learning sobre los datos abiertos de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, para responder tres preguntas de operación.

## Integrantes

| Matrícula | Nombre | Contribución |
| --- | --- | --- |
| A01840021 | Carrera Rojas, Mariana Judith | Configuración inicial: estructura con Cookiecutter Data Science, repositorio en GitHub y protección de la rama principal |
| A01797854 | Barrera Mendoza, José Manuel | Dependencias con versión fija, ingesta reproducible y versionado de datos, flujo de trabajo con Git |
| A01797868 | Barranco Cruz, Luis Fernando | Por definir |
| A01840136 | Carrera Castillo, David | Por definir |

## Preguntas del proyecto

| # | Pregunta | Tarea | Quién la usa |
| --- | --- | --- | --- |
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión del sistema |

## Datos

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Las URLs exactas de cada archivo están en `mibici/config.py`.
- **Rango:** abril a septiembre de 2026. Termina en el mismo mes que la nomenclatura de estaciones (`nomenclatura_2026_09.csv`), para que todos los viajes correspondan a estaciones vigentes.
- **Tamaño:** 2,294,886 registros crudos. Se eliminan 758 duplicados exactos y 3,954 viajes de dos estaciones (405 y 407) que no aparecen en la nomenclatura. Quedan **2,290,174 viajes** y **484 estaciones**.
- **Datos personales:** se eliminan en la ingesta `usuario_id`, `genero` y `anio_de_nacimiento`. Ninguna de las tres preguntas los necesita y no se registran en MLflow ni en el repositorio.
- **Partición:** temporal. Entrenamiento de abril a agosto y prueba en septiembre, para que ningún modelo se entrene con información posterior al periodo que predice.
- **Versionado:** `references/manifiesto_datos.json` guarda la URL, el número de registros y el hash SHA-256 de cada archivo crudo, y una `version_datos` que se registra como parámetro en cada run de MLflow.

## Instalación

Requiere Python 3.10.

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Uso

| Comando | Qué hace |
| --- | --- |
| `make data` | Descarga los seis meses y la nomenclatura, limpia, guarda en Parquet y escribe el manifiesto |
| Por definir | Experimento base en MLflow, un run por pregunta |

## Estructura

```
├── Makefile                 comandos del proyecto (make data)
├── requirements.txt         dependencias con versión fija
├── pyproject.toml
├── data/                    no se versiona
│   ├── raw/                 CSV originales descargados
│   └── processed/           viajes.parquet y estaciones.parquet
├── models/                  modelos entrenados, no se versionan
├── notebooks/               solo exploración: 1.0-iniciales-descripcion.ipynb
├── references/              manifiesto de datos, ML Canvas y diagrama de arquitectura
├── reports/figures/         figuras generadas
└── mibici/
    ├── config.py            rutas, URLs, rango de fechas y semilla
    ├── dataset.py           ingesta reproducible
    ├── features.py          ingeniería de variables
    ├── modeling/
    │   ├── train.py
    │   └── predict.py
    └── plots.py
```

## Flujo de trabajo con Git

Seguimos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).

- **Rama principal protegida.** Nadie hace push directo a `main`. Todo cambio entra por Pull Request con al menos una revisión aprobada de otro integrante.
- **Una rama por tarea**, creada desde `main` y borrada al fusionarse. Si una tarea depende de otra que aún no se fusiona, la rama nueva sale de esa y su PR se apila sobre el anterior.
- **Nombres de rama:** `tipo/descripcion-corta`, en minúsculas y con guiones. Por ejemplo, `feat/ingesta-mibici` o `chore/fijar-dependencias`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/), con el mismo tipo que la rama.

| Tipo | Para qué |
| --- | --- |
| `feat` | Funcionalidad nueva |
| `fix` | Corrección de un error |
| `docs` | Documentación |
| `chore` | Mantenimiento: dependencias, configuración |
| `refactor` | Reestructurar código sin cambiar su comportamiento |
| `test` | Pruebas |

- **Lo que nunca entra al repositorio:** datos de MiBici, carpetas de MLflow (`mlruns/`, `mlartifacts/`), modelos serializados ni credenciales (`.env`).
