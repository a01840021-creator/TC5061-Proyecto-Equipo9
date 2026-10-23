# MiBici: demanda, riesgo de rebalanceo y patrones de uso

Proyecto integrador del equipo 9. Propone y construye una solución de Machine Learning sobre los datos abiertos de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, para responder tres preguntas de operación.

## Integrantes

| Matrícula | Nombre | Contribución |
| --- | --- | --- |
| A01840021 | Carrera Rojas, Mariana Judith | Configuración inicial: estructura con Cookiecutter Data Science, repositorio en GitHub y protección de la rama principal |
| A01797854 | Barrera Mendoza, José Manuel | Formulación del problema y ML Canvas, definición de los datos, dependencias con versión fija, ingesta reproducible y versionado de datos, flujo de trabajo con Git |
| A01797868 | Barranco Cruz, Luis Fernando | Implementación de los baselines P1, P2 y P3; instrumentación con MLflow Tracking; métricas, artefactos y pruebas automatizadas |
| A01840136 | Carrera Castillo, David | Análisis exploratorio, módulo reutilizable de feature engineering sin fuga temporal y diagrama de arquitectura |

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

Con `uv`:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Como alternativa, con Conda y `pip`:

```bash
conda create -n mibici-equipo9 python=3.10 -y
conda activate mibici-equipo9
python -m pip install -r requirements.txt
```

## Uso

| Comando | Qué hace |
| --- | --- |
| `python -m mibici.dataset` | Descarga los seis meses y la nomenclatura, limpia, guarda en Parquet y escribe el manifiesto |
| `python -m mibici.modeling.train --pregunta all` | Ejecuta P1, P2 y P3 y registra tres runs en `mibici-equipo09` |
| `python -m mibici.modeling.train --pregunta P1` | Ejecuta únicamente el baseline de regresión |
| `python -m mibici.modeling.train --pregunta P2` | Ejecuta únicamente el baseline de clasificación |
| `python -m mibici.modeling.train --pregunta P3` | Ejecuta únicamente el baseline de clustering |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` | Abre la interfaz local de MLflow |
| `python -m pytest -q tests` | Ejecuta las pruebas automatizadas |

## MLflow Project y tracking avanzado

### 1. Levantar el servidor local

El backend SQLite permite usar Tracking y Model Registry. Desde la raíz del
repositorio, con el entorno activado, se ejecuta:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

El servidor permanece abierto en esa terminal. En una segunda terminal:

```bash
conda activate mibici-equipo9
cd ~/Documents/NLP/MLops/TC5061-Proyecto-Equipo9
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

La interfaz queda disponible en <http://127.0.0.1:5000>. El archivo
`mlflow.db`, los artefactos y los modelos locales están excluidos por
`.gitignore`.

### 2. Ejecutar el proyecto reproducible

El archivo `MLproject` declara el entorno `python_env.yaml` y cuatro puntos de
entrada. Las rutas, fechas y la semilla se leen de `mibici/config.py`.

```bash
# Ingesta completa de los seis meses
mlflow run . -e ingest

# Reproducir los tres baselines
mlflow run . -e baseline -P pregunta=all

# Ejecutar un único candidato
mlflow run . -e train -P pregunta=p1 -P modelo=hist_gradient_boosting \
  -P learning_rate=0.05 -P max_iter=150 -P max_leaf_nodes=31

# Búsqueda organizada como un run padre y tres runs hijos
mlflow run . -e search -P pregunta=p1
mlflow run . -e search -P pregunta=p2
mlflow run . -e search -P pregunta=p3
```

Durante desarrollo puede añadirse `--env-manager local` a cada comando para
usar el entorno Conda ya activado. Para comprobar la reproducibilidad desde un
clon limpio se omite esa opción y MLflow crea el entorno declarado.

### 3. Diseño de la experimentación

- **P1:** `HistGradientBoostingRegressor` con combinaciones de tasa de
  aprendizaje, iteraciones, hojas y regularización L2.
- **P2:** `HistGradientBoostingClassifier` con pesos balanceados; se priorizan
  recall, F1 y average precision debido al desbalance de la clase de riesgo.
- **P3:** K-Means y MiniBatch K-Means con distintos valores de `k`; se comparan
  silhouette, Davies-Bouldin, Calinski-Harabasz e inercia.

Cada búsqueda crea un run padre (`stage_of_work=search`) y tres runs hijos
(`stage_of_work=candidate`). Todos incluyen `question`, `model_family`, equipo,
commit de Git y versión de datos. Agosto se usa como validación y septiembre
como prueba; la selección se realiza con validación para no elegir un modelo a
partir del conjunto de prueba.

Cada candidato registra:

- el modelo con firma y ejemplo de entrada;
- el manifiesto y metadatos del conjunto de entrenamiento mediante
  `mlflow.log_input`;
- gráficas en `plots/`;
- configuración efectiva en `config/`;
- parámetros, métricas de validación y métricas de prueba.

### Experimento base

Todos los runs se agrupan en el experimento `mibici-equipo09` y registran el rango de
fechas, la versión de datos `0c4b34e5425a`, la semilla, el modelo, el tamaño de cada
partición y etiquetas con el equipo, la pregunta, el autor y el commit de Git.

| Pregunta | Baseline | Métricas principales | Artefactos |
| --- | --- | --- | --- |
| P1 | Demanda de la misma estación y hora una semana antes (`rezago_168h`) | MAE, RMSE, WAPE, R² y sesgo | Serie real frente a predicción, resumen horario y JSON de métricas |
| P2 | Persistencia del saldo de la hora anterior respecto de la capacidad | Precision, recall, F1, balanced accuracy y average precision | Matriz de confusión, riesgo diario y JSON de métricas |
| P3 | K-Means con `k=4` sobre perfiles horarios estandarizados | Silhouette, Davies-Bouldin, Calinski-Harabasz e inercia | Mapa de clusters, curvas horarias, tabla de estaciones y modelo |

## Estructura

```
├── Makefile                 comandos del proyecto (make data)
├── requirements.txt         dependencias con versión fija
├── pyproject.toml
├── data/                    no se versiona
│   ├── raw/                 CSV originales descargados
│   └── processed/           viajes.parquet y estaciones.parquet
├── models/                  modelos entrenados, no se versionan
├── notebooks/               solo exploración: 1.0-dcc-exploracion-inicial.ipynb
├── references/              manifiesto de datos, ML Canvas y diagrama de arquitectura
├── reports/figures/         figuras generadas
└── mibici/
    ├── config.py            rutas, URLs, rango de fechas y semilla
    ├── dataset.py           ingesta reproducible
    ├── features.py          ingeniería de variables
    ├── modeling/
    │   ├── artifacts.py      gráficas y tablas registradas en MLflow
    │   ├── baselines.py      reglas base y métricas de P1, P2 y P3
    │   ├── train.py          CLI y tracking de los tres runs
    │   └── predict.py
    └── plots.py
└── tests/                   pruebas de configuración y baselines
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
