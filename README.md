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
- **Rango:** abril a septiembre de 2026. Termina en el mismo mes que la nomenclatura de estaciones (`nomenclatura_2026_09.csv`), para que la capacidad y la ubicación de cada estación correspondan al final del periodo. Las 484 estaciones de esa nomenclatura incluyen 14 cuyo nombre lleva el prefijo "ex-"; ninguna de ellas registra viajes entre abril y septiembre. La nomenclatura no explica el prefijo.
- **Tamaño:** 2,294,886 registros crudos. Se eliminan 758 duplicados exactos y 3,954 viajes de dos estaciones (405 y 407) que no aparecen en la nomenclatura. Quedan **2,290,174 viajes** y **484 estaciones**.
- **Datos personales:** se eliminan en la ingesta `usuario_id`, `genero` y `anio_de_nacimiento`. Ninguna de las tres preguntas los necesita y no se registran en MLflow ni en el repositorio.
- **Partición:** temporal. Entrenamiento de abril a agosto y prueba en septiembre, para que ningún modelo se entrene con información posterior al periodo que predice. En P3, los perfiles se calculan con los viajes iniciados entre abril y agosto; 18 de ellos terminan el 1 de septiembre y sus llegadas cuentan en la proporción de salidas de 18 estaciones. La cambian en menos de 0.0002 y ninguna estación cambia de grupo por ello.
- **Versionado:** `references/manifiesto_datos.json` guarda la URL, el número de registros y el hash SHA-256 de cada archivo crudo, y una `version_datos` que se registra como parámetro en cada run de MLflow.

## Instalación

Requiere Python 3.10. Primero se clona el repositorio:

```bash
git clone https://github.com/a01840021-creator/TC5061-Proyecto-Equipo9.git
cd TC5061-Proyecto-Equipo9
```

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
mlflow run . --experiment-name mibici-equipo09 -e search -P pregunta=p1
mlflow run . --experiment-name mibici-equipo09 -e search -P pregunta=p2
mlflow run . --experiment-name mibici-equipo09 -e search -P pregunta=p3
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

### 4. Model Registry y alias de producción

El equipo utiliza aliases, esquema vigente de MLflow, en lugar de las etapas
clásicas `Staging` y `Production`, que están obsoletas. El alias `champion`
identifica la versión aprobada para producción y conserva el linaje hacia la
corrida, el commit y la versión de datos.

```bash
mlflow run . --experiment-name mibici-equipo09 -e register \
  -P p1_run_id=43ba9b881abd4b0bb4cc7188f98eb3ad \
  -P p2_run_id=91d87b354adf4612b258ea19342d2577 \
  -P p3_run_id=96b2a3ba470048a6aea3ea756f1a6f14 \
  --env-manager local
```

El punto de entrada es idempotente: si una versión ya proviene de la misma
corrida, reutiliza esa versión en lugar de crear un duplicado. Finalmente carga
los tres modelos mediante `models:/<nombre>@champion` para verificar que están
disponibles.

### Experimento base

Todos los runs se agrupan en el experimento `mibici-equipo09` y registran el rango de
fechas, la versión de datos `0c4b34e5425a`, la semilla, el modelo, el tamaño de cada
partición y etiquetas con el equipo, la pregunta, el autor y el commit de Git.

| Pregunta | Baseline | Métricas principales | Artefactos |
| --- | --- | --- | --- |
| P1 | Demanda de la misma estación y hora una semana antes (`rezago_168h`) | MAE, RMSE, WAPE, R² y sesgo | Serie real frente a predicción, resumen horario y JSON de métricas |
| P2 | Persistencia del saldo de la hora anterior respecto de la capacidad | Precision, recall, F1, balanced accuracy y average precision | Matriz de confusión, riesgo diario y JSON de métricas |
| P3 | K-Means con `k=4` sobre perfiles horarios estandarizados | Silhouette, Davies-Bouldin, Calinski-Harabasz e inercia | Mapa de clusters, curvas horarias, tabla de estaciones y modelo |

Los modelos candidatos de cada pregunta, la justificación de cada métrica según la decisión que apoya y los resultados de los baselines están en [`references/modelos_metricas.md`](references/modelos_metricas.md).

## Estructura

```
├── Makefile                 comandos del proyecto (make data)
├── requirements.txt         dependencias con versión fija
├── pyproject.toml
├── data/                    no se versiona
│   ├── raw/                 CSV originales descargados
│   └── processed/           viajes.parquet y estaciones.parquet
├── docs/                    carpeta de la plantilla, sin contenido por ahora
├── models/                  modelos entrenados, no se versionan
├── notebooks/               solo exploración: 1.0-dcc-exploracion-inicial.ipynb
├── references/              manifiesto de datos, ML Canvas, definición de datos,
│                            modelos y métricas, y diagrama de arquitectura
├── reports/figures/         figuras generadas
├── tests/                   pruebas de configuración y baselines
└── mibici/
    ├── config.py            rutas, URLs, fechas, partición, umbrales, semilla y MLflow
    ├── dataset.py           ingesta reproducible
    ├── features.py          ingeniería de variables
    ├── modeling/
    │   ├── artifacts.py     gráficas y tablas registradas en MLflow
    │   ├── baselines.py     reglas base y métricas de P1, P2 y P3
    │   ├── train.py         CLI y tracking de los tres runs
    │   └── predict.py       plantilla de Cookiecutter, sin usar
    └── plots.py             plantilla de Cookiecutter, sin usar
```

## Buenas prácticas

| Ámbito | Práctica | Cómo y dónde se cumple |
| --- | --- | --- |
| MLflow Tracking | Un experimento por proyecto y runs con nombre descriptivo | Todos los runs van al experimento `mibici-equipo09`, definido en `EXPERIMENTO` de `mibici/config.py`. `mibici/modeling/train.py` nombra cada run: `P1_baseline_rezago_semanal`, `P2_baseline_persistencia_saldo` y `P3_baseline_kmeans_k4`. |
| MLflow Tracking | Registrar lo necesario para reproducir el run | Cada run registra como parámetros `fecha_inicio`, `fecha_fin`, `periodo_train`, `periodo_test`, `version_datos` (del manifiesto) y `semilla`, además del modelo y sus hiperparámetros: `umbral_capacidad` y `horizonte_horas` en P2; `n_clusters` y `n_init` en P3. |
| MLflow Tracking | Ligar cada run con su commit y su pregunta | Etiquetas `git_commit` (SHA de `git rev-parse HEAD`), `pregunta` (`P1`, `P2` o `P3`), `equipo`, `autor` y `tipo_modelo`. |
| MLflow Tracking | Métricas con el conjunto en su nombre | Prefijo `test_` en P1 y P2, que se evalúan en septiembre (`test_mae`, `test_recall`), y `train_` en P3, que se evalúa sobre los perfiles de entrenamiento, calculados con los viajes iniciados entre abril y agosto (`train_silhouette`; ver [Datos](#datos) sobre los 18 que terminan el 1 de septiembre). |
| MLflow Tracking | Sin credenciales ni datos de los usuarios de MiBici en parámetros, etiquetas o artefactos | La ingesta elimina `usuario_id`, `genero` y `anio_de_nacimiento`, las columnas con datos de los usuarios de MiBici, antes de guardar los Parquet (`mibici/dataset.py`), así que ningún artefacto puede contenerlas. La etiqueta `autor` registra tal cual el `user.name` de la configuración de Git de quien ejecuta o, si no está definido, el usuario del sistema (`obtener_autor` en `mibici/modeling/train.py`). El tracking es local y no usa credenciales; `.env` está en `.gitignore`. |
| Estructura y código | Notebooks sólo para exploración, con la convención de nombres | `notebooks/1.0-dcc-exploracion-inicial.ipynb` sigue el formato `número-iniciales-descripción` e importa las funciones de `mibici.features`; la lógica final vive en el módulo `mibici/`. |
| Estructura y código | Configuración en un solo lugar | Para el flujo de ingesta, construcción de variables y experimento, `mibici/config.py` concentra las carpetas de datos, las rutas de los Parquet procesados, las URLs, las fechas, el corte de partición, la duración mínima de un viaje válido, el umbral y el horizonte de P2, los hiperparámetros de K-Means, la semilla, el nombre del experimento y la URI de tracking. Los periodos que se registran en MLflow se derivan de esas constantes. El nombre local de cada CSV crudo se arma en `mibici/dataset.py` con la clave del mes. `mibici/modeling/predict.py` y `mibici/plots.py` son plantillas de Cookiecutter sin usar y conservan sus rutas de ejemplo. |
| Estructura y código | Dependencias con versión fija y entorno reproducible | `requirements.txt` fija con `==` cada paquete; la instalación con Python 3.10 está en [Instalación](#instalación). |
| Datos y Git | Ningún dato, modelo, carpeta de MLflow ni credencial en el historial | `.gitignore` excluye `/data/`, `*.parquet`, `datos_abiertos_*.csv`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `models/*`, `*.pkl`, `*.joblib`, `*.skops` y `.env`. Los datos se versionan con su hash en `references/manifiesto_datos.json`, no con los archivos. |
| Datos y Git | Todo cambio entra a `main` por Pull Request revisado | La rama `main` está protegida con un ruleset que exige un Pull Request con al menos una aprobación e impide el push directo y el borrado. Ver [Flujo de trabajo con Git](#flujo-de-trabajo-con-git). |

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
