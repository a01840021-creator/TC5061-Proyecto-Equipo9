# MiBici: demanda, riesgo de rebalanceo y patrones de uso

Proyecto integrador del equipo 9. Propone y construye una solución de Machine Learning sobre los datos abiertos de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, para responder tres preguntas de operación.

### Contribuciones en la entrega 2: MLflow Project y tracking avanzado

| Integrante | Matrícula | Contribución | Evidencia |
| --- | --- | --- | --- |
| Carrera Rojas, Mariana Judith | A01840021 | Configuración inicial del MLflow Project, con los primeros puntos de entrada en ⁠ MLproject ⁠ y la declaración del entorno en ⁠ python_env.yaml ⁠; revisión del reporte final y actualización de la documentación. | PR #12 —propuesta inicial posteriormente sustituida y cerrada sin fusionar— y revisión aprobatoria del PR #16. |
| Barrera Mendoza, José Manuel | A01797854 | Modelos candidatos por algoritmo, corrección de fugas temporales en P2 y P3, selección automática mediante los aliases ⁠ champion ⁠ y ⁠ challenger ⁠, linaje en Model Registry, corridas finales y reporte. | PR #15 y #16; revisión aprobatoria del PR #14. |
| Barranco Cruz, Luis Fernando | A01797868 | MLflow Project con puntos de entrada parametrizados, búsquedas con nested runs, registro estructurado de artefactos y primera automatización del Model Registry. | PR #13; revisión aprobatoria del PR #15. |
| Carrera Castillo, David | A01840136 | Primera versión del reporte de experimentos e integración de capturas y evidencias de MLflow. | PR #14; revisión aprobatoria del PR #13. |

	⁠*Nota:* El PR #12 se cerró sin fusionarse porque su configuración inicial fue sustituida y ampliada mediante los PR #13 y #15.

### Contribuciones en la entrega 3: testing y seguridad

| Integrante | Matrícula | Contribución | Evidencia |
| --- | --- | --- | --- |
| Carrera Rojas, Mariana Judith | A01840021 | Prueba de reproducibilidad del baseline P3 con semilla fija. | PR #18. |
| Barranco Cruz, Luis Fernando | A01797868 | Integración de DeepChecks para nulos, duplicados y drift; pruebas sintéticas de casos normales y límite; análisis de los hallazgos de calidad. | PR de calidad de datos de esta entrega. |
| Barrera Mendoza, José Manuel | A01797854 | Pendiente de documentar al integrar su contribución. | Pendiente. |
| Carrera Castillo, David | A01840136 | Pendiente de documentar al integrar su contribución. | Pendiente. |

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
- **Partición:** temporal, para que ningún modelo se entrene con información posterior al periodo que predice. Los candidatos de P1 y P2 se entrenan de abril a julio, se validan en agosto y se prueban en septiembre; los baselines de P1 y P2 se evalúan en agosto y septiembre. En P3, los perfiles usan las salidas iniciadas y las llegadas terminadas antes del 1 de septiembre.
- **Versionado:** `references/manifiesto_datos.json` guarda la URL, el número de registros y el hash SHA-256 de cada archivo crudo, y una `version_datos` que se registra como parámetro en cada run de MLflow.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Git. Primero se clona el repositorio:

```bash
git clone https://github.com/a01840021-creator/TC5061-Proyecto-Equipo9.git
cd TC5061-Proyecto-Equipo9
```

Después se crea el entorno local, que da los comandos `mlflow`, `pytest` y `ruff`:

```bash
uv venv .venv --python 3.10.20
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` fija cada paquete con `==` e instala el paquete `mibici` en modo editable. `python_env.yaml` declara la misma versión de Python y las mismas dependencias para `mlflow run`.

## Uso

Las corridas que se reportan se lanzan con `mlflow run` (ver [MLflow Project](#mlflow-project-y-tracking-avanzado)). Los módulos también se pueden ejecutar directamente durante el desarrollo, con el entorno activado y `MLFLOW_TRACKING_URI` exportada:

| Comando | Qué hace |
| --- | --- |
| `python -m mibici.dataset` | Descarga los seis meses y la nomenclatura, limpia, guarda en Parquet y escribe el manifiesto |
| `python -m mibici.modeling.train --pregunta all` | Ejecuta los baselines de P1, P2 y P3 |
| `python -m mibici.modeling.candidates search --pregunta P1` | Ejecuta la búsqueda de candidatos de una pregunta |
| `python -m mibici.modeling.registry` | Elige y registra el champion y el challenger de cada pregunta |
| `python -m pytest -q tests` | Ejecuta las pruebas automatizadas |
| `python -m mibici.quality --pregunta all --max-filas 5000` | Genera reportes DeepChecks de integridad y drift |
| `ruff check` y `ruff format --check` | Revisan el estilo del código |

## Testing y calidad de datos

Todas las pruebas están en `tests/` y se ejecutan con un solo comando:

```bash
python -m pytest -q tests
```

La suite cubre ingesta, *feature engineering*, baselines, candidatos, clustering,
MLflow Project y Model Registry. Incluye casos normales y límite, como horizontes
inválidos, viajes en el límite temporal, estaciones sin actividad, duplicados, nulos y
drift. `test_baseline_p3_es_reproducible` entrena dos veces con la misma semilla y exige
las mismas etiquetas y métricas; `test_muestreo_es_reproducible_y_valida_el_limite`
comprueba que la muestra de calidad también sea determinista.

### Validación con DeepChecks

Primero deben existir `data/processed/viajes.parquet` y
`data/processed/estaciones.parquet`. Después se validan P1 y P2 con muestras de hasta
5,000 filas por partición:

```bash
python -m mibici.quality --pregunta all --max-filas 5000
```

El comando ejecuta las suites `data_integrity` y `train_test_validation` sobre la
partición temporal, y genera HTML y un resumen JSON en `reports/deepchecks/p1/` y
`reports/deepchecks/p2/`. Los reportes locales revisan esquema, tipos, valores faltantes,
duplicados, distribución y drift entre entrenamiento y prueba. La muestra usa
`SEMILLA=42`, por lo que dos ejecuciones sobre la misma versión de datos seleccionan las
mismas filas. Los reportes se excluyen de Git porque pueden contener ejemplos derivados
de los datos; en el repositorio sólo se conserva la captura sin registros individuales.

El drift no se oculta ni se considera automáticamente un error: al dividir por tiempo es
posible que agosto/septiembre difieran de meses anteriores. Se documenta como señal para
investigar y monitorear antes de promover un modelo.

Los resultados obtenidos, su interpretación y las decisiones de tratamiento están en
[`references/calidad_datos.md`](references/calidad_datos.md).

### Evidencia que se conserva en el README

Antes de entregar, el equipo agrega capturas legibles en `reports/figures/` con estos
nombres y sustituye los marcadores siguientes:

- `evidencia_pytest.png`: terminal con todas las pruebas aprobadas.
- `evidencia_deepchecks.png`: resumen de integridad y drift.

> **PENDIENTE ANTES DE ENTREGAR:** insertar aquí las dos capturas, después de ejecutar
> los comandos desde el commit final de `main`.

## MLflow Project y tracking avanzado

### 1. Servidor de tracking

El Model Registry necesita un almacén respaldado por base de datos. Desde la raíz del repositorio, con el entorno activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

El servidor queda abierto en esa terminal y la interfaz en <http://127.0.0.1:5000>. `mlflow.db` y `mlartifacts/` se crean en la raíz del repositorio y están en `.gitignore`. En una segunda terminal, también en la raíz del repositorio:

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### 2. Reproducir todo desde un clon limpio

El archivo `MLproject` declara el entorno (`python_env.yaml`) y cinco puntos de entrada: `ingest`, `baseline`, `train`, `search` y `register`. Rutas, fechas, semilla, umbrales y nombre del experimento se leen de `mibici/config.py`. La secuencia completa es:

```bash
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e ingest --run-name ingest_datos
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e baseline
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e search -P pregunta=p1
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e search -P pregunta=p2
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e search -P pregunta=p3
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e register --run-name register_champion_challenger
```

`ingest` descarga los datos y escribe el manifiesto; `baseline` registra los tres baselines; cada `search` registra una corrida padre con una corrida hija por candidato; `register` elige y registra el mejor modelo de cada pregunta sin recibir ningún identificador de corrida.

Se usa `--env-manager uv` porque el gestor por defecto de MLflow, `virtualenv`, instala la versión de Python de `python_env.yaml` con pyenv y falla si pyenv no está instalado. Con uv, MLflow descarga Python 3.10.20 y crea el entorno sin pasos manuales.

Un candidato aislado se entrena con `train`, indicando la pregunta, el modelo y, si se quieren cambiar, sus hiperparámetros:

```bash
mlflow run . --experiment-name mibici-equipo09 --env-manager uv -e train \
  -P pregunta=p1 -P modelo=ridge -P alpha=10
```

| Pregunta | `modelo` | Hiperparámetros |
| --- | --- | --- |
| P1 | `hist_gradient_boosting`, `ridge` | `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization`; `alpha` |
| P2 | `hist_gradient_boosting`, `logistic_regression` | los mismos de HistGradientBoosting; `C` |
| P3 | `kmeans`, `minibatch_kmeans`, `ward` | `n_clusters`, `batch_size` (MiniBatch), `estaciones` (`todas` o `activas`) |

Los valores por defecto de `train` son los de `mibici/config.py`; como `MLproject` no puede leer ese archivo, `tests/test_mlproject.py` falla si alguno se desalinea. MLflow registra todos los parámetros del punto de entrada en la corrida, aunque sólo apliquen los del modelo elegido; los que usa el modelo también quedan en `config/run_config.json`.

### 3. Diseño de la experimentación

| Pregunta | Candidatos de la búsqueda | Métrica de selección |
| --- | --- | --- |
| P1 | Tres `HistGradientBoostingRegressor` con distinta tasa de aprendizaje, iteraciones, hojas y regularización L2, y `Ridge` estandarizado | `val_mae`, menor es mejor |
| P2 | Tres `HistGradientBoostingClassifier` con pesos balanceados y una regresión logística estandarizada con `class_weight="balanced"` | `val_average_precision`, mayor es mejor |
| P3 | K-Means con k = 5 y k = 6, MiniBatch K-Means con k = 5, Ward con k = 5 y K-Means con k = 5 sin estaciones inactivas | `train_silhouette`, mayor es mejor, sólo entre corridas con todas las estaciones |

- **Partición.** Los candidatos de P1 y P2 se entrenan de abril a julio, se comparan en agosto (métricas `val_`) y se prueban en septiembre (métricas `test_`). Los baselines de P1 y P2 son reglas sin entrenamiento y se evalúan en las mismas dos ventanas, así que toda comparación usa los mismos datos. P3 agrupa los perfiles calculados antes del 1 de septiembre y sus métricas llevan el prefijo `train_`.
- **Selección.** Se elige por validación, nunca por prueba, para que septiembre siga siendo una medida honesta del modelo elegido.
- **Ward.** `AgglomerativeClustering` no tiene `predict`. `WardCentroides` (`mibici/modeling/clustering.py`) ajusta Ward, calcula el centroide de cada grupo y asigna cualquier estación nueva al centroide más cercano; así el modelo tiene firma y se carga desde MLflow como los demás.
- **Estaciones activas.** La corrida `P3_kmeans_k5_activas` excluye las estaciones sin salidas antes del corte (criterio `MIN_SALIDAS_ESTACION_ACTIVA` en `mibici/config.py`) y registra cuántas y cuáles excluyó. Se distingue por el parámetro `estaciones=activas`; su silhouette se calcula sobre otro conjunto de estaciones y no compite con las demás.
- **Etiquetas.** Toda corrida lleva `question`, `model_family`, `stage_of_work` (`baseline`, `search` o `candidate`), `equipo`, `git_commit` y `autor`.

Cada candidato registra:

- el modelo con firma e `input_example`;
- el manifiesto de datos en `data/` y una muestra del conjunto de entrenamiento con `mlflow.log_input`;
- gráficas en `plots/`: real contra predicho y residuales (P1), matriz de confusión y curva precisión-recall (P2), mapa y perfiles horarios de los grupos (P3);
- la configuración efectiva y `mibici/config.py` en `config/`;
- parámetros, métricas de validación y métricas de prueba.

### 4. Model Registry: aliases champion y challenger

`register` busca en el experimento las corridas con `stage_of_work=candidate` de cada pregunta, terminadas y del commit actual (etiqueta `git_commit` igual a `git rev-parse HEAD`), y las ordena por la métrica de selección de `METRICAS_SELECCION` en `mibici/config.py`. Registra la mejor con el alias `champion` y la segunda con `challenger`, en tres modelos:

| Pregunta | Modelo registrado |
| --- | --- |
| P1 | `mibici-equipo09-p1-demanda` |
| P2 | `mibici-equipo09-p2-riesgo` |
| P3 | `mibici-equipo09-p3-grupos` |

Cada modelo registrado tiene una descripción y cada versión, además de su descripción, las etiquetas `question`, `git_commit`, `data_version`, `source_run_id`, `experiment`, `selection_metric` y `selection_metric_value`. Con ellas se llega de la versión a la corrida, al commit y a la versión de datos que la produjeron. El punto de entrada es idempotente: si una versión ya proviene de la misma corrida, la reutiliza. Al final carga los tres `champion` para comprobar que se pueden usar.

Se usan aliases y no etapas porque las etapas (`Staging`, `Production`) están obsoletas en MLflow desde la versión 2.9. Un alias es un puntero con nombre que se mueve de una versión a otra sin crear versiones nuevas: promover un modelo o regresar al anterior es reasignar `champion`, y el código que consume el modelo no cambia. `challenger` deja identificada la siguiente mejor versión, para compararla con `champion` con datos nuevos antes de promoverla.

En producción, el modelo se carga por nombre y alias:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
modelo = mlflow.sklearn.load_model("models:/mibici-equipo09-p1-demanda@champion")
```

El entorno que carga el modelo debe tener instalado el paquete `mibici` (lo instala `requirements.txt`), porque el estimador de Ward vive en él.

### Baselines

| Pregunta | Baseline | Métricas principales | Artefactos |
| --- | --- | --- | --- |
| P1 | Demanda de la misma estación y hora una semana antes (`rezago_168h`) | MAE, RMSE, WAPE, R² y sesgo, en agosto (`val_`) y septiembre (`test_`) | Serie real frente a predicción, resumen horario y JSON de métricas |
| P2 | Persistencia del saldo de la hora anterior respecto de la capacidad | Precision, recall, F1, balanced accuracy y average precision, en agosto y septiembre | Matriz de confusión, riesgo diario y JSON de métricas |
| P3 | K-Means con `k=4` sobre perfiles horarios estandarizados | Silhouette, Davies-Bouldin, Calinski-Harabasz e inercia | Mapa de clusters, curvas horarias, tabla de estaciones y modelo |

La justificación de cada modelo y de cada métrica según la decisión que apoya está en [`references/modelos_metricas.md`](references/modelos_metricas.md).

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
│                            modelos, métricas, arquitectura y calidad de datos
├── reports/figures/         figuras generadas
├── MLproject                puntos de entrada de mlflow run
├── python_env.yaml          entorno de mlflow run
├── tests/                   pruebas de configuración, variables, modelos y registro
└── mibici/
    ├── config.py            rutas, URLs, fechas, partición, umbrales, semilla y MLflow
    ├── dataset.py           ingesta reproducible
    ├── features.py          ingeniería de variables
    ├── quality.py           validación de integridad y drift con DeepChecks
    ├── modeling/
    │   ├── artifacts.py     gráficas y tablas registradas en MLflow
    │   ├── baselines.py     reglas base y métricas de P1, P2 y P3
    │   ├── candidates.py    candidatos y búsquedas con nested runs
    │   ├── clustering.py    Ward con asignación por centroide
    │   ├── registry.py      selección y aliases en el Model Registry
    │   ├── train.py         CLI y tracking de los baselines
    │   └── predict.py       plantilla de Cookiecutter, sin usar
    └── plots.py             plantilla de Cookiecutter, sin usar
```

## Buenas prácticas

| Ámbito | Práctica | Cómo y dónde se cumple |
| --- | --- | --- |
| MLflow Tracking | Un experimento por proyecto y runs con nombre descriptivo | Todos los runs van al experimento `mibici-equipo09`, definido en `EXPERIMENTO` de `mibici/config.py`. `mibici/modeling/train.py` nombra los baselines `P1_baseline_rezago_semanal`, `P2_baseline_persistencia_saldo` y `P3_baseline_kmeans_k4`; `mibici/modeling/candidates.py` nombra cada búsqueda `P1_nested_search` y cada candidato por su modelo e hiperparámetros, por ejemplo `P1_hgb_lr005_iter100_leaf31` o `P3_ward_k5`. |
| MLflow Tracking | Registrar lo necesario para reproducir el run | Cada run registra como parámetros `fecha_inicio`, `fecha_fin`, `periodo_train`, `periodo_test`, `version_datos` (del manifiesto) y `semilla`, además del modelo y sus hiperparámetros: `umbral_capacidad` y `horizonte_horas` en P2; `n_clusters` y `n_init` en P3. |
| MLflow Tracking | Ligar cada run con su commit y su pregunta | Etiquetas `git_commit` (SHA de `git rev-parse HEAD`), `question` (`P1`, `P2` o `P3`), `model_family`, `stage_of_work`, `equipo` y `autor`, iguales en baselines y candidatos (`etiquetas` en `mibici/modeling/train.py`). |
| MLflow Tracking | Métricas con el conjunto en su nombre | Prefijo `val_` para agosto y `test_` para septiembre en P1 y P2 (`val_mae`, `test_recall`), y `train_` en P3, que se evalúa sobre los perfiles calculados antes del 1 de septiembre (`train_silhouette`). |
| MLflow Tracking | Sin credenciales ni datos de los usuarios de MiBici en parámetros, etiquetas o artefactos | La ingesta elimina `usuario_id`, `genero` y `anio_de_nacimiento`, las columnas con datos de los usuarios de MiBici, antes de guardar los Parquet (`mibici/dataset.py`), así que ningún artefacto puede contenerlas. La etiqueta `autor` registra tal cual el `user.name` de la configuración de Git de quien ejecuta o, si no está definido, el usuario del sistema (`obtener_autor` en `mibici/modeling/train.py`). El tracking es local y no usa credenciales; `.env` está en `.gitignore`. |
| Estructura y código | Notebooks sólo para exploración, con la convención de nombres | `notebooks/1.0-dcc-exploracion-inicial.ipynb` sigue el formato `número-iniciales-descripción` e importa las funciones de `mibici.features`; la lógica final vive en el módulo `mibici/`. |
| Estructura y código | Configuración en un solo lugar | Para el flujo de ingesta, construcción de variables y experimento, `mibici/config.py` concentra su propia ruta (que los candidatos registran como artefacto), las carpetas de datos, las rutas de los Parquet procesados, las URLs, las fechas, el corte de partición y el de validación, el retraso entre el cierre de cada ventana y la emisión de su predicción, que es también la duración mínima de un viaje válido, el umbral, el horizonte y el umbral de decisión de P2, los hiperparámetros por defecto de los modelos, el criterio de estación activa de P3, la semilla, el nombre del experimento, la URI de tracking, los nombres de los modelos registrados y la métrica de selección de cada pregunta. Los periodos que se registran en MLflow se derivan de esas constantes. El nombre local de cada CSV crudo se arma en `mibici/dataset.py` con la clave del mes. `mibici/modeling/predict.py` y `mibici/plots.py` son plantillas de Cookiecutter sin usar y conservan sus rutas de ejemplo. |
| Estructura y código | Dependencias con versión fija y entorno reproducible | `requirements.txt` fija con `==` cada paquete; la instalación con Python 3.10 está en [Instalación](#instalación). |
| Datos y Git | Ningún dato, modelo, carpeta de MLflow ni credencial en el historial | `.gitignore` excluye `/data/`, `*.parquet`, `datos_abiertos_*.csv`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `models/*`, `*.pkl`, `*.joblib`, `*.skops` y `.env`. Los datos se versionan con su hash en `references/manifiesto_datos.json`, no con los archivos. |
| Testing | Pruebas repetibles con casos normales y límite | `tests/` se ejecuta con PyTest; incluye reproducibilidad del entrenamiento con semilla fija y validaciones sintéticas que demuestran que DeepChecks detecta nulos, duplicados y drift. |
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

