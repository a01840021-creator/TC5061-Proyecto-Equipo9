# Seguridad, vulnerabilidades y riesgo residual

**Proyecto:** MiBici: demanda, riesgo de rebalanceo y patrones de uso  
**Responsable:** David Carrera Castillo (A01840136)  
**Fecha de la última corrida:** 1 de noviembre de 2026  
**Entorno de ejecución:** Python 3.10.20, entorno creado desde `requirements.txt`, macOS

## 1. Alcance

Este documento describe qué controles de seguridad tiene el repositorio, cómo se ejecutan y qué resultados dieron en la última corrida. Cubre tres frentes:

1. Configuración y datos: qué variables de entorno lee el código, qué excluye `.gitignore` y cómo la ingesta elimina los datos personales de la fuente.
2. Análisis estático del código con Bandit 1.9.4 sobre `mibici/` y `tests/`.
3. Auditoría de dependencias con pip-audit 2.10.1 sobre `requirements.txt`, con las remediaciones aplicadas y el riesgo que se acepta de forma explícita.

## 2. Configuración, secretos y datos personales

### 2.1. Variables de entorno

El código lee tres variables con `os.getenv`, y son las únicas que aparecen en [`.env.example`](../.env.example). `mibici/config.py` carga un archivo `.env` local con `python-dotenv`; si una variable no existe, se usa el valor por omisión que también aparece en la plantilla.

| Variable | Valor por omisión | Dónde se lee | Para qué sirve |
| :--- | :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | `mibici/config.py`, `mibici/modeling/candidates.py`, `mibici/modeling/registry.py` | Dirección del servidor de MLflow Tracking |
| `LOGURU_LEVEL` | `INFO` | `mibici/config.py` | Nivel mínimo de los mensajes de loguru |
| `LOGURU_COLORIZE` | `True` | `mibici/config.py` | Color en los mensajes; `0`, `false`, `no` u `off` lo desactivan |

Ninguna de las tres es un secreto. El proyecto no usa claves de API, contraseñas ni tokens. El servidor de MLflow por omisión escucha en la interfaz local (`127.0.0.1`).

### 2.2. Exclusiones en `.gitignore`

[`.gitignore`](../.gitignore) contiene, entre otros, estos patrones relacionados con secretos y datos:

- Entorno y secretos: `.env`, `.env.*` (con la excepción `!.env.example`), `credentials.json`, `client_secret*.json`, `token.json`, `secrets.json` y `.pypirc`.
- Certificados y llaves: `*.pem`, `*.key`, `*.crt`, `*.pfx` y `*.p12`.
- Entornos virtuales: `.venv`, `env/`, `venv/` y `ENV/`.
- Bases de datos y seguimiento de MLflow: `mlflow.db`, `*.db`, `db.sqlite3`, `mlruns/` y `mlartifacts/`.
- Modelos serializados: `models/*` (con la excepción `!models/.gitkeep`), `*.pkl`, `*.joblib` y `*.skops`.
- Datos: `/data/`, `*.parquet` y `datos_abiertos_*.csv`.
- Reportes locales: `reports/deepchecks/` y `reports/security/`.

### 2.3. Datos personales y minimización en la ingesta

La fuente son los datos abiertos de MiBici, que se descargan de `mibici.net` (las URL están en `mibici/config.py`). Cada archivo mensual de viajes trae ocho columnas: `Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id` y `Destino_Id`. Tres de ellas son datos personales: el identificador del usuario, su género y su año de nacimiento.

Ninguna de las tres preguntas del proyecto necesita esos datos, así que `mibici/dataset.py` los elimina durante la ingesta. La lista `COLUMNAS_PERSONALES` contiene `usuario_id`, `genero` y `anio_de_nacimiento`; esas columnas se usan solo para detectar duplicados exactos (una fila idéntica en las ocho columnas) y se quitan antes de escribir `data/processed/viajes.parquet`. El Parquet resultante tiene cinco columnas: `viaje_id`, `inicio_del_viaje`, `fin_del_viaje`, `origen_id` y `destino_id`.

En consecuencia, los datos personales no llegan a los Parquet procesados, a los conjuntos de entrenamiento, a lo que se registra en MLflow ni al repositorio. Los CSV crudos sí los contienen y se quedan en `data/raw/` de cada máquina, carpeta que `.gitignore` excluye junto con el patrón `datos_abiertos_*.csv`.

## 3. Análisis estático con Bandit

### 3.1. Configuración

El archivo [`.bandit`](../.bandit) no omite ninguna regla de forma global. Solo excluye las carpetas `.venv`, `mlruns` y `mlartifacts`. Las excepciones son puntuales:

- `B101` (`assert_used`) se omite únicamente en los archivos que coinciden con `tests/test_*.py`, mediante la opción `assert_used.skips` de Bandit. En las pruebas, `assert` es la forma en que pytest verifica resultados. En `mibici/` no hay ningún `assert`, así que la regla queda activa para todo el paquete.
- `B404`, `B603` y `B607` se omiten línea por línea con `# nosec` y el código exacto de la regla en `mibici/modeling/train.py`. Ahí `subprocess` ejecuta `git config user.name` y `git rev-parse HEAD` para registrar el autor y el commit de cada corrida en MLflow. Las dos llamadas usan una lista fija de argumentos, no usan `shell=True` y no reciben datos del usuario.

| Línea en `train.py` | Código | Comentario en la línea |
| :--- | :--- | :--- |
| `import subprocess` | `# nosec B404` | solo para `git rev-parse HEAD` y `git config user.name` |
| llamada a `git config user.name` | `# nosec B603, B607` | lista fija de argumentos, sin shell y sin datos de usuario |
| llamada a `git rev-parse HEAD` | `# nosec B603, B607` | lista fija de argumentos, sin shell y sin datos de usuario |

### 3.2. Resultados de la corrida

Comando: `bandit -c .bandit -r mibici/ tests/`. La salida completa queda en `reports/security/bandit_report.txt`.

| Métrica | Valor |
| :--- | :--- |
| Líneas de código escaneadas | 2,446 (2,094 en `mibici/` y 352 en `tests/`) |
| Hallazgos de severidad alta | 0 |
| Hallazgos de severidad media | 0 |
| Hallazgos de severidad baja | 0 |
| Hallazgos omitidos con `# nosec` y código de regla | 5 (1 de `B404`, 2 de `B603` y 2 de `B607`) |
| Líneas omitidas con `# nosec` sin código | 0 |
| Archivos omitidos | 0 |
| Código de salida | 0 |

Como referencia, la misma corrida sin `assert_used.skips` y sin los comentarios `# nosec` reporta 61 hallazgos de severidad baja: 56 de `B101`, todos en `tests/`, y los 5 de `subprocess` en `train.py`.

Bandit no escribe en la consola ningún aviso del tipo "nosec encountered, but no failed test". En las dos llamadas a `git`, la salida de `subprocess.check_output` se guarda primero en la variable `salida` y `.strip()` se aplica en una línea aparte. Así, la línea que lleva `# nosec B603, B607` contiene una sola llamada, que es la que Bandit señala.

## 4. Auditoría de dependencias con pip-audit

### 4.1. Cómo se audita

La auditoría se hace sobre `requirements.txt`, no sobre el entorno local, para que el resultado no dependa de lo que cada quien tenga instalado:

```bash
pip-audit -r requirements.txt --no-deps --disable-pip
```

`requirements.txt` fija con `==` la versión exacta de cada paquete instalado en el entorno, incluidas las dependencias transitivas de Bandit y pip-audit, así que pip-audit puede leerlo sin resolver dependencias y sin crear un entorno temporal. La línea `-e .` aparece en la salida como omitida ("could not deduce package version from URL requirement") porque corresponde al propio proyecto, no a un paquete publicado; no impide la auditoría del resto.

El modo con resolución de dependencias (`pip-audit -r requirements.txt` sin `--no-deps`) no se pudo usar en macOS con el Python 3.10.20 que instala uv: pip-audit crea un entorno temporal y el paso `ensurepip` de ese entorno termina con `SIGABRT`.

### 4.2. Remediación de versiones anteriores

Las versiones anteriores se auditaron con un archivo temporal con `certifi==2026.7.22`, `mlflow==3.17.1`, `mlflow-skinny==3.17.1`, `mlflow-tracing==3.17.1` y `pyasn1==0.6.4`. pip-audit reportó seis vulnerabilidades, todas corregidas por las versiones que hoy fija `requirements.txt`:

| Paquete | Versión anterior | Versión actual | Identificador (alias) | Versión que corrige | Descripción del aviso |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `certifi` | 2026.7.22 | 2026.10.23 | PYSEC-2026-4320 (GHSA-6q48-6c7w-9gw2) | 2026.10.18 | El almacén incluía la raíz `emSign ECC Root CA - G3`, de la que dependen certificados intermedios con llaves privadas comprometidas y ya revocados |
| `mlflow` | 3.17.1 | 3.17.2 | CVE-2026-53995 (GHSA-wqmh-qccw-pqwf) | 3.17.2 | Recorrido de rutas en el backend FileStore de tracking mediante un identificador de traza manipulado enviado a `POST /api/3.0/mlflow/traces`, sin autenticación |
| `pyasn1` | 0.6.4 | 0.6.5 | CVE-2026-119194 (GHSA-fm9c-5rcf-2q5p), CVE-2026-119195 (GHSA-9m75-vc3g-4x39), CVE-2026-119196 (GHSA-2qm8-x737-gmw5), CVE-2026-119197 (GHSA-mm57-4v3p-rchx) | 0.6.5 | Consumo de CPU que crece de forma cuadrática al decodificar estructuras ASN.1 controladas por un atacante, lo que puede agotar el procesador |

`mlflow-skinny` y `mlflow-tracing` 3.17.1 no tienen avisos; se actualizaron a 3.17.2 junto con `mlflow`. Con MLflow 3.17.2 instalado, `ruff check`, `ruff format --check` y las 40 pruebas de `python -m pytest -q tests` pasan.

### 4.3. Riesgo residual aceptado

La auditoría actual de `requirements.txt` encuentra 59 vulnerabilidades conocidas en tres paquetes. En los tres, la primera versión que las corrige exige Python 3.11 o superior (`requires_python >=3.11` en PyPI), y el proyecto fija Python 3.10, así que no se pueden actualizar sin cambiar la versión de Python de todo el entorno. En el caso de `filelock`, la versión fijada (4.1.0) es la última publicada en PyPI que admite Python 3.10; desde 4.1.1 todas exigen Python 3.11 o superior.

| Paquete | Versión fijada | Versión que corrige | Requisito de Python de la versión que corrige | Vulnerabilidades | Quién lo instala | Superficie de ataque en el proyecto | Nivel de riesgo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pillow` | 12.3.0 | 13.0.0 | Python 3.11 o superior | 39 | `matplotlib` | Abrir o decodificar imágenes de origen externo. El código no abre imágenes | Bajo |
| `tornado` | 6.5.10 | 6.6 | Python 3.11 o superior | 3 | `jupyter-server`, `jupyterlab`, `notebook`, `ipykernel`, `jupyter-client`, `terminado` | Servidor HTTP de Jupyter. Solo corre al levantar Jupyter en la máquina local | Bajo |
| `filelock` | 4.1.0 | 4.2.1 | Python 3.11 o superior | 17 | `pip-audit`, mediante `cachecontrol[filecache]` | Archivos de bloqueo de la caché HTTP de pip-audit. Solo corre durante la auditoría y la carpeta pertenece al usuario | Bajo |

El nivel de riesgo se asigna según dos criterios: si el código del proyecto llega a la función vulnerable y qué acceso necesita un atacante para explotarla. Los tres casos son bajos porque el código de `mibici/` no usa la función afectada y la explotación requiere entregar al entorno una imagen preparada (`pillow`), alcanzar un servidor de Jupyter que por omisión escucha solo en la interfaz local (`tornado`) o tener otra cuenta en la misma máquina (`filelock`).

Identificadores de `pillow`: PYSEC-2026-4326 a PYSEC-2026-4343 (18 identificadores consecutivos), CVE-2026-96638, CVE-2026-96640, CVE-2026-96778, CVE-2026-112390, CVE-2026-112391, CVE-2026-112393, CVE-2026-112395, CVE-2026-112399, CVE-2026-112400, CVE-2026-112401, CVE-2026-112404, CVE-2026-112406, CVE-2026-112891, CVE-2026-112892, CVE-2026-112893, CVE-2026-112894, CVE-2026-112895, CVE-2026-112897, CVE-2026-112899, CVE-2026-112900 y GHSA-78g4-xqwv-q3g8.

Identificadores de `tornado`: CVE-2026-115927, CVE-2026-116577 y CVE-2026-116578.

Identificadores de `filelock`: CVE-2026-112906 a CVE-2026-112915 (10 identificadores consecutivos), CVE-2026-112934, CVE-2026-112935 y CVE-2026-112937 a CVE-2026-112941 (5 identificadores consecutivos).

La aceptación es explícita y revisable: cada identificador está escrito en el `Makefile`, en las variables `VULNS_ACEPTADAS_PILLOW`, `VULNS_ACEPTADAS_TORNADO` y `VULNS_ACEPTADAS_FILELOCK`, y `make security` los pasa a pip-audit con `--ignore-vuln`. Así el comando termina en éxito ("No known vulnerabilities found, 59 ignored") y falla si encuentra una vulnerabilidad cuyo identificador no está en la lista de excepciones. Las excepciones deben revisarse cuando cambien las versiones de las dependencias. `make security-report` genera `reports/security/pip_audit_report.md` sin esas omisiones, para que el reporte liste también lo aceptado.

Análisis de superficie de ataque:

- `pillow`. Según las descripciones de los avisos, se activan al abrir o decodificar archivos de imagen preparados por un atacante (PNG, TIFF, GIF, BMP, PDF, EPS, JPEG 2000 y otros) o al guardar en TIFF, PDF o JPEG a partir de esas imágenes. El código de `mibici/` no importa `PIL` ni abre imágenes; solo crea figuras con matplotlib y las guarda como PNG con `savefig` en `mibici/modeling/artifacts.py`.
- `tornado`. Los avisos afectan al servidor HTTP: el analizador de encabezados, la lectura de cuerpos con codificación por fragmentos y `RedirectHandler`. Tornado solo corre cuando alguien levanta Jupyter para explorar un notebook; el entrenamiento, la validación y el registro de modelos se ejecutan como scripts de Python y no lo usan.
- `filelock`. El código de `mibici/` y `tests/` no lo importa. En el entorno, fuera de su propio paquete, solo lo importa `cachecontrol/caches/file_cache.py`, la caché HTTP que usa pip-audit; ahí crea un `FileLock` (en macOS y Linux, `UnixFileLock`) sobre archivos `.lock` dentro de la carpeta de caché de pip-audit, que en macOS es `~/Library/Caches/pip-audit` y en la máquina de la corrida tiene permisos `drwx------`. Los 17 avisos tienen vector de ataque local (`AV:L` en su puntuación CVSS 3.1): requieren otra cuenta en la misma máquina con permiso para crear, renombrar o enlazar archivos en la carpeta del bloqueo. La mayoría afecta a clases que esa caché no crea (`SoftFileLock`, `MarkerSoftFileLock`, `SoftFileLease`, `StrictSoftFileLock`, `SoftReadWriteLock`, `ReadWriteLock`) o solo a Windows. Algunos de esos alcanzan a `FileLock` cuando cae a un bloqueo suave porque el sistema de archivos no implementa `flock` (error `ENOSYS`). El que afecta directamente a `UnixFileLock` (CVE-2026-112909) requiere que otro usuario pueda renombrar o enlazar un archivo de quien ejecuta pip-audit sobre la ruta del bloqueo.

Control compensatorio: no abrir con el entorno del proyecto imágenes de origen externo, no exponer Jupyter fuera de la máquina local y no compartir con otras cuentas la carpeta de caché de pip-audit ni redirigirla con `--cache-dir` a una carpeta con permisos de escritura para otros usuarios. El riesgo se resuelve al migrar el proyecto a Python 3.11 o superior, actualizar `pillow`, `tornado` y `filelock`, y retirar los identificadores del `Makefile`.

## 5. Relación con marcos de seguridad en IA

### 5.1. OWASP Machine Learning Security Top 10

- ML01, manipulación de entradas. El proyecto no tiene defensas contra ataques de evasión. Lo que sí hace es validar los datos de entrada: la ingesta en `mibici/dataset.py` falla si falta alguna columna esperada, si un identificador tiene nulos o no es entero y si una fecha no tiene el formato `%Y-%m-%d %H:%M:%S`; también advierte, sin filtrar, cuando un viaje inicia fuera de la ventana de análisis. `mibici/features.py` descarta los viajes espurios de menos de un minuto que terminan donde empezaron, y `mibici/quality.py` corre las suites `data_integrity` y `train_test_validation` de DeepChecks sobre muestras deterministas.
- ML02, envenenamiento de datos. `references/manifiesto_datos.json` guarda la URL, el número de registros y el SHA-256 de cada archivo crudo descargado, y una `version_datos` derivada de esos hashes. Cada corrida de baselines y de candidatos registra `version_datos` como parámetro en MLflow, y cada corrida de candidatos adjunta además el manifiesto como artefacto. Esto permite saber con qué archivos crudos se entrenó cada modelo y detectar si cambian; no detecta datos alterados en el origen antes de la descarga.
- ML03 y ML04, inversión de modelo e inferencia de pertenencia. Los modelos se entrenan sin datos personales (sección 2.3) y emiten viajes por estación y hora (P1), probabilidades de riesgo por estación y hora (P2) y etiquetas de grupo por estación (P3). No hay datos de personas que un modelo pueda revelar.
- ML06, ataques a la cadena de suministro. `requirements.txt` fija versiones exactas y `make security` audita esas versiones con pip-audit, con el riesgo aceptado declarado por identificador (sección 4.3).

### 5.2. NIST AI Risk Management Framework (AI RMF 1.0)

- GOVERN. Los cambios entran a `main` por pull request, y la tabla de contribuciones del README registra quién hizo cada parte y qué revisiones aprobó.
- MAP. Este documento enumera las dependencias con vulnerabilidades conocidas, quién las instala y qué parte del proyecto las usa.
- MEASURE. Bandit reporta 0 hallazgos con 5 omisiones justificadas por línea; pip-audit reporta 0 vulnerabilidades fuera de las 59 aceptadas; pytest aprueba 40 de 40 pruebas; DeepChecks genera reportes de integridad y drift.
- MANAGE. El riesgo residual está declarado identificador por identificador en el `Makefile`, con su control compensatorio y la condición para retirarlo.

## 6. Reproducción

Desde la raíz del repositorio, con el entorno activado:

```bash
# Bandit y pip-audit, con el riesgo aceptado (termina en éxito)
make security

# Por separado
bandit -c .bandit -r mibici/ tests/
pip-audit -r requirements.txt --no-deps --disable-pip   # lista también las 59 aceptadas

# Reportes en reports/security/
make security-report
```

`make security-report` escribe `reports/security/bandit_report.txt` y `reports/security/pip_audit_report.md`. La carpeta está en `.gitignore`, así que los reportes se regeneran en cada máquina.
