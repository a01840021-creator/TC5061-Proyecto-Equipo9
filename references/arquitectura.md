# Arquitectura del Sistema

Proyecto MiBici, equipo 9. Este documento describe los componentes del sistema para las tres preguntas del proyecto, el flujo de datos entre etapas y las herramientas utilizadas, con base en las lecturas de la semana (Sculley et al., 2015; Fowler, 2019).

## Diagrama de componentes

```mermaid
flowchart TD
    subgraph Fuentes["1. Fuentes de datos"]
        A1["Portal de datos abiertos de MiBici<br/>(abril a septiembre de 2026)"]
        A2["Nomenclatura de estaciones<br/>(septiembre de 2026)"]
    end

    subgraph Ingesta["2. Ingesta y versionado (mibici.dataset)"]
        B1["Descarga en streaming y validación de columnas"]
        B2["Normalización a snake_case y tipado"]
        B3["Deduplicación y descarte de estaciones 405 y 407"]
        B4["Eliminación de usuario_id, genero y anio_de_nacimiento"]
        B5["Hashes SHA-256 y version_datos"]
    end

    subgraph Datos["3. Almacenamiento local"]
        C1[("data/raw/<br/>CSVs originales")]
        C2[("data/processed/<br/>viajes.parquet y estaciones.parquet")]
    end

    subgraph Features["4. Variables y partición (mibici.features)"]
        D1["Filtro de viajes espurios (&lt;1 min mismo origen y destino)"]
        D2["P1: Grid cartesiano estación-hora, ceros explícitos y rezagos"]
        D3["P2: Saldo neto futuro a 2h, medias móviles y rezagos en t"]
        D4["P3: Distribución horaria, fin de semana y ratio de flujo"]
        D5["Partición temporal: Train abr-ago | Test sep"]
    end

    subgraph Modelos["5. Modelado y MLflow (mibici.modeling)"]
        E1["P1 Regresión: rezago semanal vs Ridge con rezagos"]
        E2["P2 Clasificación: persistencia de saldo vs Regresión Logística balanceada"]
        E3["P3 Clustering: KMeans sobre perfiles horarios estandarizados"]
        E4["MLflow Tracking: experimento mibici-equipo09 con tags, métricas y artefactos"]
    end

    A1 & A2 --> B1 --> B2 --> B3 --> B4 --> C2
    B1 --> C1
    B4 --> B5
    C2 --> D1 --> D2 & D3 & D4 --> D5 --> E1 & E2 & E3 --> E4
```

## Herramientas y justificación

| Componente | Herramienta | Justificación |
| --- | --- | --- |
| Ingesta | `requests`, `typer`, `hashlib` | Automatiza la descarga por streaming de archivos grandes y genera hashes reproducibles sin pasos manuales. |
| Almacenamiento | Parquet (`pyarrow`) | Reduce el peso de los viajes de ~200 MB a 43 MB, conserva tipos de datos y acelera la lectura de columnas para análisis temporal. |
| Variables | `pandas`, `numpy` | Permite operaciones vectorizadas sobre series de tiempo (shift, rolling) y generación del producto cartesiano estación-hora para completar ceros. |
| Modelado | `scikit-learn` | Implementa baselines estándar, escalamiento y particiones sin introducir dependencias pesadas innecesarias en la fase inicial. |
| Registro de experimentos | `mlflow` | Centraliza parámetros, métricas con prefijo de partición y artefactos visuales, ligando cada run con el commit de Git y la versión de datos. |

## Prevención de fuga de información

- **Partición temporal:** entrenamiento del 1 de abril al 31 de agosto de 2026 y prueba en septiembre de 2026.
- **P1:** las variables predictoras usan rezagos de al menos 24 horas ($t-24\text{h}, t-48\text{h}, t-168\text{h}$) y variables de calendario, disponibles al cierre del día anterior.
- **P2:** las variables de la hora $t$ usan solo datos observados hasta $t$. Las dos últimas horas de agosto se descartan de entrenamiento para que la ventana futura de 2 horas no cruce a septiembre.
- **P3:** los perfiles y el agrupamiento se ajustan únicamente con los datos del periodo de entrenamiento.
