# Resultados de calidad de datos con DeepChecks

## Alcance y reproducción

Los resultados de este documento salen de una corrida del 1 de noviembre de 2026 sobre la versión de datos `0c4b34e5425a` de `references/manifiesto_datos.json`:

```bash
python -m mibici.quality --pregunta all
```

El comando valida cuatro conjuntos y escribe un HTML por suite y un `resumen.json` por conjunto en `reports/deepchecks/<conjunto>/`. Cada opción también se puede correr por separado con `--pregunta p1`, `p2`, `p3` o `base`.

| Conjunto | Tabla que se valida | Suites | Filas evaluadas por DeepChecks |
| --- | --- | --- | --- |
| P1 | Tablón de demanda, partido en entrenamiento y prueba por `FECHA_CORTE` | `data_integrity` en cada partición y `train_test_validation` | 5,000 de entrenamiento y 5,000 de prueba |
| P2 | Tablón de riesgo, con la misma partición | Las mismas que P1 | 5,000 de entrenamiento y 5,000 de prueba |
| P3 | Perfiles de estaciones de `generar_dataset_p3(..., solo_train=True)`, la tabla que agrupan los modelos de P3, con sus 26 variables | `data_integrity` más `Percent Of Nulls` | Las 484 estaciones |
| Base | Viajes y nomenclatura de `data/raw/`, y `viajes.parquet` y `estaciones.parquet` de `data/processed/` | `data_integrity` más `Percent Of Nulls` en cada una de las cuatro tablas | 5,000 viajes crudos de 2,294,886, las 484 estaciones crudas, 5,000 viajes procesados de 2,290,174 y las 484 estaciones procesadas |

- **Muestras.** Cuando una tabla tiene más filas que `--max-filas` (5,000 por omisión), DeepChecks recibe una muestra de `muestrear_deterministicamente` con `SEMILLA=42`. Dos corridas sobre la misma versión de datos evalúan las mismas filas.
- **Conteos sobre todos los datos.** Los duplicados exactos, los nulos, los valores que no se pueden tipar y los viajes con estaciones fuera de la nomenclatura se cuentan con pandas sobre todas las filas, en la sección `diagnostico` de cada `resumen.json`. Una muestra de 5,000 viajes casi nunca contiene los dos ejemplares de un duplicado: con 758 duplicados entre 2,294,886 viajes, se esperan menos de 0.01 pares en la muestra.
- **Relación entre viajes y estaciones.** DeepChecks valida una tabla a la vez y no tiene un chequeo de integridad referencial. `verificar_estaciones_de_viajes` en `mibici/quality.py` aplica la regla de `quitar_sin_estacion` de la ingesta: un viaje queda fuera si su origen o su destino no es un `id` de la nomenclatura.
- **Nulos.** La suite `data_integrity` sólo revisa si una columna mezcla tipos de nulo (`Mixed Nulls`). Para P3 y el dataset base se agrega `Percent Of Nulls` con un umbral de 0 %.
- **Datos personales.** En los crudos, `usuario_id`, `genero` y `anio_de_nacimiento` se usan sólo para contar duplicados exactos con las ocho columnas, como en la ingesta, y se quitan antes de muestrear y de construir el `Dataset` de DeepChecks. De esas columnas el resumen sólo da el número total de celdas vacías; ningún reporte contiene sus valores.
- **Chequeos que no se ejecutan.** `Conflicting Labels`, `Feature Label Correlation` e `Identifier Label Correlation` requieren una etiqueta, y P3 y el dataset base no la tienen. En P1 y P2, `Index Train Test Leakage` no se ejecuta porque los tablones no tienen índice, y en P1 `New Label Train Test` no aplica a regresión. Cada resumen los lista en `chequeos_no_ejecutados` con el motivo que da DeepChecks.

### Corrección de compatibilidad con NumPy 2

DeepChecks 0.19.1 usa `np.NaN`, que NumPy 2.0 retiró. `componentes_deepchecks` ya reponía `np.Inf` pero no `np.NaN`, así que `Feature Feature Correlation` fallaba al ejecutarse en todas las suites de integridad. El resumen anterior sólo listaba los chequeos no aprobados y ese fallo no aparecía. Ahora `componentes_deepchecks` repone también `np.NaN`, el chequeo se ejecuta y el resumen incluye `chequeos_aprobados` y `chequeos_no_ejecutados`. Los resultados de `Feature Feature Correlation` de este documento son los primeros que produce el proyecto.

## Dataset base: lo que corrige la ingesta

### Conteos sobre todos los datos

| Revisión | Crudos (`data/raw/`) | Procesados (`data/processed/`) |
| --- | --- | --- |
| Viajes | 2,294,886 | 2,290,174 |
| Duplicados exactos | 758, con las ocho columnas del CSV | 0, con las cinco columnas que quedan |
| `viaje_id` repetidos | 758 | 0 |
| Viajes con estación fuera de la nomenclatura | 3,954 | 0 |
| Viajes con origen fuera / con destino fuera | 1,960 / 2,123 | 0 / 0 |
| Estaciones ausentes de la nomenclatura | 405 (3 viajes) y 407 (3,951 viajes) | ninguna |
| Nulos en `viaje_id`, fechas, `origen_id` y `destino_id` | 0 | 0 |
| Celdas vacías en las tres columnas personales | 14,629 | las columnas no existen |
| Identificadores no enteros y fechas con otro formato | 0 | 0; los tipos son `int64` y `datetime64[ns]` |
| Estaciones | 484 | 484 |
| Estaciones duplicadas, con `id` repetido o con nulos | 0 | 0 |
| Estaciones con capacidad cero | 4 | 4 |

Interpretación y tratamiento:

- **Duplicados exactos.** Los 758 duplicados coinciden en las ocho columnas, y hay exactamente 758 `viaje_id` repetidos en los crudos y ninguno en los procesados. Es decir, cada viaje repetido es una copia exacta de otro y no hay dos viajes distintos con el mismo identificador. La ingesta los elimina con `drop_duplicates` sobre `COLUMNAS_VIAJES`.
- **Estaciones fuera de la nomenclatura.** 3,954 viajes salen de o llegan a las estaciones 405 o 407, que no están en `nomenclatura_2026_09.csv`. La suma de orígenes y destinos fuera (4,083) es mayor que 3,954 porque 129 viajes tienen las dos puntas fuera. La ingesta los descarta con `quitar_sin_estacion`, porque sin la nomenclatura no hay capacidad ni ubicación para esas estaciones.
- **Nulos.** Las columnas que usa el proyecto no tienen nulos. Las 14,629 celdas vacías están en las columnas personales, que la ingesta elimina, así que no afectan a ningún modelo.
- **Tipos.** Ningún identificador deja de ser entero y ninguna fecha deja de cumplir `FORMATO_FECHA` (`%Y-%m-%d %H:%M:%S`), así que la conversión de tipos de la ingesta no falla con estos archivos. Los Parquet guardan los identificadores como `int64` y las fechas como `datetime64[ns]`.
- **Capacidad cero.** Cuatro estaciones tienen `dpcapacity` igual a 0 en las dos versiones. La ingesta las conserva, porque P1 y P3 sí las usan, y `generar_dataset_p2` las excluye de P2, donde el umbral de riesgo depende de la capacidad.

### Resultados de DeepChecks

| Tabla | Chequeos aprobados | Chequeos no aprobados |
| --- | --- | --- |
| Viajes crudos (muestra de 5,000) | 9: `Is Single Value`, `Special Characters`, `Mixed Nulls`, `Mixed Data Types`, `String Mismatch`, `Data Duplicates`, `String Length Out Of Bounds`, `Feature Feature Correlation` y `Percent Of Nulls` | ninguno |
| Estaciones crudas (484) | 8: los mismos salvo `Feature Feature Correlation` | `Feature Feature Correlation` |
| Viajes procesados (muestra de 5,000) | 9, los mismos que en los crudos | ninguno |
| Estaciones procesadas (484) | 8, los mismos que en las crudas | `Feature Feature Correlation` |

- **Viajes.** La muestra de crudos aprueba `Data Duplicates` porque no contiene ningún par duplicado; los 758 duplicados se ven en el conteo sobre todos los datos. Las dos muestras aprueban `Percent Of Nulls` y `Mixed Data Types`, lo que coincide con los conteos de nulos y de tipos.
- **Estaciones.** `Feature Feature Correlation` reporta una correlación mayor que 0.9 de `name` con `latitude`, `longitude` y `dpcapacity`. `name` tiene un valor distinto por estación, así que como variable categórica determina cualquier otra columna de la misma fila; la correlación viene de tratar un identificador como variable, no de columnas redundantes. Ningún modelo usa `name`, así que no requiere tratamiento.
- **Lo que DeepChecks no detecta en la nomenclatura.** 14 estaciones tienen el prefijo "ex-" en el nombre y ninguna registra viajes (ver la sección de datos del README). `String Mismatch`, que busca variantes de escritura de un mismo valor, aprueba, así que el prefijo no aparece como hallazgo de integridad. Se conservan en P3 como estaciones sin salidas.

## P1: predicción de demanda

`data_integrity` aprobó en las dos particiones ocho chequeos: `Is Single Value`, `Special Characters`, `Mixed Nulls`, `Mixed Data Types`, `String Mismatch`, `Data Duplicates`, `String Length Out Of Bounds` y `Feature Label Correlation`. No aprobó tres:

- **Conflicting Labels.** El 0.08 % de la muestra de entrenamiento y el 0.12 % de la de prueba son filas con los mismos valores en todas las variables y distinta demanda. No son duplicados (`Data Duplicates` aprobó). Una estación puede tener distinta demanda en dos horas con la misma capacidad, calendario y rezagos, así que es un rasgo de la variable objetivo y no un error de los datos.
- **Feature Feature Correlation.** `dpcapacity` y `estacion_id` tienen una correlación mayor que 0.9 en las dos particiones. La capacidad es un atributo fijo de cada estación, así que `estacion_id`, que la suite trata como categórica, la determina por construcción. Los dos modelos de P1 toleran variables redundantes (los árboles de HistGradientBoosting y Ridge con regularización L2), así que no se elimina ninguna.
- **Identifier Label Correlation.** `fecha_hora` tiene un PPS de 0.01 con la demanda en la muestra de entrenamiento y de 0.08 en la de prueba. La fecha ordena y divide los datos, pero no es variable del modelo (`FEATURES_P1`), así que no hay fuga.

`train_test_validation` aprobó los diez chequeos que se ejecutaron. Entre ellos, `Feature Drift` (puntaje menor que 0.2 en cada variable), `Label Drift` (menor que 0.15) y `Multivariate Drift` (menor que 0.25) no encontraron un cambio entre abril a agosto y septiembre que supere sus umbrales, y `Date Train Test Leakage Duplicates`, `Date Train Test Leakage Overlap` y `Train Test Samples Mix` confirmaron que ninguna fecha ni fila de prueba aparece en el entrenamiento.

## P2: clasificación de riesgo

`data_integrity` aprobó en las dos particiones diez chequeos: los ocho de P1 más `Conflicting Labels` e `Identifier Label Correlation`. El único no aprobado es `Feature Feature Correlation`, por el mismo par `dpcapacity` y `estacion_id` de P1 y con la misma interpretación; tampoco se elimina ninguna variable.

`train_test_validation` aprobó los once chequeos que se ejecutaron, incluidos `New Label Train Test`, `Feature Drift`, `Label Drift`, `Multivariate Drift` y los de fuga por fecha.

## P3: perfiles de estaciones

DeepChecks evaluó las 484 estaciones con las 26 variables del agrupamiento y aprobó nueve chequeos: `Is Single Value`, `Special Characters`, `Mixed Nulls`, `Mixed Data Types`, `String Mismatch`, `Data Duplicates`, `String Length Out Of Bounds`, `Feature Feature Correlation` y `Percent Of Nulls`.

| Revisión sobre los 484 perfiles | Resultado |
| --- | --- |
| Estaciones con más de un perfil | 0 |
| Nulos en las variables | 0 |
| Proporciones fuera de [0, 1] | 0 |
| Perfiles repetidos | 15 |
| Estaciones sin salidas antes del corte | 16 |
| Perfiles repetidos entre estaciones con salidas | 0 |
| Perfiles de estaciones fuera de la nomenclatura | 0 |
| Estaciones de la nomenclatura sin perfil | 0 |

Las 16 estaciones sin salidas antes del corte tienen el mismo perfil: proporciones horarias y de fin de semana en cero y `ratio_salidas` de 0.5. Eso produce las 15 filas repetidas (16 menos la primera) y no hay repetidos entre las estaciones con salidas. `Data Duplicates` aprueba porque su condición por omisión tolera hasta 5 % y los repetidos son el 3.1 %. El tratamiento ya existe: la corrida `P3_kmeans_k5_activas` excluye esas estaciones con `MIN_SALIDAS_ESTACION_ACTIVA`, y el Model Registry sólo compara por silhouette las corridas con todas las estaciones.

## Tratamiento de los reportes

Los HTML y los resúmenes se generan en `reports/deepchecks/`, que está en `.gitignore`. Los HTML pueden mostrar filas de las muestras, sin columnas personales. Los resúmenes JSON sólo contienen nombres de chequeos, condiciones, conteos y nombres de columnas. Los chequeos deben repetirse cuando cambie la versión de datos o antes de promover una nueva versión de un modelo. Las pruebas de `tests/test_calidad.py` comprueban con datos sintéticos que estas verificaciones detectan duplicados, nulos y estaciones fuera de la nomenclatura, que no reportan nada con datos limpios y que ningún valor de las columnas personales llega a los HTML ni al resumen.
