# Modelos y métricas

Proyecto MiBici, equipo 9. Para cada pregunta, este documento define el baseline, los modelos candidatos y las métricas de evaluación, y justifica cada métrica por la decisión que toma quien usa el resultado. Los baselines están en `mibici/modeling/baselines.py` y `mibici/modeling/train.py`; los candidatos, en `mibici/modeling/candidates.py`; la selección y el registro, en `mibici/modeling/registry.py`. Todos se registran en el experimento `mibici-equipo09` de MLflow con la versión de datos `0c4b34e5425a`.

## Resumen

| Pregunta | Baseline | Candidatos | Métricas | Selección |
| --- | --- | --- | --- | --- |
| P1, regresión | Rezago semanal: la demanda de la misma estación y hora una semana antes | Tres HistGradientBoosting y Ridge estandarizado | `mae`, `rmse`, `wape`, `r2`, `bias` | Menor `val_mae` |
| P2, clasificación | Persistencia: el saldo de la hora previa respecto de la capacidad | Tres HistGradientBoosting con pesos balanceados y regresión logística balanceada | `recall`, `precision`, `f1`, `balanced_accuracy`, `average_precision` | Mayor `val_average_precision` |
| P3, agrupamiento | K-Means con k = 4 sobre perfiles estandarizados | K-Means con k = 5 y k = 6, MiniBatch K-Means con k = 5, Ward con k = 5 y K-Means con k = 5 sin estaciones inactivas | `train_silhouette`, `train_davies_bouldin`, `train_calinski_harabasz`, `train_inercia` | Mayor `train_silhouette` entre las corridas con las 484 estaciones |

El prefijo de cada métrica indica el conjunto en que se calcula. En P1 y P2, los candidatos se entrenan de abril a julio; `val_` corresponde a agosto y `test_` a septiembre. Los baselines de P1 y P2 son reglas sin entrenamiento y se evalúan en las mismas dos ventanas, de modo que la comparación usa exactamente las mismas filas. En P3, `train_` corresponde a los perfiles calculados antes del 1 de septiembre. El modelo de cada pregunta se elige con la métrica de validación, nunca con la de prueba, para que septiembre siga siendo una medida independiente del modelo elegido.

Cada predicción se emite al cierre de su ventana más `RETRASO_DISPONIBILIDAD_SEG` (60 segundos, en `mibici/config.py`): al cierre del día anterior en P1, al cierre de cada hora en P2 y al corte del 1 de septiembre en P3. Es la misma constante que define un viaje espurio, así que a la emisión ya se sabe cuáles de las últimas salidas fueron préstamos fallidos y las variables no usan información posterior. El detalle está en `references/definicion_datos.md`.

## P1. Viajes que inician por estación y hora

**Decisión que apoya.** Al cierre de cada día, el operador decide cuántas bicicletas dejar en cada estación para el día siguiente y en qué franjas reforzar la operación.

**Baseline.** Predice la demanda de cada estación y hora con la observada en la misma estación y hora siete días antes (`rezago_168h`). No aprende nada, pero captura el ciclo semanal, que es la regularidad más fuerte de los datos: en el periodo de prueba su correlación con la demanda es 0.71, contra 0.63 del rezago de 24 horas.

**Candidatos.** Todos usan las mismas variables: los tres rezagos (24, 48 y 168 horas), la hora del día, el día de la semana, el indicador de fin de semana, la capacidad y el identificador de la estación. No se usa el mes: entrenamiento, validación y prueba cubren meses distintos, así que el modelo aprendería un efecto de valores que nunca vuelve a ver.

- Tres `HistGradientBoostingRegressor` que varían la tasa de aprendizaje (0.05 y 0.10), las iteraciones (100 a 200), las hojas por árbol (31 y 63) y la regularización L2 (0 a 1). Captan interacciones no lineales entre hora, día y rezagos sin codificarlas a mano.
- `Ridge` con las variables estandarizadas (`StandardScaler` y `Ridge` en un `Pipeline`) y `alpha` como parámetro. Es la alternativa lineal: combina los rezagos en lugar de elegir uno, la penalización L2 estabiliza los coeficientes cuando los rezagos están correlacionados entre sí, se ajusta en segundos y sus coeficientes se pueden leer. Las variables entran como numéricas, sin codificar la hora ni el día como categorías, y las predicciones no se recortan en cero.

**Métricas.**

| Métrica | Qué mide | Por qué sirve a la decisión |
| --- | --- | --- |
| `test_mae` | Error absoluto medio, en viajes por estación y hora | Está en la unidad con la que razona el operador: cuántas bicicletas sobraron o faltaron en promedio. Es la métrica principal para comparar baseline y candidato. |
| `test_rmse` | Raíz del error cuadrático medio, en viajes | Castiga más los errores grandes, que se concentran en las horas pico, donde quedarse corto deja usuarios sin bicicleta. Si el RMSE crece mucho más que el MAE, el modelo falla justo en esas horas. |
| `test_wape` | Suma de errores absolutos entre la suma de la demanda real | Expresa el error como fracción del volumen. El 61.9 % de las combinaciones estación y hora vale cero, así que un error porcentual por observación (como el MAPE) no está definido; el WAPE sí, y permite comparar meses con volúmenes distintos, como septiembre, con 24.3 % más viajes que agosto. |
| `test_r2` | Proporción de la varianza de la demanda que explica el modelo | Sirve de contexto para saber cuánto de la variación entre estaciones y horas se captura, aunque no se traduce directamente en bicicletas. |
| `test_bias` | Promedio de predicción menos real, en viajes | El signo importa: un sesgo negativo subestima la demanda y deja estaciones sin bicicletas, que es el error más costoso para el servicio; uno positivo deja bicicletas ociosas, que cuesta traslados. |

Cada métrica se registra para agosto (`val_mae`) y para septiembre (`test_mae`). El candidato que pasa al Model Registry es el de menor `val_mae`.

**Resultado del baseline (septiembre).** MAE 0.949, RMSE 1.970, WAPE 0.701, R² 0.445 y sesgo de menos 0.112 viajes. El sesgo negativo es coherente con el aumento de viajes de septiembre: la semana previa subestima el mes siguiente.

## P2. Riesgo de vaciado o llenado en las próximas dos horas

**Decisión que apoya.** Cada hora, el equipo de rebalanceo decide a qué estaciones mandar el camión y si debe llevar o retirar bicicletas.

**Baseline.** Regla de persistencia: calcula el valor absoluto del saldo de la hora previa entre la capacidad y marca riesgo si alcanza el 30 %. Supone que el desbalance reciente continúa.

**Candidatos.** Todos usan las variables del tablón de P2: salidas, llegadas y saldo de la hora previa, saldo medio de las últimas tres horas, hora, día de la semana, fin de semana, capacidad e identificador de la estación. Todos compensan que la clase de riesgo sea una fracción pequeña del entrenamiento y entregan una probabilidad; la alerta se emite cuando la probabilidad alcanza `UMBRAL_DECISION_P2` (0.5), y ese umbral se puede ajustar después al número de camiones disponibles.

- Tres `HistGradientBoostingClassifier` con los mismos hiperparámetros que en P1 y pesos de muestra balanceados (`compute_sample_weight`).
- Regresión logística con `class_weight="balanced"`, variables estandarizadas con un escalador ajustado sólo en entrenamiento y `C` como parámetro. Es la alternativa lineal: sus coeficientes muestran qué variables empujan el riesgo.

El candidato que pasa al Model Registry es el de mayor `val_average_precision`, porque evalúa la probabilidad en todos los umbrales y no depende del 0.5 elegido. La recuperación sigue siendo la métrica que traduce el error más caro a la operación, una vez fijado el umbral.

**Métricas.** El costo de los errores no es simétrico. Un falso negativo es una estación que se vacía o se llena sin que nadie la atienda: se pierden viajes o el usuario no tiene dónde dejar la bicicleta. Un falso positivo es un traslado innecesario del camión, que consume tiempo de una flota limitada.

| Métrica | Qué mide | Por qué sirve a la decisión |
| --- | --- | --- |
| `test_recall` | De las ventanas con riesgo real, cuántas se detectan | Mide el error más caro, el falso negativo. Es la métrica principal. |
| `test_precision` | De las alertas emitidas, cuántas eran riesgo real | Indica qué fracción de los envíos del camión sería útil; con una flota limitada, una precisión baja satura la operación. |
| `test_f1` | Media armónica de precisión y recuperación | Resume en un número el equilibrio entre ambos errores a un umbral fijo. |
| `test_balanced_accuracy` | Promedio de la recuperación de cada clase | Con 95 % de ventanas sin riesgo en prueba, la exactitud simple premiaría responder siempre "sin riesgo"; la versión balanceada no. |
| `test_average_precision` | Área bajo la curva de precisión y recuperación | Evalúa el puntaje en todos los umbrales, no sólo en el 30 %, y se compara contra la prevalencia: un clasificador al azar obtiene una precisión promedio cercana a la proporción de positivos. Con clases tan desbalanceadas, esta curva informa mejor sobre la clase minoritaria que la curva ROC. |
| `test_prevalencia_riesgo` | Proporción de ventanas con riesgo en prueba | No evalúa al modelo; da la referencia para leer las demás. |

**Resultado del baseline (septiembre).** Recuperación 0.069, precisión 0.187, F1 0.101, exactitud balanceada 0.526 y precisión promedio 0.101, frente a una prevalencia de 0.050. La regla detecta menos de una de cada catorce ventanas de riesgo: el desbalance de la hora previa casi nunca continúa con la misma magnitud. Su precisión promedio es el doble de la prevalencia, así que el saldo reciente sí contiene señal, pero un umbral fijo sobre una sola hora la aprovecha poco. Ese es el margen que debe cubrir el candidato.

## P3. Grupos de estaciones con patrones de uso parecidos

**Decisión que apoya.** Planeación urbana decide dónde ampliar capacidad o abrir estaciones y qué tipo de estación conviene en cada zona, a partir de grupos que se puedan ubicar en el mapa e interpretar por su curva horaria.

**Baseline.** K-Means con k = 4, 20 inicializaciones y semilla 42, sobre los 26 rasgos de perfil estandarizados (24 proporciones horarias, proporción de fin de semana y proporción de salidas sobre la actividad total), calculados con las salidas que inician y las llegadas que terminan antes del 1 de septiembre.

**Candidatos.** Todos usan los mismos 26 rasgos estandarizados que el baseline.

- K-Means con k = 5 y k = 6, para ver si más grupos separan mejor los patrones que el k = 4 del baseline.
- MiniBatch K-Means con k = 5, que ajusta con lotes y cuesta menos al crecer el número de estaciones.
- Ward con k = 5. Ward minimiza la misma varianza dentro de los grupos que K-Means, así que la comparación es justa, pero es determinista y no depende de la semilla. `AgglomerativeClustering` de scikit-learn no tiene `predict`; por eso el proyecto lo envuelve en `WardCentroides` (`mibici/modeling/clustering.py`), un estimador compatible con scikit-learn que ajusta Ward, calcula el centroide de cada grupo en el espacio estandarizado y asigna cualquier estación nueva al centroide más cercano. Así el modelo tiene firma y se carga desde el Model Registry como los demás.
- K-Means con k = 5 sin las estaciones inactivas, es decir, sin salidas antes del corte (criterio `MIN_SALIDAS_ESTACION_ACTIVA` en `mibici/config.py`; ver el hallazgo de abajo). Es una variación de los datos de entrada, no del algoritmo: la corrida registra el parámetro `estaciones=activas`, el número de estaciones excluidas, sus identificadores y el archivo `data/estaciones_excluidas_p3.csv`.

El candidato que pasa al Model Registry es el de mayor `train_silhouette` entre las corridas con las 484 estaciones. La corrida sin inactivas no compite: su silhouette se calcula sobre otro conjunto de estaciones y quitar 16 vectores idénticos cambia la métrica por sí solo, así que compararla con las demás no diría qué algoritmo agrupa mejor. Se reporta aparte como evidencia del efecto de esas estaciones.

**Métricas.** Ninguna métrica interna garantiza que un grupo sea útil para planear; por eso se combinan con los artefactos del run (mapa y curvas horarias promedio) para interpretar cada grupo.

| Métrica | Qué mide | Por qué sirve a la decisión |
| --- | --- | --- |
| `train_silhouette` | Promedio global de `silhouette_score`: para cada estación compara su distancia media a las de su grupo con la distancia media a las del grupo vecino, en una escala de menos 1 a 1, y promedia sobre todas las estaciones | Resume en un número qué tan separados quedan los grupos y es el criterio para elegir k. El valor por estación (`silhouette_samples`) señalaría las estaciones mal asignadas que no conviene usar como ejemplo de su grupo, pero hoy no se registra. |
| `train_davies_bouldin` | Parecido promedio de cada grupo con el más similar a él; menor es mejor | Detecta grupos que se traslapan, es decir, tipos de estación que planeación no podría distinguir en la práctica. |
| `train_calinski_harabasz` | Dispersión entre grupos frente a dispersión dentro de ellos, cada una dividida entre sus grados de libertad, con el factor `(n - k) / (k - 1)`; mayor es mejor | Complementa a las anteriores con una medida global de separación. Esa normalización por k permite compararla entre modelos con distinto número de grupos, como apoyo a la silhouette al elegir k. |
| `train_inercia` | Suma de distancias cuadradas de cada estación al centro de su grupo | Es la función que minimiza K-Means; su caída al aumentar k (método del codo) ayuda a acotar el rango de k. |

**Resultado del baseline (abril a agosto).** Silhouette 0.123, Davies-Bouldin 2.038, Calinski-Harabasz 53.8 e inercia 9,418.6. La silhouette baja indica grupos poco separados.

### Hallazgo: un grupo del baseline son las estaciones sin actividad

Los grupos del run tienen 212, 92, 164 y 16 estaciones (grupos 0 a 3). Las 16 estaciones del grupo 3 no tienen ninguna salida ni ninguna llegada entre el 1 de abril y el 31 de agosto. Su perfil queda con las 24 proporciones horarias y la proporción de fin de semana en cero y la proporción de salidas en 0.5, así que las 16 tienen exactamente el mismo vector y K-Means las aparta como un grupo propio. El baseline encuentra, en realidad, tres patrones de uso más un grupo de estaciones sin datos, y ese grupo también entra en el cálculo de las métricas anteriores.

| Estación | Nombre en la nomenclatura | Capacidad | Observación |
| ---: | --- | ---: | --- |
| 22 | ex-(GDL-020) Juan Manuel / C.General Coronado | 15 | Sin viajes en el periodo |
| 97 | ex-(ZPN-007) C. Libertad / Nicolás Bravo | 19 | Sin viajes en el periodo |
| 99 | ex-(ZPN-009) C. Del Corral / C. Nicolás Bravo | 0 | Sin viajes; capacidad cero, excluida de P2 |
| 109 | ex-(ZPN-019) C. Cabecera Mpal. Juan Manuel / C. | 0 | Sin viajes; capacidad cero, excluida de P2 |
| 113 | ex-(ZPN-023) C. Juan Diego / C. Sarcófago | 11 | Sin viajes en el periodo |
| 114 | ex-(ZPN-024) Manuel H. Alatorre / C. Industria | 11 | Sin viajes en el periodo |
| 116 | ex-(ZPN-026) C. 16 Sep. / C. Fray A. de Segovia | 0 | Sin viajes; capacidad cero, excluida de P2 |
| 138 | ex-(ZPN-044) | 0 | Sin viajes; capacidad cero, excluida de P2 |
| 147 | ex-(TLQ-006) C. Zalatitán / Av. Francisco Silva | 15 | Sin viajes en el periodo |
| 149 | ex-(TLQ-008) C. Juárez / C.Francisco de Miranda | 15 | Sin viajes en el periodo |
| 180 | (GDL-113) C. Ghilardi / C. Arista | 15 | Sin viajes antes del corte; 478 salidas en septiembre sin contar viajes espurios |
| 223 | ex-(GDL-156) C. Crepúsculo / Av. Mariano Otero | 15 | Sin viajes en el periodo |
| 240 | ex-(GDL-173) C. Caballerango / C. Caporales | 19 | Sin viajes en el periodo |
| 250 | ex-GDL-182 Av. Plan de San Luis / Av. Enríque | 15 | Sin viajes en el periodo |
| 310 | ex-(ZPN-063) C. San Felipe Neri / Av. Guadalu | 15 | Sin viajes en el periodo |
| 492 | (GDL-297) Calz. Independencia / Av. República | 19 | Sin viajes en el periodo |

Los nombres se copian tal como aparecen en `nomenclatura_2026_09.csv`, incluidos los truncados. Catorce de las 16 llevan el prefijo "ex-" en su nombre, lo que sugiere estaciones dadas de baja; la nomenclatura no lo explica. La estación 180 empieza a registrar salidas en septiembre y la 492 no registra viajes en ningún mes del periodo. Al leer el mapa y las curvas del run, el grupo 3 debe interpretarse como estaciones sin actividad en entrenamiento y no como un patrón de uso. Son exactamente las 16 que excluye la corrida `P3_kmeans_k5_activas`.
