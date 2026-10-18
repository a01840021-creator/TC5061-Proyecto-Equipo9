# ML Canvas y formulación del problema

Proyecto MiBici, equipo 9. Este documento formula las tres preguntas del proyecto como problemas de Machine Learning y las reúne en un ML Canvas (Dorard, 2018). Las cifras provienen de la ingesta de abril a septiembre de 2026 (`references/definicion_datos.md`).

## Formulación de las preguntas

### P1. Demanda por estación

| | |
| --- | --- |
| **Qué se predice** | Número de viajes que inician en cada estación en cada hora |
| **Tipo de tarea** | Regresión |
| **Granularidad** | Estación por hora |
| **Horizonte** | Las 24 horas del día siguiente, con la información disponible al cierre del día anterior |
| **Por qué es relevante** | La demanda no se reparte igual entre estaciones ni entre horas: en los días en que tiene actividad, una estación inicia en promedio 30 viajes, pero en el 61 % de las combinaciones estación y hora no sale ningún viaje. Saber dónde y cuándo se concentrará la demanda es la base para distribuir la flota |
| **Quién lo usa** | El operador del sistema |
| **Qué decisión toma** | Cuántas bicicletas dejar en cada estación al inicio del día y en qué franjas reforzar la operación |

### P2. Riesgo de quedarse sin bicicletas o sin anclajes

| | |
| --- | --- |
| **Qué se predice** | Si una estación entrará en riesgo de vaciarse o de llenarse en las próximas dos horas |
| **Tipo de tarea** | Clasificación binaria |
| **Granularidad** | Estación por hora |
| **Horizonte** | Las dos horas siguientes, con la información disponible hasta la hora actual |
| **Por qué es relevante** | Una estación vacía pierde viajes y una llena impide terminar el viaje ahí. Las dos dañan el servicio y las dos se corrigen con el mismo recurso: el camión de rebalanceo |
| **Quién lo usa** | El equipo de rebalanceo |
| **Qué decisión toma** | A qué estaciones mandar el camión en las próximas dos horas y si debe llevar o retirar bicicletas |

**Cómo se aproxima el riesgo.** MiBici no publica cuántas bicicletas hay en cada estación, así que el inventario no se puede observar. Lo que sí se observa es su variación: cada viaje que llega suma una bicicleta a la estación de destino y cada viaje que sale resta una a la de origen. El **saldo neto** de una ventana (llegadas menos salidas) es exactamente el cambio de inventario en ese lapso.

La etiqueta compara ese saldo con la capacidad de la estación:

- **Riesgo de vaciarse:** en las dos horas siguientes salen más bicicletas de las que llegan, por al menos el 30 % de la capacidad.
- **Riesgo de llenarse:** llegan más de las que salen, por al menos el 30 % de la capacidad.

La variable objetivo vale 1 si ocurre cualquiera de los dos. Con este umbral, el 4.0 % de las combinaciones estación y ventana de dos horas queda marcado como riesgo: 1.9 % por vaciado y 2.1 % por llenado. Es una clase minoritaria pero no extrema, y la proporción entre los dos tipos de riesgo está equilibrada.

El supuesto es que el operador mantiene las estaciones en un nivel intermedio, así que un movimiento neto del 30 % de la capacidad en dos horas es suficiente para llevarlas a un extremo. El umbral se puede ajustar sin cambiar la formulación.

### P3. Grupos de estaciones con patrones de uso parecidos

| | |
| --- | --- |
| **Qué se agrupa** | Estaciones, según su perfil de uso |
| **Tipo de tarea** | Clustering |
| **Granularidad** | Una observación por estación, que resume su comportamiento en el periodo de entrenamiento |
| **Horizonte** | No aplica. Los grupos describen el periodo observado; su estabilidad se verifica con el mes de prueba |
| **Por qué es relevante** | Dos estaciones con el mismo volumen pueden servir a viajes distintos: una de origen matutino en zona habitacional y otra de destino en zona de oficinas. Agruparlas por patrón, y no solo por volumen, revela cómo se usa el sistema en cada parte de la ciudad |
| **Quién lo usa** | Planeación urbana y expansión del sistema |
| **Qué decisión toma** | Dónde ampliar capacidad o abrir estaciones nuevas, y qué tipo de estación conviene en cada zona |

El perfil de cada estación se construye con su distribución de salidas por hora del día, la proporción de uso entre semana y fin de semana, y su saldo entre viajes que salen y viajes que llegan. Los grupos se ubican en el mapa con las coordenadas de la nomenclatura.

## ML Canvas

| Bloque | Contenido |
| --- | --- |
| **Propuesta de valor** | Anticipar la demanda y el desbalance de las estaciones de MiBici para que el operador asigne flota y camiones antes de que el problema ocurra, y entender cómo se usa el sistema en cada zona para planear su expansión |
| **Tareas de predicción** | P1, regresión: viajes que inician por estación y hora. P2, clasificación: riesgo de vaciarse o llenarse en las dos horas siguientes. P3, clustering: grupos de estaciones por perfil de uso |
| **Decisiones** | P1: distribución de bicicletas al inicio del día. P2: ruta y carga del camión de rebalanceo. P3: dónde ampliar o abrir estaciones |
| **Simulación de impacto** | Se compara cada modelo contra una línea base simple en el mes de prueba. P1: reducción del error absoluto medio frente al promedio histórico. P2: proporción de estaciones en riesgo detectadas a tiempo frente a una regla de persistencia. P3: cohesión de los grupos y estabilidad al asignar el mes de prueba |
| **Cómo se generan las predicciones** | P1, una vez al día al cierre de operación, para las 24 horas siguientes. P2, cada hora, para las dos horas siguientes. P3, por lote al cierre de cada periodo de análisis |
| **Recolección de datos** | MiBici publica un archivo de viajes por mes. Cada mes nuevo se incorpora con la misma ingesta y queda registrado con su hash en el manifiesto de datos |
| **Fuentes de datos** | Viajes mensuales de abril a septiembre de 2026 y nomenclatura de estaciones de septiembre de 2026, del portal de datos abiertos de MiBici |
| **Construcción de modelos** | Un experimento en MLflow por proyecto, con al menos un run por pregunta. Cada run registra el rango de fechas, la versión de los datos, la semilla y los hiperparámetros. Los modelos se reentrenan cada mes con los datos nuevos |
| **Variables** | Calendario (hora, día de la semana, fin de semana), historial de la propia estación (salidas y llegadas en horas y días anteriores, en los mismos horarios de semanas previas), capacidad y ubicación. Toda variable se calcula solo con información disponible antes del momento de la predicción |
| **Monitoreo** | Error de P1 y recuperación de P2 por mes, comparados contra su línea base. Cambios en el volumen total de viajes o en la lista de estaciones de la nomenclatura, que obligan a reentrenar |

## Referencias

Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning. https://medium.com/louis-dorard/from-data-to-ai-with-the-machine-learning-canvas-part-i-d171b867b047

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/
