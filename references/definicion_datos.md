# Definición de los datos

Proyecto MiBici, equipo 9. Cubre el rango de fechas, la fuente, el tamaño real, las variables y la construcción de cada variable objetivo, las condiciones de uso y la estrategia de partición. Las cifras salen de la ingesta (`make data`), versión de datos `0c4b34e5425a`; las que dependen del filtro de viajes espurios salen de los conjuntos que construye `mibici/features.py`, y en cada caso se indica a qué conjunto se refieren.

## Rango de fechas

**Abril a septiembre de 2026.**

- Son los seis meses más recientes publicados al momento de la ingesta.
- Terminan en el mismo mes que la nomenclatura de estaciones disponible (septiembre de 2026). Así, cada viaje se puede ligar con la capacidad y la ubicación vigentes de su estación, que P2 y P3 necesitan.
- Seis meses dan más de dos millones de viajes y cubren cambios de temporada, como el aumento de viajes de septiembre, sin volver inmanejable el procesamiento en una computadora personal.

## Fuente

Portal de [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (MiBici, s. f.-a). Las URLs exactas también están en `mibici/config.py`.

| Archivo | URL |
| --- | --- |
| Viajes abril 2026 | https://www.mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv |
| Viajes mayo 2026 | https://www.mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv |
| Viajes junio 2026 | https://www.mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv |
| Viajes julio 2026 | https://www.mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv |
| Viajes agosto 2026 | https://www.mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv |
| Viajes septiembre 2026 | https://www.mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv |
| Nomenclatura de estaciones | https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv |

## Tamaño real obtenido

| Mes | Registros crudos | Viajes tras la limpieza |
| --- | ---: | ---: |
| Abril | 351,851 | 351,851 |
| Mayo | 315,895 | 315,895 |
| Junio | 355,303 | 353,323 |
| Julio | 388,257 | 386,283 |
| Agosto | 393,524 | 393,524 |
| Septiembre | 490,056 | 489,298 |
| **Total** | **2,294,886** | **2,290,174** |

La limpieza de la ingesta elimina 758 duplicados exactos, todos en septiembre, y 3,954 viajes de las estaciones 405 y 407, que no aparecen en la nomenclatura. Quedan **6 meses, 2,290,174 viajes y 484 estaciones**, de las cuales 469 registran al menos una salida en el periodo.

## Variables originales

| Archivo | Variable | Uso |
| --- | --- | --- |
| Viajes | `viaje_id` | Identificador del viaje; sirve para detectar duplicados |
| Viajes | `inicio_del_viaje`, `fin_del_viaje` | Fecha y hora de salida y de llegada |
| Viajes | `origen_id`, `destino_id` | Estación de salida y de llegada |
| Nomenclatura | `id`, `name` | Identificador y nombre de la estación |
| Nomenclatura | `latitude`, `longitude` | Ubicación, para el mapa de P3 |
| Nomenclatura | `dpcapacity` | Número de anclajes, para la etiqueta de P2 |

Se descartan en la ingesta `usuario_id`, `genero` y `anio_de_nacimiento` (ver condiciones de uso).

## Construcción de las variables objetivo

**P1. Viajes que inician por estación y hora.** Se agrupan los viajes por `origen_id` y por la hora de `inicio_del_viaje` truncada a la hora. Las combinaciones sin viajes se completan con cero, porque una hora sin salidas es una observación válida y no un faltante. En el tablón de P1 (484 estaciones, del 8 de abril al 30 de septiembre, sin viajes espurios), el 61.9 % de las combinaciones estación y hora vale cero: 63.3 % en entrenamiento y 55.3 % en prueba.

**P2. Riesgo en las dos horas siguientes.** Para cada estación y cada hora *t*:

1. Llegadas por hora: viajes agrupados por `destino_id` y hora de `fin_del_viaje`.
2. Salidas por hora: viajes agrupados por `origen_id` y hora de `inicio_del_viaje`.
3. Saldo neto de la ventana: llegadas menos salidas en las horas *t+1* y *t+2*.
4. La etiqueta vale 1 si el valor absoluto del saldo neto es al menos el 30 % de `dpcapacity`, y 0 en otro caso.

En el tablón de P2 (480 estaciones, sin viajes espurios y sin las horas cuya ventana futura está incompleta), el 3.9 % de las observaciones queda en riesgo: 1.8 % por vaciado y 2.1 % por llenado. En entrenamiento la proporción es 3.7 % (1.7 % y 2.0 %) y en prueba 5.0 % (2.4 % y 2.6 %). Las estaciones 99, 109, 116 y 138 tienen capacidad cero en la nomenclatura y se excluyen de P2, porque su etiqueta no se puede calcular; el código las identifica en los datos con `dpcapacity == 0`.

**P3. Perfil de uso por estación.** Sin variable objetivo. Cada estación se describe con la proporción de sus salidas en cada hora del día, la proporción de sus salidas en fin de semana y la proporción de salidas sobre el total de sus salidas y llegadas. Las salidas se cuentan con los viajes que inician antes del 1 de septiembre y las llegadas con los que terminan antes de esa fecha, de forma independiente. Las 16 estaciones sin salidas en ese periodo quedan con perfil horario cero y proporción de salidas de 0.5; el efecto que esto tiene en el agrupamiento, y la corrida que las excluye, se describen en `references/modelos_metricas.md`.

### Tratamiento de viajes espurios

Hay 81,947 viajes de menos de un minuto que terminan en la misma estación donde empezaron,
el 3.6 % del total. Se interpretan como préstamos fallidos o devoluciones inmediatas, no
como viajes reales. La ingesta los conserva para mantener fidelidad respecto de la fuente,
pero `features.py` los excluye antes de construir P1, P2 y P3. De esta manera, la regla queda
documentada, es reproducible y no altera los archivos crudos.

El filtro decide con el fin del viaje, que puede caer después del cierre de la ventana de
una predicción. Por eso cada predicción se emite al cierre de su ventana más
`RETRASO_DISPONIBILIDAD_SEG` (60 segundos, en `mibici/config.py`), y el umbral del filtro,
`DURACION_MINIMA_VIAJE_S`, es esa misma constante. A la emisión, el desenlace de cualquier
viaje espurio ya es observable: si empezó antes del cierre, ya volvió a su origen; un viaje
que sigue abierto a la emisión dura al menos el retraso y no es espurio. Así, la limpieza no
usa información posterior a la emisión.

## Condiciones de uso y datos personales

El portal de datos abiertos no publica una licencia explícita de uso. Su pie de página indica "Todos los derechos reservados" y señala que el sistema es operado por BKT Bici pública S.A. de C.V. y administrado por la Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara (MiBici, s. f.-a). Los términos y condiciones regulan la suscripción y el uso del servicio, y tampoco se refieren a los datos abiertos (MiBici, s. f.-b). El aviso de privacidad de la Agencia remite a la Ley de Protección de Datos Personales en Posesión de Sujetos Obligados del Estado de Jalisco y sus Municipios, y a la ley general correspondiente, pero no se refiere a los datos abiertos de viajes.

Ante esa falta de licencia, el proyecto adopta un uso conservador:

- **Uso académico y sin fines comerciales**, citando a MiBici como fuente.
- **Sin redistribución.** Los archivos no se versionan en Git: el script los descarga de la fuente original cada vez, y el repositorio solo guarda sus hashes en el manifiesto.
- **Sin datos personales.** Los archivos publicados incluyen un identificador de usuario, su género y su año de nacimiento. Combinados con las horas y estaciones de cada viaje, permitirían reconstruir los hábitos de una persona. Ninguna de las tres preguntas los necesita, así que se eliminan en la ingesta, antes de guardar los datos procesados, y nunca se registran en MLflow ni en el repositorio.

## Estrategia de partición

**Partición temporal, nunca aleatoria.** Una partición aleatoria mezclaría horas de septiembre en el entrenamiento y le daría al modelo información del futuro que no tendría en operación.

| Conjunto | Periodo | Viajes | Quién lo usa |
| --- | --- | ---: | --- |
| Entrenamiento | 1 de abril al 31 de agosto de 2026 | 1,800,876 | Perfiles de P3 y periodo previo al corte de los baselines |
| Entrenamiento de candidatos | 1 de abril al 31 de julio de 2026 | 1,407,352 | Ajuste de los candidatos de P1 y P2 |
| Validación | 1 al 31 de agosto de 2026 | 393,524 | Comparación de candidatos y baselines de P1 y P2, y selección para el Model Registry |
| Prueba | 1 al 30 de septiembre de 2026 | 489,298 | Medida final de baselines y candidatos de P1 y P2 |

Los viajes de la tabla son los de la ingesta. Después de quitar los viajes espurios quedan 1,736,571 antes del corte y 471,656 en prueba. Las fechas de corte y de validación están en `mibici/config.py` (`FECHA_CORTE` y `FECHA_VALIDACION`).

Reglas para que ninguna variable use información del futuro. En las tres preguntas, la predicción se emite al cierre de su ventana más `RETRASO_DISPONIBILIDAD_SEG` (ver "Tratamiento de viajes espurios"):

- **P1:** la predicción del día *d* se emite al cierre del día *d−1* más el retraso, y sus variables se calculan solo con viajes iniciados antes de ese cierre: por ejemplo, las salidas de la misma hora el día anterior o el mismo día de la semana anterior. No se usa el mes como variable: entrenamiento, validación y prueba cubren meses distintos.
- **P2:** la predicción de la hora *t* se emite al cierre de esa hora más el retraso, y sus variables usan solo datos de la hora *t* y anteriores. La etiqueta usa *t+1* y *t+2*. Las ventanas cuya etiqueta cruza el 1 de septiembre se descartan del entrenamiento; en los candidatos, que se entrenan hasta julio, también se descartan las que cruzan el 1 de agosto, para que la etiqueta de entrenamiento no use horas de validación.
- **P3:** los perfiles se emiten el 1 de septiembre a las 00:00 más el retraso; los perfiles, el escalamiento y el agrupamiento usan sólo eventos anteriores al 1 de septiembre: las salidas por fecha de inicio y las llegadas por fecha de fin. Los 18 viajes que salen antes del corte y llegan el 1 de septiembre cuentan como salida de su origen, pero no como llegada a su destino.
- Los baselines de P1 y P2 no escalan ni imputan variables. Los escaladores de Ridge, de la regresión logística y de los agrupamientos de P3 forman parte de su `Pipeline` y se ajustan sólo con el conjunto de entrenamiento de cada corrida.

Septiembre tiene 24.3 % más viajes que agosto (489,298 frente a 393,524). Una explicación posible es el regreso a clases, pero estos datos no permiten comprobarla. Sea cual sea la causa, el mes de prueba es exigente: mide si los modelos resisten un cambio de volumen como los que encontrarán en operación.

## Referencias

Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara. (s. f.). *Aviso de privacidad*. MiBici. https://mibici.net/es/aviso-de-privacidad/

MiBici. (s. f.-a). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/

MiBici. (s. f.-b). *Términos y condiciones*. https://mibici.net/es/termino-y-condiciones/
