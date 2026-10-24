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

**P3. Perfil de uso por estación.** Sin variable objetivo. Cada estación se describe con la proporción de sus salidas en cada hora del día, la proporción de sus salidas en fin de semana y la proporción de salidas sobre el total de sus salidas y llegadas. Las tres se calculan con los viajes que inician antes del 1 de septiembre. Las estaciones sin salidas en ese periodo quedan con perfil horario cero y proporción de salidas de 0.5; el efecto que esto tiene en el agrupamiento se describe en `references/modelos_metricas.md`.

### Tratamiento de viajes espurios

Hay 81,947 viajes de menos de un minuto que terminan en la misma estación donde empezaron,
el 3.6 % del total. Se interpretan como préstamos fallidos o devoluciones inmediatas, no
como viajes reales. La ingesta los conserva para mantener fidelidad respecto de la fuente,
pero `features.py` los excluye antes de construir P1, P2 y P3. De esta manera, la regla queda
documentada, es reproducible y no altera los archivos crudos.

## Condiciones de uso y datos personales

El portal de datos abiertos no publica una licencia explícita de uso. Su pie de página indica "Todos los derechos reservados" y señala que el sistema es operado por BKT Bici pública S.A. de C.V. y administrado por la Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara (MiBici, s. f.-a). Los términos y condiciones regulan la suscripción y el uso del servicio, y tampoco se refieren a los datos abiertos (MiBici, s. f.-b). El aviso de privacidad de la Agencia remite a la Ley de Protección de Datos Personales en Posesión de Sujetos Obligados del Estado de Jalisco y sus Municipios, y a la ley general correspondiente, pero no se refiere a los datos abiertos de viajes.

Ante esa falta de licencia, el proyecto adopta un uso conservador:

- **Uso académico y sin fines comerciales**, citando a MiBici como fuente.
- **Sin redistribución.** Los archivos no se versionan en Git: el script los descarga de la fuente original cada vez, y el repositorio solo guarda sus hashes en el manifiesto.
- **Sin datos personales.** Los archivos publicados incluyen un identificador de usuario, su género y su año de nacimiento. Combinados con las horas y estaciones de cada viaje, permitirían reconstruir los hábitos de una persona. Ninguna de las tres preguntas los necesita, así que se eliminan en la ingesta, antes de guardar los datos procesados, y nunca se registran en MLflow ni en el repositorio.

## Estrategia de partición

**Partición temporal, nunca aleatoria.** Una partición aleatoria mezclaría horas de septiembre en el entrenamiento y le daría al modelo información del futuro que no tendría en operación.

| Conjunto | Periodo | Viajes |
| --- | --- | ---: |
| Entrenamiento | 1 de abril al 31 de agosto de 2026 | 1,800,876 |
| Prueba | 1 al 30 de septiembre de 2026 | 489,298 |

Los viajes de la tabla son los de la ingesta. Después de quitar los viajes espurios quedan 1,736,571 en entrenamiento y 471,656 en prueba.

Reglas para que ninguna variable use información del futuro:

- **P1:** las variables del día *d* se calculan solo con datos hasta el cierre del día *d−1*: por ejemplo, las salidas de la misma hora el día anterior o el mismo día de la semana anterior.
- **P2:** las variables de la hora *t* usan solo datos hasta *t*. La etiqueta usa *t+1* y *t+2*. Las ventanas cuya etiqueta cruza el 1 de septiembre se descartan del entrenamiento.
- **P3:** los perfiles, el escalamiento y el agrupamiento se calculan con los viajes que inician entre el 1 de abril y el 31 de agosto. De ellos, 18 terminan el 1 de septiembre, entre las 00:00 y las 09:07, y su llegada cuenta en la proporción de salidas de su estación de destino. Son 18 de 1,736,571 viajes de entrenamiento: cambian esa proporción en 18 estaciones, en no más de 0.0002, ninguna estación cambia de grupo y la silhouette del run cambia en menos de una millonésima.
- Los baselines de P1 y P2 no escalan ni imputan variables; el único escalamiento, el de P3, se ajusta con los perfiles de entrenamiento.

Septiembre tiene 24.3 % más viajes que agosto (489,298 frente a 393,524). Una explicación posible es el regreso a clases, pero estos datos no permiten comprobarla. Sea cual sea la causa, el mes de prueba es exigente: mide si los modelos resisten un cambio de volumen como los que encontrarán en operación.

## Referencias

Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara. (s. f.). *Aviso de privacidad*. MiBici. https://mibici.net/es/aviso-de-privacidad/

MiBici. (s. f.-a). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/

MiBici. (s. f.-b). *Términos y condiciones*. https://mibici.net/es/termino-y-condiciones/
