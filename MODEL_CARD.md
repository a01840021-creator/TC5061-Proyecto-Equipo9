# Model Card — Predicción horaria de demanda de MiBici (P1)

**Estado:** modelo vigente para la API del proyecto · **Última revisión:** 25 de noviembre
de 2026 · **Equipo responsable:** TC5061, Equipo 09

Este documento acompaña al modelo de regresión que estima cuántos viajes iniciarán en una
estación durante una hora. Resume su propósito, procedencia, evaluación, límites y controles
de operación. Debe leerse junto con el README y no sustituye el criterio operativo humano.

## 1. Detalles del modelo

| Campo | Valor |
| --- | --- |
| Nombre registrado | `mibici-equipo09-p1-demanda` |
| Estado operativo | alias `champion` |
| Versión | `1` |
| URI lógica | `models:/mibici-equipo09-p1-demanda@champion` |
| Corrida de origen | `ed97390337ce4d2f8d95d0a58cc02b8f` |
| Periodo de entrenamiento | `2026-04-01/2026-07-31` |
| Familia | `HistGradientBoostingRegressor` de scikit-learn |
| Entorno del artefacto | Python 3.10.20, scikit-learn 1.7.2, MLflow 3.17.1 |
| Semilla | 42 |
| Licencia de los datos | Datos abiertos de MiBici; consultar sus términos de uso |

La imagen de la API incluye una instantánea exportada de esta versión en `models/p1/`. El
alias sigue siendo la identidad lógica de producción, mientras que versión, `run_id` y
periodo dejan fija la identidad material que se ejecuta.

## 2. Uso previsto

**Usuarios previstos.** Personal técnico u operativo que planifica el rebalanceo de la red
MiBici y aplicaciones internas que requieren una estimación horaria por estación.

**Decisión apoyada.** Priorizar estaciones y horas donde conviene revisar disponibilidad o
redistribuir bicicletas. La predicción es una señal de planeación; no es una orden automática.

**Usos fuera de alcance.** No debe emplearse para decisiones sobre personas, vigilancia,
tarificación individual, sanciones, pronósticos fuera de la red de Guadalajara, inventario en
tiempo real ni decisiones críticas sin revisión humana. Tampoco se validó para horizontes,
estaciones o calendarios posteriores sin monitoreo de drift.

## 3. Datos

La fuente son los datos abiertos de viajes y nomenclatura de estaciones de MiBici entre abril
y septiembre de 2026. La ingesta recibió 2,294,886 viajes, eliminó 758 duplicados exactos y
3,954 viajes de estaciones ausentes de la nomenclatura, y conservó 2,290,174 viajes y 484
estaciones. El manifiesto `references/manifiesto_datos.json` registra archivos y SHA-256.

La partición es temporal para evitar aprender del futuro: abril-julio para entrenamiento,
agosto para validación y septiembre para prueba. Las variables personales `usuario_id`,
`genero` y `anio_de_nacimiento` se eliminan antes de guardar los datos procesados. El modelo
trabaja con agregados estación-hora, no con registros de personas.

Riesgos de representatividad: el periodo cubre seis meses de un solo sistema; clima, obras,
festivos, nuevas estaciones, cierres o cambios de política posteriores pueden modificar la
demanda. Una estación con poca historia o una topología nueva puede quedar insuficientemente
representada.

## 4. Variables y salida

| Variable | Interpretación y validación de la API |
| --- | --- |
| `estacion_id` | Identificador de estación; entero mayor o igual que 1 |
| `dpcapacity` | Capacidad registrada; número no negativo |
| `hora` | Hora local, entero de 0 a 23 |
| `dia_semana` | Lunes=0 a domingo=6 |
| `es_fin_de_semana` | 0/1 y consistente con `dia_semana` |
| `rezago_24h` | Viajes observados en la misma estación 24 horas antes; no negativo |
| `rezago_48h` | Viajes observados 48 horas antes; no negativo |
| `rezago_168h` | Viajes observados una semana antes; no negativo |

La salida `prediction` es un número real que representa viajes esperados por estación-hora.
No es un conteo observado ni un intervalo probabilístico. El consumidor debe tratar valores
fraccionarios como una estimación y aplicar reglas operativas de redondeo sólo fuera del
modelo.

## 5. Entrenamiento

Se compararon un baseline de persistencia semanal, tres configuraciones de
HistGradientBoosting y Ridge. La selección usó el menor MAE de validación, nunca el conjunto
de prueba. El champion usa `learning_rate=0.05`, `max_iter=200`, `max_leaf_nodes=63`,
`l2_regularization=1.0` y semilla 42. Los rezagos sólo contienen observaciones disponibles
antes de emitir la predicción.

Para observaciones \(y_i\) y predicciones \(\hat y_i\), la métrica primaria es
\(\mathrm{MAE}=n^{-1}\sum_i |y_i-\hat y_i|\). Se complementa con
\(\mathrm{RMSE}=\sqrt{n^{-1}\sum_i(y_i-\hat y_i)^2}\), WAPE, \(R^2\) y sesgo medio.

## 6. Evaluación

El A/B más reciente compara las versiones sobre las mismas 174,240 filas del 16 al 30 de
septiembre de 2026, posteriores a ambos entrenamientos.

| Métrica | Champion v1 | Challenger v3 |
| --- | ---: | ---: |
| MAE | 0.8879 | 0.8856 |
| RMSE | 1.6199 | 1.6084 |
| WAPE | 59.09% | 58.94% |
| \(R^2\) | 0.6625 | 0.6673 |
| Sesgo | -0.2039 | -0.1822 |

El challenger reduce el MAE 0.0023 viajes (0.26%) y gana 11 de 15 días, pero el intervalo
bootstrap del 95% para la mejora diaria es [-0.0008, 0.0051] y la prueba t pareada por día
produce p=0.151. Como el intervalo incluye cero y p>0.05, la evidencia no es suficiente para
promoverlo. Se mantiene `champion` en la versión `1`; `challenger` permanece en la versión
`3`, corrida `3f30c52e3196483cb612dcce49fe8f8e`, para nuevas pruebas.

La unidad inferencial es el día, no cada fila: estaciones de una misma hora comparten clima,
eventos y demanda del sistema. Tratar 174,240 filas como independientes produciría un error
estándar artificialmente pequeño.

## 7. Limitaciones

- No incluye clima, festivos, eventos, cierres, inventario actual ni conectividad de la red.
- Los rezagos exigen observaciones recientes correctas; retrasos o errores de captura se
  propagan a la predicción.
- Una estimación puntual no comunica incertidumbre ni garantiza disponibilidad de bicicleta.
- El error agregado puede ocultar estaciones u horas con desempeño sistemáticamente peor.
- El modelo se validó sólo en la red y ventanas de 2026 descritas; extrapolar exige nueva
  validación.
- El MAE aumentó 21% de agosto a septiembre y el sesgo se volvió negativo: el champion
  subestima parte del crecimiento de demanda aunque siga siendo la versión vigente.

## 8. Consideraciones éticas

El servicio no usa atributos protegidos ni identificadores personales. Aun así, una política
de rebalanceo basada únicamente en demanda histórica puede favorecer zonas con mayor uso y
perpetuar desigualdades geográficas. Se recomienda revisar métricas por estación/zona,
incorporar objetivos explícitos de cobertura y permitir intervención humana. No debe inferirse
ninguna característica de una persona a partir de los agregados.

Los datos abiertos reducen, pero no eliminan, el deber de minimización: el pipeline borra los
campos personales antes del almacenamiento procesado y no registra ejemplos crudos en MLflow.

## 9. Despliegue y seguridad

FastAPI valida tipos, rangos, campos adicionales y coherencia del fin de semana con Pydantic.
La respuesta incluye nombre, alias, versión, URI y latencia para auditoría. El contenedor corre
con usuario sin privilegios, usa dependencias fijadas y un `HEALTHCHECK`.

El artefacto `cloudpickle` sólo debe cargarse desde el repositorio y pipeline confiables:
deserializar modelos de origen desconocido puede ejecutar código. Secretos, bases de MLflow y
datos no se incluyen en Git ni en la imagen. Bandit y pip-audit forman parte de CI; los riesgos
aceptados y controles compensatorios están en `references/seguridad.md`.

En la prueba de carga se completaron 13,565 predicciones sin fallos: 37.59 solicitudes/s en
promedio, latencia p50=11 ms, p95=43 ms y p99=180 ms. Es una caracterización local, no un SLA.

## 10. Monitoreo y continuous training

Evidently compara la referencia abril-julio con ventanas posteriores. En la primera quincena
de septiembre `demanda` presentó drift (0.1152); los rezagos quedaron cerca del umbral. La
degradación confirmada por MAE y sesgo activa la política `reentrenar_o_ajustar_challenger`,
pero jamás mueve automáticamente el alias.

La promoción requiere, sobre datos posteriores al entrenamiento: MAE menor; límite inferior
del intervalo bootstrap diario mayor que cero; prueba t diaria con p<0.05; y ausencia de
deterioro material en RMSE y \(R^2\). Si falla cualquier criterio, se conserva el champion,
se investiga el drift y se amplía o ajusta el challenger. El rollback consiste en reasignar el
alias `champion` a la versión anterior y reconstruir la instantánea del servicio.

## 11. Gobernanza y linaje

MLflow enlaza modelo, versión, corrida, commit, versión de datos y métricas. Se usan aliases
porque las etapas de Registry están obsoletas: `champion` designa producción y `challenger` la
alternativa. El inventario reproducible `reports/final/registry_inventory.json` contrasta las
instantáneas exportadas con el A/B; se genera con `make governance-report`.

La versión 3 fue creada desde un árbol con cambios sin commit. Ese hecho no se oculta: sus
etiquetas declaran `git_dirty=true` y conservan una instantánea de `challenger.py` y su SHA-256.
Esta limitación de proceso es aceptable para la evidencia académica, pero en producción un
pipeline debe rechazar entrenamientos desde árboles sucios.

## 12. FinOps

La demanda estimada es 348,480 inferencias mensuales (484 estaciones × 24 horas × 30 días).
Con precios de referencia documentados en `references/finops.md`, Lambda cuesta cerca de
USD 0.17/mes sin capa gratuita (USD 0.50 por millón), frente a USD 6.13/mes para EC2
t4g.micro. Por el patrón periódico y de baja utilización, la opción serverless minimiza tiempo
ocioso; la decisión debe reevaluarse si la carga sostenida supera aproximadamente 4.67 RPS.

## 13. Reproducibilidad

Python se fija en 3.10.20 y las dependencias usan versiones exactas. `MLproject` define los
entry points; las semillas, ventanas y rutas viven en `mibici/config.py`. Las pruebas verifican
reproducibilidad, linaje, API, drift y A/B. Para comprobar esta documentación y el Registry:

```bash
make governance-report
make audit-final
python -m pytest -q tests/test_model_governance.py
```

Los modelos exportados permiten ejecutar la demo sin la base local de MLflow; la base y los
artefactos originales no se versionan. La exportación conserva la identidad necesaria para
reconstruir su origen.

## 14. Responsables y cambios

El Equipo 09 es responsable de revisar datos, métricas, drift, seguridad y costo antes de una
promoción. Todo cambio pasa por Pull Request y revisión de otro integrante.

| Fecha | Cambio | Decisión |
| --- | --- | --- |
| 2026-10-26 | Registro del champion v1 | Seleccionado por validación |
| 2026-11-15 | Challenger v3 y A/B por día | Mantener champion v1 |
| 2026-11-25 | Model Card e inventario final | Documentar estado vigente y controles |
