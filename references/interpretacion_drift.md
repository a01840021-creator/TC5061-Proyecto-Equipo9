# Interpretación del drift y recomendación de Continuous Training

## Periodos analizados

- **Referencia:** 1 de abril al 31 de julio de 2026, periodo utilizado para entrenar los candidatos de P1.
- **Control intermedio:** agosto de 2026, correspondiente a validación.
- **Periodo actual:** septiembre de 2026, correspondiente a prueba y al periodo posterior que simula producción.

La comparación conserva el orden temporal y evita usar septiembre durante el entrenamiento. La ventana de referencia es más extensa que las ventanas actuales; por ello, los resultados deben interpretarse junto con el posible efecto estacional y no como demostración automática de pérdida de desempeño.

## Hallazgos

En agosto no se detectó drift en las nueve variables evaluadas. En septiembre, Evidently detectó drift en cuatro: `rezago_24h`, `rezago_48h`, `rezago_168h` y `demanda`.

| Variable | Rol | Método | Score | Umbral | Interpretación |
| --- | --- | --- | ---: | ---: | --- |
| `rezago_24h` | Entrada | Wasserstein normalizada | 0.1754 | 0.10 | Cambio relevante en demanda reciente. |
| `rezago_48h` | Entrada | Wasserstein normalizada | 0.1588 | 0.10 | Cambio relevante en el patrón de dos días. |
| `rezago_168h` | Entrada | Wasserstein normalizada | 0.1490 | 0.10 | Cambio frente al patrón semanal aprendido. |
| `demanda` | Objetivo | Wasserstein normalizada | 0.1971 | 0.10 | La distribución del resultado también cambió. |

Evidently no marcó drift global del dataset porque 4 de 9 columnas equivalen a 44.44%, por debajo del umbral de 50%. Sin embargo, esta bandera agregada no debe ocultar que las tres variables de rezago usadas por el modelo y el objetivo presentan drift. La señal operativa se clasifica como de prioridad alta.

Las variables `dpcapacity`, `estacion_id`, `hora`, `dia_semana` y `es_fin_de_semana` permanecieron estables. Esto sugiere que el cambio no se explica por una modificación general del esquema, capacidad o composición temporal, sino por el nivel reciente y semanal de la demanda.

## Implicaciones para P1

Los rezagos son predictores directos del modelo de demanda. Si sus distribuciones se alejan de las observadas durante entrenamiento, el champion puede enfrentar combinaciones menos representadas. El drift del objetivo refuerza la necesidad de evaluar su MAE en septiembre. No obstante, el drift por sí solo no demuestra degradación: es necesario medir el error con las etiquetas actuales.

## Recomendación

Se recomienda generar y evaluar un challenger reentrenado con información disponible hasta agosto, manteniendo septiembre exclusivamente como conjunto de comparación. No se recomienda mover automáticamente el alias `champion` por la sola detección de drift.

La promoción debería ocurrir únicamente si:

1. Champion y challenger se evalúan sobre exactamente las mismas filas de septiembre.
2. El challenger mejora el MAE y el intervalo de confianza de la diferencia favorece la mejora.
3. No se deterioran de manera material RMSE, R² o latencia.
4. Los datos superan las validaciones de calidad y el modelo conserva linaje en MLflow.
5. La promoción se realiza gradualmente y existe capacidad de rollback.

Con la evidencia disponible hasta esta etapa, la decisión es **evaluar un challenger**, no promoverlo todavía.
