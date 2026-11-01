# Resultados de calidad de datos con DeepChecks

## Alcance

Se evaluaron P1 y P2 con muestras deterministas de 5,000 filas de entrenamiento y 5,000
de prueba por pregunta. La semilla fue 42 y la separación conservó el corte temporal del
proyecto. Se ejecutaron las suites `data_integrity` y `train_test_validation`.

## Resultados

### P1: predicción de demanda

Los conjuntos de entrenamiento y prueba aprobaron las revisiones de duplicados, nulos,
tipos de datos, caracteres especiales, variantes de texto y correlación entre las
variables predictoras y la etiqueta. La comparación temporal entre entrenamiento y
prueba también aprobó sus condiciones, sin una señal de drift que invalide la evaluación.

DeepChecks generó dos observaciones de integridad:

- **Conflicting Labels.** En la muestra de entrenamiento, 0.12 % de las filas comparte
  los mismos valores predictivos con otra fila, pero presenta una demanda diferente. No
  son registros duplicados: el chequeo `Data Duplicates` reportó 0 %. El comportamiento
  es coherente con una variable objetivo estocástica, porque una estación puede registrar
  distinta demanda aun cuando coincidan capacidad, calendario y rezagos.
- **Identifier Label Correlation.** `fecha_hora` presentó un PPS de 0.08 con la demanda.
  Es una relación baja y esperable por estacionalidad. La fecha se utiliza para ordenar y
  dividir temporalmente los datos, pero no se incluye como variable directa del modelo;
  por ello no se considera fuga de información.

Las observaciones se conservan como evidencia y no se eliminan mediante cambios
artificiales en los umbrales.

### P2: clasificación de riesgo

Las suites de integridad para entrenamiento y prueba aprobaron todas sus condiciones. La
validación entre ambas particiones también aprobó, sin hallazgos de nulos, duplicados,
inconsistencias de tipos o drift que impidan continuar con el modelo.

## Tratamiento

Los reportes HTML se generan en `reports/deepchecks/` y permanecen fuera de Git porque
pueden mostrar ejemplos derivados de los datos. El repositorio conserva únicamente las
pruebas automatizadas, esta interpretación y capturas sin registros individuales. Los
chequeos deben repetirse cuando cambie la versión de datos o antes de promover una nueva
versión del modelo.
