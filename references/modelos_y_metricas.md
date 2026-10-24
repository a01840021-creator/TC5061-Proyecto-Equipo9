# Modelos Candidatos, Líneas Base y Métricas de Evaluación

**Proyecto MiBici — Equipo 9**  
Integrantes: Mariana Judith Carrera Rojas, José Manuel Barrera Mendoza, Luis Fernando Barranco Cruz, David Carrera Castillo.

Este documento formaliza para cada una de las tres preguntas de negocio:
1. La **línea base (baseline)** y su justificación.
2. Al menos un **modelo candidato** para iteraciones futuras.
3. Las **métricas de evaluación** seleccionadas (guía de scikit-learn) y su coherencia con la decisión operativa a tomar.

---

## Pregunta 1 (P1): Demanda de Viajes por Estación y Hora

### 1. Definición y Tarea
- **Tipo de problema:** Regresión supervisada.
- **Variable objetivo:** `demanda` (entero $\ge 0$, número de viajes iniciados en la estación $i$ en la hora $h$).
- **Granularidad y horizonte:** Estación $\times$ hora; predicción para las 24 horas del día siguiente ($d$) con corte a las 23:59 del día anterior ($d-1$).

### 2. Líneas Base y Modelo Candidato
- **Baseline Simple (Mínimo):** `DummyRegressor(strategy="mean")`. Predice la media histórica global de salidas por estación-hora. Permite calibrar la varianza mínima explicada.
- **Baseline Estructurado (Implementado en MLflow):** `Ridge Regression` con regularización $L_2$ ($\alpha=1.0$) sobre variables de calendario (hora, día de semana, fin de semana), capacidad de la estación y rezagos autorregresivos estrictos ($t-24\text{h}$, $t-48\text{h}$, $t-168\text{h}$).
- **Modelos Candidatos Futuros:**
  1. *Gradient Boosting / HistGradientBoostingRegressor / LightGBM:* Captura no-linealidades complejas entre capacidad instalada, ubicación geográfica y patrones horarios.
  2. *Poisson / Negative Binomial Regressor:* Modela explícitamente datos de conteo con alta proporción de ceros (61% de las observaciones).

### 3. Métricas de Evaluación y Justificación
- **`test_mae` (Error Absoluto Medio):** **Métrica primaria**. Expresa el error en las mismas unidades de la variable objetivo (viajes). Para el operador de flota, un MAE de 0.9 viajes por estación-hora es intuitivo y se traduce directamente en cuántas bicicletas de sobra o de falta se asignan a una estación.
- **`test_rmse` (Raíz del Error Cuadrático Medio):** Penaliza fuertemente errores grandes. Es crítico en horas pico donde subestimar la demanda por 10 bicicletas colapsa la estación.
- **`test_r2` (Coeficiente de Determinación):** Proporción de la varianza temporal explicada frente al promedio.

---

## Pregunta 2 (P2): Riesgo de Quedarse sin Bicicletas o sin Anclajes

### 1. Definición y Tarea
- **Tipo de problema:** Clasificación binaria supervisada (con desbalance de clases moderado: ~5% de positivos en prueba).
- **Variable objetivo:** `riesgo` $\in \{0, 1\}$. Vale `1` si el saldo neto acumulado en las siguientes 2 horas ($|\text{llegadas}_{t+1, t+2} - \text{salidas}_{t+1, t+2}|$) es $\ge 30\%$ de la capacidad total de la estación (`dpcapacity`), y `0` en otro caso.
- **Granularidad y horizonte:** Estación $\times$ hora; ventana móvil de 2 horas hacia adelante.

### 2. Líneas Base y Modelo Candidato
- **Baseline Simple (Mínimo):** `DummyClassifier(strategy="most_frequent")`. Predice siempre la clase mayoritaria (0: sin riesgo). Refleja que un modelo trivial alcanza un accuracy del ~95% sin detectar ninguna estación crítica.
- **Baseline Estructurado (Implementado en MLflow):** `LogisticRegression(class_weight="balanced")` sobre actividad reciente en $t$, rezago $t-1$, media móvil de saldo en 3 horas, capacidad de la estación y variables temporales.
- **Modelos Candidatos Futuros:**
  1. *Balanced Random Forest / HistGradientBoostingClassifier:* Árboles de decisión con muestreo balanceado para capturar la interacción entre la capacidad y la hora del día.
  2. *XGBoost con focal loss o scale_pos_weight:* Maximiza el área bajo la curva Precision-Recall en escenarios desbalanceados.

### 3. Métricas de Evaluación y Justificación
- **`test_recall` (Sensibilidad):** **Métrica prioritaria**. Mide la fracción de estaciones verdaderamente en riesgo que el modelo logra detectar. Para el equipo de rebalanceo, no detectar una estación que se vaciará deja varados a los usuarios; el costo de un falso negativo es mayor al de un falso positivo.
- **`test_f1_score` (Media armónica de Precision y Recall):** Equilibra la detección de riesgo evitando que el operador envíe camiones a toda la ciudad indiscriminadamente (falsas alarmas).
- **`test_roc_auc` (Área bajo la curva ROC):** Mide la capacidad de discriminación del modelo a través de todos los umbrales de decisión posibles, independientemente de la probabilidad base de corte (0.5).
- **`test_accuracy`:** Se reporta como referencia, pero **no** se usa para tomar decisiones debido a la paradoja del desbalance de clases (un modelo constante logra 95% de exactitud con utilidad cero).

---

## Pregunta 3 (P3): Grupos de Estaciones con Patrones de Uso Parecidos

### 1. Definición y Tarea
- **Tipo de problema:** Aprendizaje no supervisado (Clustering).
- **Entidad analizada:** Estación (484 estaciones activas).
- **Variables de perfil:** 24 proporciones horarias de demanda, 1 proporción de actividad en fines de semana, y 1 ratio de flujo neto de salidas versus llegadas ($\text{salidas} / (\text{salidas} + \text{llegadas})$). Ajustado únicamente con datos de entrenamiento (abril a agosto).

### 2. Líneas Base y Modelo Candidato
- **Baseline Estructurado (Implementado en MLflow):** `KMeans(n_clusters=4, random_state=42)` con estandarización de variables (`StandardScaler`).
- **Modelos Candidatos Futuros:**
  1. *Gaussian Mixture Models (GMM):* Asignación probabilística suave (soft clustering), reconociendo que una estación puede tener naturaleza mixta (oficina y habitacional).
  2. *HDBSCAN (Hierarchical Density-Based Spatial Clustering):* Descubre clusters de densidad variable y aísla estaciones atípicas o periféricas como ruido sin forzarlas a pertenecer a un cluster.

### 3. Métricas de Evaluación y Justificación
- **`train_silhouette_score` (Coeficiente de Silueta):** Mide la cohesión interna de cada cluster comparada con la separación hacia el cluster más cercano (rango $[-1, 1]$).
- **`train_davies_bouldin_score`:** Razón entre la dispersión intra-cluster y la distancia inter-cluster. Valores más bajos indican agrupaciones más densas y mejor separadas.
- **`train_inertia`:** Suma de distancias al cuadrado hacia el centroide; útil para inspeccionar el punto de codo en la selección de $k$.
- **Inspección Cualitativa:** Coherencia de los perfiles horarios (pico matutino en zonas residenciales vs pico vespertino en zonas comerciales/financieras) y contigüidad geográfica en el mapa de estaciones.
