# Análisis FinOps del servicio de predicción de P1

## 1. Objetivo y alcance

Este documento estima cuánto cuesta cada inferencia del servicio de predicción de P1 (`POST /predict/p1`) en dos formas de desplegar la API en AWS: un servidor dedicado en EC2 y una función serverless en Lambda. Con esas estimaciones se calcula a partir de qué volumen conviene cada opción y se proponen optimizaciones respaldadas por las cifras.

El costo por inferencia considera solo el contenedor de la API. La imagen empaqueta el champion de P1 en `models/p1/` y la API lo carga una sola vez al arrancar (`lifespan` en `app/main.py`). `docker-compose.yml` y el `Dockerfile` fijan `MODEL_URI=/srv/models/p1`, así que `ModelService` lee los metadatos del `metadata.json` de esa carpeta y no crea un cliente de MLflow. `MlflowClient` solo se usa cuando `MODEL_URI` empieza con `models:/`, y aun así únicamente al cargar el modelo. Para responder, `predict_p1` llama a `ModelService.predict`, que solo ejecuta el modelo en memoria. El servidor de MLflow no participa en ninguna respuesta y queda fuera del costo.

## 2. Datos de entrada

### 2.1. Prueba de carga

Las cifras de desempeño vienen de `reports/load_testing/load_test_summary.json`: una prueba de `POST /predict/p1` con Locust que subió de 5 a 100 usuarios concurrentes en 357 segundos. Cada solicitud pide una sola predicción.

| Métrica | Valor |
| :--- | ---: |
| Solicitudes totales | 13,565 |
| Solicitudes exitosas | 13,565 |
| Solicitudes fallidas | 0 |
| Tasa de error | 0.00% |
| RPS promedio | 37.5929 |
| Latencia promedio | 17.736 ms |
| Latencia p50 | 11 ms |
| Latencia p95 | 43 ms |
| Latencia p99 | 180 ms |

Como no hubo fallas, las solicitudes exitosas y las totales coinciden, y el costo por inferencia útil es igual al costo por solicitud.

### 2.2. Consumo de recursos de la API

`docker-compose.yml` limita el contenedor `mibici-api` a 1 CPU (`cpus: 1.0`) y 1 GiB de memoria (`mem_limit: 1g`). Los porcentajes de `docker stats` se refieren a esos límites: 100% de CPU equivale a una CPU completa y 100% de memoria a 1,024 MiB. Se tomaron 72 muestras durante la prueba.

| Recurso | Porcentaje medido | Equivalente |
| :--- | ---: | ---: |
| CPU promedio | 26.068% | 0.26 vCPU |
| CPU máxima | 60.32% | 0.60 vCPU |
| Memoria promedio | 16.086% | 164.7 MiB |
| Memoria máxima | 16.47% | 168.7 MiB |

### 2.3. Demanda real de P1

P1 genera una predicción por estación y hora para 484 estaciones. Son 24 predicciones por estación al día y, en meses de 30 días, 348,480 predicciones al mes. En promedio son 484 predicciones por hora, equivalentes a 0.134 RPS.

## 3. Opciones de despliegue y precios de referencia

Todos los precios corresponden a AWS en la región US East (N. Virginia), bajo demanda y en USD.

### 3.1. Servidor dedicado: EC2, Linux

Precios publicados por AWS el 21 de noviembre de 2026 en la página de precios bajo demanda de EC2. Se toman 730 horas por mes.

| Instancia | USD por hora | vCPU | Memoria | Arquitectura |
| :--- | ---: | ---: | ---: | :--- |
| t3.small | 0.0208 | 2 | 2 GiB | x86 |
| t4g.small | 0.0168 | 2 | 2 GiB | Arm |
| t4g.micro | 0.0084 | 2 | 1 GiB | Arm |

### 3.2. Serverless: Lambda, x86

Precios consultados el 21 de noviembre de 2026 en la página de precios de Lambda.

| Concepto | Valor |
| :--- | :--- |
| Peticiones | 0.20 USD por millón |
| Cómputo | 0.0000166667 USD por GB-segundo |
| Nivel gratuito mensual | 1,000,000 peticiones y 400,000 GB-segundos |
| Facturación de la duración | Por milisegundo, redondeando hacia arriba |
| Memoria configurable | De 128 MB a 10,240 MB; la CPU es proporcional a la memoria |
| Memoria supuesta | 1,024 MB (1 GB), igual al límite de la API en Compose |
| Exposición HTTP | Function URL, sin costo adicional |

## 4. Supuestos y exclusiones

Supuestos:

- El costo de EC2 es fijo: la instancia se paga las 730 horas del mes, se use o no.
- Una instancia EC2 atiende al menos el RPS promedio medido en la prueba. La prueba corrió con 1 CPU y cada instancia ofrece 2 vCPU.
- En Lambda, la duración facturada de cada petición es la latencia promedio medida desde el cliente, redondeada al milisegundo superior. La p95 y la p99 se usan como análisis de sensibilidad.
- La demanda real es de 348,480 predicciones al mes, cada una en una solicitud distinta, como en la prueba.

Se excluyen:

- almacenamiento: volúmenes EBS de EC2 y el registro de imágenes ECR;
- transferencia de datos;
- impuestos;
- arranques en frío de Lambda;
- el servidor de MLflow, que no participa en las respuestas (sección 1).

## 5. Método

**EC2.**

- Costo mensual = precio por hora × 730.
- Capacidad medida por hora = RPS promedio × 3,600.
- Costo por millón a la capacidad medida = precio por hora ÷ capacidad medida por hora × 1,000,000.
- Costo por millón con la demanda real = costo mensual ÷ 348,480 × 1,000,000.
- Utilización = demanda real por hora ÷ capacidad medida por hora = 484 ÷ capacidad medida por hora.

**Lambda.**

- Duración facturada = latencia redondeada hacia arriba al milisegundo.
- GB-segundos por petición = duración facturada en segundos × 1 GB.
- Costo por petición = 0.20 ÷ 1,000,000 + GB-segundos por petición × 0.0000166667.
- Costo por millón = costo por petición × 1,000,000.
- Costo mensual sin nivel gratuito = 348,480 × costo por petición.
- Costo mensual con nivel gratuito = máx(0, peticiones − 1,000,000) × 0.20 ÷ 1,000,000 + máx(0, GB-segundos del mes − 400,000) × 0.0000166667.

**Punto de equilibrio, sin nivel gratuito.**

- Predicciones por mes = costo mensual de la instancia ÷ costo por petición en Lambda.
- RPS promedio sostenido = predicciones por mes ÷ (730 × 3,600). Se usan las mismas 730 horas con que se factura EC2.
- Distancia a la demanda real = predicciones por mes del equilibrio ÷ 348,480.

## 6. Resultados

### 6.1. EC2

La capacidad medida es de 37.5929 × 3,600 = 135,334 predicciones por hora, igual para las tres instancias porque viene de la misma prueba.

| Instancia | Costo mensual (USD) | Capacidad medida (pred/h) | Costo por millón a la capacidad medida (USD) | Costo por millón con la demanda real (USD) | Utilización |
| :--- | ---: | ---: | ---: | ---: | ---: |
| t3.small | 15.18 | 135,334 | 0.1537 | 43.57 | 0.36% |
| t4g.small | 12.26 | 135,334 | 0.1241 | 35.19 | 0.36% |
| t4g.micro | 6.13 | 135,334 | 0.0621 | 17.60 | 0.36% |

Con la demanda real, la instancia trabaja 0.36% de lo que mostró poder atender. Por eso el costo por millón con la demanda real es 283.5 veces el costo a la capacidad medida: casi todo lo que se paga es capacidad ociosa.

### 6.2. Lambda

| Latencia usada | Duración facturada | GB-s por petición | Costo por petición (USD) | Costo por millón (USD) | Mensual con la demanda real, sin nivel gratuito (USD) | Mensual con la demanda real, con nivel gratuito (USD) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Promedio (17.736 ms) | 18 ms | 0.018 | 0.00000050 | 0.50 | 0.17 | 0.00 |
| p95 (43 ms) | 43 ms | 0.043 | 0.00000092 | 0.92 | 0.32 | 0.00 |
| p99 (180 ms) | 180 ms | 0.180 | 0.00000320 | 3.20 | 1.12 | 0.00 |

Con la latencia promedio, el cargo por petición es 40% del costo de cada predicción y el cómputo el 60% restante. La demanda real suma 6,273 GB-segundos al mes, 1.6% de los 400,000 gratuitos, y 348,480 peticiones, 34.8% del millón gratuito. Aun con la p99, el mes consume 62,726 GB-segundos y cabe en el nivel gratuito.

### 6.3. Comparación con la demanda real

| Opción | Costo mensual (USD) | Costo por millón (USD) |
| :--- | ---: | ---: |
| Lambda, con nivel gratuito | 0.00 | 0.00 |
| Lambda, sin nivel gratuito | 0.17 | 0.50 |
| EC2 t4g.micro | 6.13 | 17.60 |
| EC2 t4g.small | 12.26 | 35.19 |
| EC2 t3.small | 15.18 | 43.57 |

Sin nivel gratuito, Lambda cuesta 2.8% de lo que cuesta la t4g.micro, la instancia más barata.

## 7. Escalabilidad

### 7.1. Punto de equilibrio

El costo de EC2 no cambia con el volumen y el de Lambda crece en proporción a las peticiones. El punto de equilibrio es el volumen en que ambos cuestan lo mismo. Sin nivel gratuito y con la latencia promedio:

| Instancia | Costo mensual (USD) | Equilibrio (pred/mes) | RPS promedio sostenido | Veces la demanda real |
| :--- | ---: | ---: | ---: | ---: |
| t4g.micro | 6.13 | 12,263,985 | 4.67 | 35.2 |
| t4g.small | 12.26 | 24,527,971 | 9.33 | 70.4 |
| t3.small | 15.18 | 30,367,964 | 11.56 | 87.1 |

La demanda real está 35 veces por debajo del equilibrio más cercano, el de la t4g.micro. Mientras el volumen mensual no pase de unos 12.3 millones de predicciones, Lambda es la opción más barata.

### 7.2. Sensibilidad a la duración facturada

Si la duración facturada fuera mayor que la latencia promedio, Lambda se encarecería y el equilibrio bajaría:

| Instancia | Equilibrio con p95 (pred/mes) | Veces la demanda real | Equilibrio con p99 (pred/mes) | Veces la demanda real |
| :--- | ---: | ---: | ---: | ---: |
| t4g.micro | 6,689,444 | 19.2 | 1,916,246 | 5.5 |
| t4g.small | 13,378,888 | 38.4 | 3,832,493 | 11.0 |
| t3.small | 16,564,338 | 47.5 | 4,744,991 | 13.6 |

Aun en el caso extremo de facturar todas las peticiones con la p99, la demanda real queda 5.5 veces por debajo del equilibrio con la t4g.micro. La conclusión no depende de qué latencia se use.

### 7.3. Con nivel gratuito

El nivel gratuito desplaza el equilibrio hacia arriba. Con 18 ms por petición, el millón de peticiones gratuitas se agota antes que los GB-segundos, que alcanzan para 22.2 millones de peticiones. Calculado con nivel gratuito, el equilibrio sube a 25,997,329 predicciones al mes (9.89 RPS) contra la t4g.micro, 38,261,314 (14.56 RPS) contra la t4g.small y 44,101,307 (16.78 RPS) contra la t3.small.

### 7.4. Qué opción conviene según el volumen

- **Hasta 12.3 millones de predicciones al mes (4.67 RPS sostenidos):** Lambda.
- **De 12.3 millones a por lo menos 98.8 millones al mes (37.59 RPS sostenidos):** una t4g.micro. Esa capacidad medida es 8.1 veces el RPS del equilibrio, así que una sola instancia cubre el rango completo en que EC2 empieza a convenir.
- **Por encima de 98.8 millones al mes:** la prueba no llegó a ese volumen. Habría que medir la capacidad real de la instancia antes de decidir entre una instancia más grande o varias instancias.

Como referencia, a la capacidad medida sostenida todo el mes (98,794,141 predicciones), Lambda costaría 49.40 USD contra 6.13 USD de la t4g.micro.

## 8. Optimizaciones

1. **Desplegar en Lambda mientras la demanda sea la actual.** Con la demanda real, Lambda cuesta 0.00 USD al mes con nivel gratuito y 0.17 USD sin él, contra 6.13 USD de la t4g.micro (sección 6.3). La demanda está 35.2 veces por debajo del equilibrio y 5.5 veces aun con la p99 (sección 7.2). Para usar Function URL, la API de FastAPI necesita un adaptador que traduzca los eventos de Lambda a peticiones HTTP; el modelo y el código de predicción no cambian.

2. **Si se requiere un servidor dedicado, usar t4g.micro.** Cuesta 59.6% menos que la t3.small (6.13 contra 15.18 USD al mes) y 50% menos que la t4g.small. Con la demanda real, cualquiera de las tres trabaja al 0.36% de su capacidad medida (sección 6.1), y el consumo de la API cabe en la t4g.micro (sección 2.2): 0.60 vCPU como máximo de 2 y 168.7 MiB de memoria de 1 GiB. Requiere construir la imagen para Arm, porque la t4g no ejecuta la imagen x86 actual.

3. **Revisar la decisión con un umbral explícito.** Conviene pasar de Lambda a una t4g.micro cuando el volumen mensual supere 12.3 millones de predicciones, o unos 4.67 RPS sostenidos (sección 7.1). Para vigilarlo basta registrar el número mensual de peticiones y la duración facturada promedio.

4. **Medir una memoria menor para Lambda.** La API usó como máximo 168.7 MiB, 16.5% de los 1,024 MB supuestos (sección 2.2), y el cómputo es 60% del costo por predicción (sección 6.2). Reducir la memoria baja el precio de cada milisegundo en la misma proporción, pero también reduce la CPU y puede alargar la duración. Solo conviene si una prueba confirma que el producto de memoria por duración baja.

## 9. Limitaciones

- La prueba se corrió en una computadora local con arquitectura x86, no en la nube. La latencia y la capacidad pueden ser distintas en una instancia EC2 o en Lambda, y en las instancias t4g, de arquitectura Arm, no se midieron.
- La API no llegó a saturarse: la CPU máxima fue de 60.32% de su límite de 1 CPU y no hubo errores. La capacidad real es mayor que los 37.59 RPS medidos, así que el costo por inferencia a la capacidad medida es un tope y no el costo mínimo alcanzable.
- Los arranques en frío de Lambda no se midieron. Con una demanda baja y repartida a lo largo del día, buena parte de las peticiones podría encontrar la función inactiva. Eso alarga la duración facturada y la latencia que percibe el cliente.
- Con 1,024 MB, Lambda asigna una fracción de CPU menor que la CPU completa que tuvo la API en Compose. La latencia en Lambda puede ser mayor que la medida y, con ella, la duración facturada.
- La latencia medida desde el cliente incluye la red y el procesamiento HTTP, así que es una aproximación conservadora de la duración facturable. Pesa en contra de esa ventaja que tampoco incluye los arranques en frío ni el efecto de la CPU de Lambda.
- Las instancias t3 y t4g son de rendimiento ampliable y administran la CPU con créditos. Este análisis no modela ese mecanismo. Con la utilización estimada de 0.36%, la API debería quedar muy por debajo de su uso sostenido, pero no se comprobó.
- La demanda real se tomó como un promedio de 484 predicciones por hora. No se sabe si llegan repartidas o todas juntas al inicio de cada hora; si llegan juntas, el pico de RPS es mayor que el promedio.
- El análisis excluye almacenamiento, transferencia de datos e impuestos (sección 4). Suman a las dos opciones y pueden mover el punto de equilibrio.

## 10. Fuentes

- Precios bajo demanda de Amazon EC2, consultados el 21 de noviembre de 2026: https://aws.amazon.com/ec2/pricing/on-demand/
- Precios de AWS Lambda, consultados el 21 de noviembre de 2026: https://aws.amazon.com/lambda/pricing/
- Anuncio de Function URLs de AWS Lambda: https://aws.amazon.com/about-aws/whats-new/2022/04/aws-lambda-function-urls-built-in-https-endpoints/
- Resultados de la prueba de carga: `reports/load_testing/load_test_summary.json`
- Límites del contenedor de la API: `docker-compose.yml`
