# Guía operativa y guion de comandos para la demostración final

Este documento concentra la verificación de **despliegue, operación y demostración** (FastAPI, Docker Compose, Locust y monitoreo de recursos) y los comandos listos para manejar el teclado durante la grabación del video final (de 2 a 5 minutos).

---

## 1. Preparación antes de iniciar la grabación (Pre-flight)

Ejecutar estos pasos **antes** de comenzar a grabar pantalla para que las imágenes ya estén construidas y ambos contenedores estén saludables desde el segundo cero:

```bash
# 1. Validar configuración y levantar API (:8000) + MLflow (:5001)
docker compose config -q
docker compose up -d --build --wait

# 2. Confirmar los 10 smoke tests y la verificación operativa en vivo
make smoke-test
python -m mibici.operacion_final --verificar-vivo
```

### Pestañas recomendadas listas en el navegador
- **Swagger / OpenAPI de FastAPI:** `http://localhost:8000/docs`
- **Interfaz de MLflow:** `http://localhost:5001`
- **Terminal limpia** (letra grande, fondo de alto contraste, ubicada en la raíz del repositorio).

> **Importante:** Las evidencias oficiales de la prueba de carga de 6 minutos están preservadas en `reports/load_testing/`. Si deseas correr una prueba rápida de Locust antes de grabar, redirige siempre la salida a una carpeta temporal para no sobrescribir los artefactos oficiales:
>
> ```bash
> LOAD_OUTPUT_DIR=/tmp/mibici_load_smoke make load-test-smoke
> ```

---

## 2. Guion de teclado para la demo en vivo (~60 a 90 segundos)

Para evitar errores de escritura durante la grabación continua, el proyecto incluye el script [`scripts/demo_final.sh`](../scripts/demo_final.sh):

- **Modo narrado paso a paso** (pausa con `Enter` entre cada bloque):
  ```bash
  ./scripts/demo_final.sh --pausa
  ```
- **Modo continuo sin pausas:**
  ```bash
  make demo-final
  ```

### Secuencia que muestra en pantalla

| Paso | Tiempo sugerido | Qué se ejecuta | Qué demuestra en el video |
| :--- | :---: | :--- | :--- |
| **1. Orquestación** | 10 s | `docker compose ps` | `mibici-api` (`:8000`) y `mibici-mlflow` (`:5001`) activos en estado `healthy` sobre la red dedicada. |
| **2. Salud y linaje** | 15 s | `GET /health/ready`, `GET /model-info` y `MLflow /health` | El servicio tiene cargado el modelo `mibici-equipo09-p1-demanda`, alias `champion`, versión `1`, y MLflow responde `200`. |
| **3. Predicción real** | 20 s | `POST /predict/p1` (estación 2, martes 8:00 h) | Respuesta JSON con demanda pronosticada (~3.61 viajes/h), linaje del modelo y latencia en milisegundos. |
| **4. Validación Pydantic** | 10 s | `POST /predict/p1` con `hora: 24` | Rechazo controlado con `HTTP 422` y detalle del campo inválido. |
| **5. Recursos en vivo** | 10 s | `docker stats --no-stream` | Consumo acotado de CPU y memoria dentro de los límites de `1 vCPU` y `1 GB RAM`. |
| **6. Evidencia Locust** | 15 s | Lectura de `reports/load_testing/load_test_summary.json` | Cifras oficiales de desempeño bajo carga sostenida (0% errores, 37.59 RPS, p95 de 43 ms). |

*(Opcional)* Si el equipo decide incluir tráfico en vivo durante la grabación, `./scripts/demo_final.sh --con-carga` añade un burst de 20 s de Locust hacia `/tmp/` sin alterar `reports/load_testing/`.

---

## 3. Evidencias preservadas de carga y recursos

Las evidencias de la prueba escalonada de 6 minutos (de 5 a 100 usuarios concurrentes) permanecen versionadas y auditadas en [`reports/final/operacion_evidencia.json`](../reports/final/operacion_evidencia.json):

| Indicador | Resultado medido | Artefacto de respaldo |
| :--- | :--- | :--- |
| Endpoint evaluado | `POST /predict/p1` | `locustfile.py` |
| Solicitudes totales / fallidas | **13,565** / **0** (**0.0%** error) | `reports/load_testing/load_test_summary.json` |
| Throughput promedio | **37.59 RPS** (357 s) | `reports/load_testing/locust_performance.png` |
| Latencia p50 / p95 / p99 | **11.0 ms** / **43.0 ms** / **180.0 ms** | `reports/load_testing/locust_performance.png` |
| CPU `mibici-api` (prom / máx) | **26.07%** / **60.32%** (límite 1 vCPU) | `reports/load_testing/resource_usage.png` |
| RAM `mibici-api` (prom / máx) | **16.09%** / **16.47%** (límite 1 GB) | `reports/load_testing/resources.csv` |
| CPU / RAM `mibici-mlflow` (prom) | **5.18%** CPU / **99.01%** RAM (sin carga) | `reports/load_testing/resources.csv` |

---

## 4. Cierre y limpieza después de grabar

Para apagar los contenedores conservando el volumen persistente de MLflow:

```bash
make compose-down
```
