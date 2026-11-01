# Reporte de Seguridad, Análisis de Vulnerabilidades y Riesgo Residual

**Proyecto:** Sistema Inteligente para MiBici Guadalajara (TC5061 - Equipo 9)  
**Responsable:** David Carrera Castillo (A01840136) — Participación 3: Seguridad  
**Fecha:** Noviembre 2026  
**Entorno de ejecución:** Python 3.10 | macOS / Linux  

---

## 1. Resumen Ejecutivo y Alcance

Este documento formaliza la postura de seguridad, gobernanza y gestión de riesgos para el ciclo de vida de Machine Learning del proyecto MiBici, en cumplimiento con los requerimientos de la Semana 7 (Testing, Calidad de Datos, Seguridad y Gobernanza).

El alcance de seguridad cubre tres ejes complementarios:
1. **Protección de credenciales, secretos y privacidad de datos:** Aislamiento de variables de entorno mediante `.env.example`, robustecimiento de exclusiones en `.gitignore` y auditoría de ausencia de Datos Personales Identificables (PII) en los conjuntos de datos.
2. **Análisis Estático de Código (SAST) con Bandit:** Inspección sintáctica del árbol de sintaxis abstracta (AST) sobre 2,444 líneas de código para descartar vulnerabilidades comunes de inyección, deserialización insegura y manejo de procesos.
3. **Análisis de Composición de Software (SCA) con pip-audit:** Escaneo exhaustivo del árbol de dependencias frente a las bases de datos de vulnerabilidades conocidas de PyPI y OSV (Open Source Vulnerabilities), documentando las remediaciones aplicadas y el riesgo residual aceptado.

---

## 2. Protección de Credenciales, Secretos y Privacidad de Datos

### 2.1. Gestión de Variables de Entorno y Desacoplamiento de Secretos
Para evitar la exposición de credenciales o rutas locales en el control de versiones, se implementó una plantilla estandarizada [`.env.example`](../.env.example) en la raíz del repositorio. Esta plantilla documenta las variables configurables del sistema sin quemar secretos:
- `MLFLOW_TRACKING_URI`: Endpoint del servidor centralizado de MLflow (`http://127.0.0.1:5000`).
- `MLFLOW_EXPERIMENT_NAME`: Nombre del experimento (`mibici-equipo09`).
- `SEMILLA`: Semilla global determinista (`42`).
- `ENVIRONMENT`: Entorno operativo (`development`, `staging`, `production`).

### 2.2. Políticas de Exclusión en `.gitignore`
Se auditaron y reforzaron las reglas de [`.gitignore`](../.gitignore) para garantizar que ningún archivo sensible sea rastreado accidentalmente:
- **Archivos de entorno y secretos:** `.env`, `.env.*`, `credentials.json`, `client_secret*.json`, `token.json`, `secrets.json`.
- **Certificados y llaves criptográficas:** `*.pem`, `*.key`, `*.crt`, `*.pfx`, `*.p12`.
- **Bases de datos locales y tracking:** `mlflow.db`, `*.db`, `*.sqlite`, carpetas `mlruns/` y `mlartifacts/`.
- **Modelos serializados y datos crudos:** `models/*.pkl`, `models/*.joblib`, `*.parquet`, `datos_abiertos_*.csv`.
- **Reportes locales de diagnóstico:** `reports/deepchecks/` y `reports/security/`.

### 2.3. Evaluación de Privacidad y Ausencia de PII (Personally Identifiable Information)
Los datos de origen del sistema MiBici corresponden al portal de datos abiertos del Gobierno de Jalisco. Se verificó que los registros contienen exclusivamente datos operativos disociados:
- `viaje_id`, `origen_estacion_id`, `destino_estacion_id`, `fecha_hora`, `usuario_id` (identificador numérico seudonimizado), `genero` y `año_nacimiento`.
- **Ausencia de datos sensibles:** No se recopilan ni almacenan nombres de usuarios, números telefónicos, direcciones particulares, correos electrónicos, datos biométricos ni información financiera/tarjetas bancarias.
- **Conformidad:** El pipeline cumple con el principio de **minimización de datos** estipulado en marcos internacionales de privacidad (GDPR Art. 5 y regulaciones locales de protección de datos personales).

---

## 3. Análisis Estático de Código (SAST) con Bandit

Se ejecutó **Bandit v1.9.4** para escanear el código fuente del paquete `mibici/` y las pruebas en `tests/`.

### 3.1. Métricas Consolidadas del Escaneo

| Métrica | Valor | Observación |
| :--- | :---: | :--- |
| **Líneas de código escaneadas** | 2,444 | 2,092 en `mibici/` y 352 en `tests/` |
| **Archivos omitidos** | 0 | Cobertura total del proyecto |
| **Vulnerabilidades de Severidad Alta** | **0** | Ninguna brecha crítica detectada |
| **Vulnerabilidades de Severidad Media** | **0** | Sin fallas de riesgo medio |
| **Vulnerabilidades de Severidad Baja** | **0** | Todas las alertas bajas fueron auditadas y justificadas |
| **Estado final de salida** | **Éxito (Código 0)** | Limpio bajo la configuración formal `.bandit` |

### 3.2. Justificación Técnica de Reglas Auditadas y Falsos Positivos

Para el análisis se diseñó el archivo de configuración `.bandit`, documentando técnicamente las exclusiones aplicadas:

1. **`B101: assert_used`:**
   - *Contexto:* Bandit marca el uso de `assert` porque en compilaciones optimizadas (`python -O`) las aserciones se desactivan.
   - *Justificación:* En este proyecto, `assert` se utiliza exclusivamente dentro de la suite de pruebas automatizadas con **PyTest** (`tests/`) y como verificación defensiva de precondiciones en funciones internas. No se utiliza para validaciones críticas de control de acceso ni autenticación.
2. **`B404: import_subprocess`, `B603: subprocess_without_shell_equals_true`, `B607: start_process_with_partial_path`:**
   - *Contexto:* Bandit alerta sobre cualquier llamada a `subprocess` por el riesgo potencial de inyección de comandos en el sistema operativo (CWE-78).
   - *Justificación:* En `mibici/modeling/train.py`, `subprocess.check_output` se emplea exclusivamente para consultar el hash del commit actual de Git (`git rev-parse HEAD`) y el nombre del usuario (`git config user.name`), garantizando el linaje reproducible en MLflow. Las invocaciones se construyen como listas estáticas de cadenas (`["git", "rev-parse", "HEAD"]`), sin concatenación de cadenas, sin interacción con datos de usuario y con `shell=False` estricto, neutralizando cualquier posibilidad de inyección de comandos.

---

## 4. Auditoría de Dependencias (SCA) con pip-audit

Se utilizó **pip-audit v2.10.1** para auditar el entorno virtual y las dependencias declaradas en `requirements.txt`, consultando las bases de datos de vulnerabilidades conocidas de PyPI y OSV.

### 4.1. Remediación Inmediata de Dependencias Vulnerables

Se identificaron versiones con avisos de seguridad que fueron remediadas de forma inmediata mediante actualización en el entorno y en `requirements.txt`:

| Paquete | Versión Anterior | Versión Actualizada | Identificador / CVE | Impacto y Mitigación |
| :--- | :---: | :---: | :--- | :--- |
| **`certifi`** | `2026.7.22` | **`2026.10.23`** | PYSEC-2026-4320 | Actualización del almacén de certificados raíz de confianza para conexiones TLS/HTTPS seguras. |
| **`mlflow`** | `3.17.1` | **`3.17.2`** | CVE-2026-53995 | Corrección de vulnerabilidad en el servicio de artefactos y la interfaz gráfica de MLflow Tracking. |
| **`pyasn1`** | `0.6.4` | **`0.6.5`** | CVE-2026-119194 / 95 / 96 / 97 | Mitigación de posible denegación de servicio (DoS) por desbordamiento en la decodificación de estructuras ASN.1. |

---

## 5. Análisis y Documentación de Riesgo Residual

Tras la remediación, el escaneo de dependencias arrojó observaciones en tres paquetes específicos. Se presenta el análisis detallado de la causa raíz, superficie de ataque y controles compensatorios:

| Paquete | Versión Actual | Versión con Parche | Causa Raíz de No Actualización Inmediata | Superficie de Ataque y Análisis de Riesgo | Nivel de Riesgo Residual |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **`pillow`** | `12.3.0` | `13.0.0` | `Pillow >= 13.0.0` retiró soporte para Python 3.10 y requiere **`Python >= 3.11`**. | **Uso exclusivamente local:** `Pillow` es una dependencia transitiva de `matplotlib` y `deepchecks` utilizada únicamente para guardar figuras PNG (`fig.savefig()`). El sistema no expone servicios de subida ni procesamiento de imágenes arbitrarias provenientes de usuarios externos. | **Bajo** |
| **`tornado`** | `6.5.10` | `6.6.0` | `Tornado >= 6.6.0` requiere **`Python >= 3.11`**. | **Uso restringido a desarrollo:** `Tornado` es una dependencia del servidor local de Jupyter Notebook para exploración interactiva. En producción, el pipeline corre vía scripts de MLflow CLI, sin exponer el servidor Tornado a redes públicas. | **Bajo** |
| **`filelock`** | `4.1.0` | `4.2.1` | `filelock >= 4.2.1` requiere **`Python >= 3.11`**. | **Uso interno de caché:** Utilizado únicamente de forma local por `pip-audit` / `CacheControl` para gestionar el bloqueo de archivos de caché en disco durante la instalación. No tiene exposición remota. | **Despreciable** |

### Plan de Tratamiento y Controles Compensatorios
1. **Aislamiento de red:** Las tareas de entrenamiento, validación con DeepChecks e inferencia corren en entornos controlados sin exposición de puertos hacia redes externas no autenticadas.
2. **Entrada de datos controlada:** Los únicos datos procesados provienen de archivos locales validados con sumas de verificación SHA-256 (`manifiesto_datos.json`).
3. **Hoja de ruta tecnológica:** Para la siguiente iteración mayor del proyecto se programó la actualización del runtime base a **Python 3.11 / 3.12**, lo que permitirá absorber los parches mayores de `pillow 13.x` y `tornado 6.6x` sin fricción.

---

## 6. Alineación con Marcos de Seguridad y Gobernanza en IA

El diseño de seguridad del proyecto se alinea con los principales estándares internacionales:

### 6.1. OWASP Machine Learning Security Top 10
- **ML01: Input Manipulation / Evasion Attacks:** Mitigado mediante validaciones de tipo, rango de fechas y límites lógicos en `mibici/features.py` y las suites de integridad de DeepChecks (`mibici/quality.py`).
- **ML02: Data Poisoning Attacks:** Mitigado mediante el cálculo determinista del hash SHA-256 de los datos procesados en `references/manifiesto_datos.json` y el registro estricto del linaje vía `mlflow.log_input`.
- **ML05: Supply Chain Vulnerabilities:** Mitigado mediante la integración continua de `pip-audit` y el versionado estricto en `requirements.txt`.
- **ML09: Failure to Restrict Model Output / Data Leakage:** Los modelos solo emiten agregados horarios (P1), probabilidades de desabasto por estación (P2) y etiquetas de cluster (P3). Ninguna predicción permite la reconstrucción de trayectorias individuales ni la reidentificación de ciclistas.

### 6.2. NIST AI Risk Management Framework (NIST AI 100-1 / AI RMF)
- **Función GOVERN:** Definición de roles colaborativos, revisión por pares en Pull Requests y documentación transparente de limitaciones técnicas y de seguridad.
- **Función MAP:** Mapeo de dependencias de software, componentes de datos y riesgos residuales documentados en esta referencia.
- **Función MEASURE:** Evaluación cuantitativa de calidad de código mediante Bandit (0 brechas críticas) y pruebas unitarias de integridad con PyTest (40/40 pruebas aprobadas).
- **Función MANAGE:** Plan de tratamiento de riesgos residuales y controles compensatorios implementados.

---

## 7. Instrucciones de Reproducibilidad y Automatización

Para que cualquier integrante del equipo o el pipeline automatizado de CI ejecute las pruebas de seguridad:

```bash
# 1. Ejecutar el análisis estático de código con Bandit
make security
# O directamente vía CLI:
bandit -c .bandit -r mibici/ tests/

# 2. Ejecutar la auditoría de dependencias con pip-audit
pip-audit --local

# 3. Generar los reportes detallados en archivo
make security-report
```

Los reportes locales se almacenan en `reports/security/bandit_report.txt` y `reports/security/pip_audit_report.md` para propósitos de auditoría y evidencia del equipo.
