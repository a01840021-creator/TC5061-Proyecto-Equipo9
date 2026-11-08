#!/usr/bin/env bash
# ==============================================================================
# Script de pruebas integrales y de humo (Smoke Tests) para la API de MiBici
#
# Verifica el funcionamiento de los endpoints de salud, metadatos y predicción,
# comprobando códigos de estado HTTP, estructura JSON y validaciones de Pydantic.
# ==============================================================================

set -euo pipefail

# Configuración por defecto
API_URL="${API_URL:-http://localhost:8000}"
RUN_CONTAINER=false
CONTAINER_NAME="mibici-api-smoke"
IMAGE_NAME="mibici-api"
TIMEOUT_SEGUNDOS=30

# Colores para salida en terminal
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  VERDE="\033[0;32m"
  ROJO="\033[0;31m"
  AMARILLO="\033[1;33m"
  AZUL="\033[0;36m"
  NEGRITA="\033[1m"
  RESET="\033[0m"
else
  VERDE=""
  ROJO=""
  AMARILLO=""
  AZUL=""
  NEGRITA=""
  RESET=""
fi

mostrar_ayuda() {
  cat <<EOF
Uso: $0 [OPCIONES]

Ejecuta pruebas de humo contra la API de predicción de MiBici (FastAPI).

Opciones:
  -u, --url URL          URL base de la API (por defecto: http://localhost:8000)
  -c, --run-container    Construye y levanta temporalmente un contenedor Docker/Podman
  -t, --timeout SEC      Tiempo máximo de espera para arranque del contenedor (segundos, defecto: 30)
  -h, --help             Muestra esta ayuda y termina
EOF
}

# Procesar argumentos
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)
      API_URL="$2"
      shift 2
      ;;
    -c|--run-container)
      RUN_CONTAINER=true
      shift
      ;;
    -t|--timeout)
      TIMEOUT_SEGUNDOS="$2"
      shift 2
      ;;
    -h|--help)
      mostrar_ayuda
      exit 0
      ;;
    *)
      echo -e "${ROJO}Opción desconocida: $1${RESET}" >&2
      mostrar_ayuda
      exit 1
      ;;
  esac
done

CONTENEDOR_LEVANTADO=false

limpiar_contenedor() {
  if [ "$CONTENEDOR_LEVANTADO" = true ]; then
    echo -e "\n${AMARILLO}Deteniendo contenedor temporal '${CONTAINER_NAME}'...${RESET}"
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo -e "${VERDE}Contenedor temporal removido.${RESET}"
  fi
}

trap limpiar_contenedor EXIT INT TERM

if [ "$RUN_CONTAINER" = true ]; then
  echo -e "${AZUL}${NEGRITA}==> Levantando contenedor temporal para pruebas de humo...${RESET}"
  
  # Si ya existe un contenedor con ese nombre, eliminarlo
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  
  docker run -d --name "${CONTAINER_NAME}" -p 8000:8000 "${IMAGE_NAME}" >/dev/null
  CONTENEDOR_LEVANTADO=true
  
  echo -n "Esperando a que la API esté lista en ${API_URL}..."
  INICIO=$(date +%s)
  LISTO=false
  while [ $(( $(date +%s) - INICIO )) -lt "$TIMEOUT_SEGUNDOS" ]; do
    if curl -s -f "${API_URL}/health/ready" >/dev/null 2>&1; then
      LISTO=true
      break
    fi
    echo -n "."
    sleep 1
  done
  echo ""
  
  if [ "$LISTO" = false ]; then
    echo -e "${ROJO}Error: El contenedor no respondió en ${TIMEOUT_SEGUNDOS} segundos.${RESET}" >&2
    docker logs "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  echo -e "${VERDE}Contenedor listo y respondiendo.${RESET}"
fi

TOTAL_PRUEBAS=0
PRUEBAS_EXITOSAS=0
PRUEBAS_FALLIDAS=0

ejecutar_prueba() {
  local DESCRIPCION="$1"
  local METODO="$2"
  local RUTA="$3"
  local HTTP_ESPERADO="$4"
  local CUERPO_JSON="${5:-}"
  local PATRON_ESPERADO="${6:-}"

  TOTAL_PRUEBAS=$((TOTAL_PRUEBAS + 1))
  echo -e "\n${AZUL}[Prueba ${TOTAL_PRUEBAS}]${RESET} ${NEGRITA}${DESCRIPCION}${RESET}"
  echo "  Solicitud: ${METODO} ${API_URL}${RUTA}"

  local ARCHIVO_RESPUESTA
  ARCHIVO_RESPUESTA=$(mktemp)

  local CODIGO_HTTP
  if [ -n "$CUERPO_JSON" ]; then
    echo "  Payload: ${CUERPO_JSON}"
    CODIGO_HTTP=$(curl -s -o "$ARCHIVO_RESPUESTA" -w "%{http_code}" -X "$METODO" "${API_URL}${RUTA}" \
      -H "Content-Type: application/json" \
      -d "$CUERPO_JSON")
  else
    CODIGO_HTTP=$(curl -s -o "$ARCHIVO_RESPUESTA" -w "%{http_code}" -X "$METODO" "${API_URL}${RUTA}")
  fi

  local CONTENIDO_RESPUESTA
  CONTENIDO_RESPUESTA=$(cat "$ARCHIVO_RESPUESTA")
  rm -f "$ARCHIVO_RESPUESTA"

  echo "  Respuesta HTTP: ${CODIGO_HTTP}"
  echo "  Cuerpo: ${CONTENIDO_RESPUESTA}"

  local PASO=true

  if [ "$CODIGO_HTTP" -ne "$HTTP_ESPERADO" ]; then
    echo -e "  ${ROJO}FALLÓ: Código HTTP esperado ${HTTP_ESPERADO}, recibido ${CODIGO_HTTP}.${RESET}"
    PASO=false
  fi

  if [ -n "$PATRON_ESPERADO" ]; then
    if ! echo "$CONTENIDO_RESPUESTA" | grep -q "$PATRON_ESPERADO"; then
      echo -e "  ${ROJO}FALLÓ: La respuesta no contiene el patrón esperado: '${PATRON_ESPERADO}'.${RESET}"
      PASO=false
    fi
  fi

  if [ "$PASO" = true ]; then
    echo -e "  ${VERDE}PASÓ CORRECTAMENTE${RESET}"
    PRUEBAS_EXITOSAS=$((PRUEBAS_EXITOSAS + 1))
  else
    PRUEBAS_FALLIDAS=$((PRUEBAS_FALLIDAS + 1))
  fi
}

echo -e "\n${NEGRITA}======================================================================${RESET}"
echo -e "${NEGRITA}  SUITE DE PRUEBAS DE HUMO (SMOKE TESTS) - SERVICIO DE PREDICCIÓN   ${RESET}"
echo -e "${NEGRITA}  Destino: ${API_URL}${RESET}"
echo -e "${NEGRITA}======================================================================${RESET}"

# 1. Liveness probe
ejecutar_prueba \
  "Sonda de vivacidad (/health/live)" \
  "GET" \
  "/health/live" \
  200 \
  "" \
  '"status":"ok"'

# 2. Readiness probe
ejecutar_prueba \
  "Sonda de disponibilidad con modelo cargado (/health/ready)" \
  "GET" \
  "/health/ready" \
  200 \
  "" \
  '"model_status":"loaded"'

# 3. Healthcheck estándar
ejecutar_prueba \
  "Endpoint de salud general (/health)" \
  "GET" \
  "/health" \
  200 \
  "" \
  '"status":"ok"'

# 4. Metadatos del modelo cargado
ejecutar_prueba \
  "Consulta de metadatos del champion (/model-info)" \
  "GET" \
  "/model-info" \
  200 \
  "" \
  '"model_name":"mibici-equipo09-p1-demanda"'

# 5. Predicción normal P1 (hora pico día laboral)
ejecutar_prueba \
  "Predicción P1: Demanda válida en estación 2 (martes 8:00 hrs)" \
  "POST" \
  "/predict/p1" \
  200 \
  '{"estacion_id": 2, "dpcapacity": 15.0, "hora": 8, "dia_semana": 1, "es_fin_de_semana": 0, "rezago_24h": 3.0, "rezago_48h": 2.0, "rezago_168h": 4.0}' \
  '"prediction":'

# 6. Predicción normal P1 (fin de semana)
ejecutar_prueba \
  "Predicción P1: Demanda válida en estación 10 (sábado 14:00 hrs)" \
  "POST" \
  "/predict/p1" \
  200 \
  '{"estacion_id": 10, "dpcapacity": 20.0, "hora": 14, "dia_semana": 5, "es_fin_de_semana": 1, "rezago_24h": 5.0, "rezago_48h": 4.0, "rezago_168h": 6.0}' \
  '"prediction":'

# 7. Validación Pydantic: Hora fuera de rango (hora 24)
ejecutar_prueba \
  "Validación Pydantic: Rechazo con 422 por hora inválida (hora: 24)" \
  "POST" \
  "/predict/p1" \
  422 \
  '{"estacion_id": 2, "dpcapacity": 15.0, "hora": 24, "dia_semana": 1, "es_fin_de_semana": 0, "rezago_24h": 3.0, "rezago_48h": 2.0, "rezago_168h": 4.0}' \
  'Input should be less than or equal to 23'

# 8. Validación Pydantic: Inconsistencia entre día y fin de semana
ejecutar_prueba \
  "Validación Pydantic: Rechazo con 422 por inconsistencia fin de semana (martes con es_fin_de_semana=1)" \
  "POST" \
  "/predict/p1" \
  422 \
  '{"estacion_id": 2, "dpcapacity": 15.0, "hora": 8, "dia_semana": 1, "es_fin_de_semana": 1, "rezago_24h": 3.0, "rezago_48h": 2.0, "rezago_168h": 4.0}' \
  'es_fin_de_semana no coincide'

# 9. Validación Pydantic: Campo faltante requerido
ejecutar_prueba \
  "Validación Pydantic: Rechazo con 422 por campo faltante (sin rezago_168h)" \
  "POST" \
  "/predict/p1" \
  422 \
  '{"estacion_id": 2, "dpcapacity": 15.0, "hora": 8, "dia_semana": 1, "es_fin_de_semana": 0, "rezago_24h": 3.0, "rezago_48h": 2.0}' \
  'Field required'

# 10. Documentación Swagger disponible
ejecutar_prueba \
  "Documentación OpenAPI (/openapi.json)" \
  "GET" \
  "/openapi.json" \
  200 \
  "" \
  '"openapi":'

echo -e "\n${NEGRITA}======================================================================${RESET}"
echo -e "${NEGRITA}  RESUMEN DE PRUEBAS DE HUMO${RESET}"
echo -e "${NEGRITA}======================================================================${RESET}"
echo "Total de pruebas ejecutadas: ${TOTAL_PRUEBAS}"
echo -e "Pruebas exitosas:            ${VERDE}${PRUEBAS_EXITOSAS}${RESET}"
if [ "$PRUEBAS_FALLIDAS" -gt 0 ]; then
  echo -e "Pruebas fallidas:            ${ROJO}${PRUEBAS_FALLIDAS}${RESET}"
  echo -e "\n${ROJO}${NEGRITA}RESULTADO: FALLO EN AL MENOS UNA PRUEBA.${RESET}\n"
  exit 1
else
  echo -e "Pruebas fallidas:            0"
  echo -e "\n${VERDE}${NEGRITA}RESULTADO: TODAS LAS PRUEBAS PASARON EXITOSAMENTE (100%).${RESET}\n"
  exit 0
fi
