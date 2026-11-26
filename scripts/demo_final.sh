#!/usr/bin/env bash
# ==============================================================================
# Guion ejecutable de la demostración operativa en vivo (Proyecto Final MiBici)
#
# Recorre en ~60-90 segundos el despliegue con Docker Compose, salud de FastAPI
# y MLflow, linaje del champion, predicción real, validación Pydantic (422),
# consumo en vivo con docker stats y resumen de evidencias preservadas de Locust.
# ==============================================================================

set -euo pipefail

API_URL="${MIBICI_API_URL:-http://localhost:8000}"
MLFLOW_URL="${MIBICI_MLFLOW_URL:-http://localhost:5001}"
MODO_PAUSA=false
EJECUTAR_CARGA_SMOKE=false
PYTHON_BIN="${PYTHON_BIN:-$(if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)}"

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

Ejecuta la demostración operativa del proyecto final MiBici contra los servicios
levantados con Docker Compose (FastAPI en :8000 y MLflow en :5001).

Opciones:
  --pausa                Espera Enter entre cada paso (ideal al narrar la grabación)
  --rapido               Ejecuta todos los pasos sin pausas (por omisión)
  --con-carga            Corre además un burst corto de Locust (20 s) hacia /tmp sin
                         sobrescribir las evidencias oficiales en reports/load_testing/
  --url URL              URL de la API FastAPI (defecto: http://localhost:8000)
  --mlflow-url URL       URL del servidor MLflow (defecto: http://localhost:5001)
  -h, --help             Muestra esta ayuda y termina
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pausa)
      MODO_PAUSA=true
      shift
      ;;
    --rapido)
      MODO_PAUSA=false
      shift
      ;;
    --con-carga)
      EJECUTAR_CARGA_SMOKE=true
      shift
      ;;
    --url)
      API_URL="$2"
      shift 2
      ;;
    --mlflow-url)
      MLFLOW_URL="$2"
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

pausar_si_aplica() {
  if [ "$MODO_PAUSA" = true ]; then
    echo -e "\n${AMARILLO}[Presiona Enter para continuar al siguiente paso...]${RESET}"
    read -r _
  fi
}

imprimir_json() {
  "$PYTHON_BIN" -m json.tool 2>/dev/null || cat
}

paso() {
  echo -e "\n${AZUL}${NEGRITA}======================================================================${RESET}"
  echo -e "${AZUL}${NEGRITA}  $1${RESET}"
  echo -e "${AZUL}${NEGRITA}======================================================================${RESET}"
}

# Prechequeo: los servicios deben estar activos para no perder tiempo de video en el build
if ! curl -sf "${API_URL}/health/ready" >/dev/null 2>&1; then
  echo -e "${ROJO}La API no responde en ${API_URL}/health/ready.${RESET}" >&2
  echo "Levanta primero el stack antes de iniciar la grabación:" >&2
  echo "  docker compose up -d --build --wait" >&2
  exit 1
fi

paso "1/6 - Orquestación activa con Docker Compose (FastAPI + MLflow)"
docker compose ps
pausar_si_aplica

paso "2/6 - Salud del servicio y linaje del modelo champion"
echo -e "${NEGRITA}GET ${API_URL}/health/ready${RESET}"
curl -sf "${API_URL}/health/ready" | imprimir_json
echo -e "\n${NEGRITA}GET ${API_URL}/model-info${RESET}"
curl -sf "${API_URL}/model-info" | imprimir_json
echo -e "\n${NEGRITA}GET ${MLFLOW_URL}/health${RESET}"
MLFLOW_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${MLFLOW_URL}/health")
echo -e "MLflow HTTP status: ${VERDE}${MLFLOW_STATUS}${RESET}"
test "$MLFLOW_STATUS" -eq 200
pausar_si_aplica

paso "3/6 - Predicción real en vivo (POST /predict/p1, martes 8:00 hrs)"
PAYLOAD_VALIDO='{"estacion_id": 2, "dpcapacity": 15.0, "hora": 8, "dia_semana": 1, "es_fin_de_semana": 0, "rezago_24h": 3.0, "rezago_48h": 2.0, "rezago_168h": 4.0}'
echo "Payload: ${PAYLOAD_VALIDO}"
curl -sf -X POST "${API_URL}/predict/p1" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD_VALIDO}" | imprimir_json
pausar_si_aplica

paso "4/6 - Validación estricta de contrato con Pydantic (hora=24 -> HTTP 422)"
PAYLOAD_INVALIDO='{"estacion_id": 2, "dpcapacity": 15.0, "hora": 24, "dia_semana": 1, "es_fin_de_semana": 0, "rezago_24h": 3.0, "rezago_48h": 2.0, "rezago_168h": 4.0}'
echo "Payload inválido: ${PAYLOAD_INVALIDO}"
TMP_ERR=$(mktemp)
HTTP_ERR=$(curl -s -o "$TMP_ERR" -w "%{http_code}" -X POST "${API_URL}/predict/p1" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD_INVALIDO}")
echo -e "Código HTTP recibido: ${VERDE}${HTTP_ERR}${RESET}"
imprimir_json < "$TMP_ERR"
rm -f "$TMP_ERR"
test "$HTTP_ERR" -eq 422
pausar_si_aplica

paso "5/6 - Monitoreo de recursos en vivo (docker stats)"
docker stats --no-stream \
  --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" \
  mibici-api mibici-mlflow
pausar_si_aplica

paso "6/6 - Resumen de evidencia oficial de carga (Locust + docker stats)"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

datos = json.loads(Path("reports/load_testing/load_test_summary.json").read_text(encoding="utf-8"))
api = datos["recursos_por_contenedor"]["mibici-api"]
mlf = datos["recursos_por_contenedor"]["mibici-mlflow"]
print(f"Endpoint evaluado:     {datos['endpoint']}")
print(f"Solicitudes totales:   {datos['solicitudes_totales']:,} (fallidas: {datos['solicitudes_fallidas']}, error: {datos['tasa_error_pct']}%)")
print(f"Throughput promedio:   {datos['rps_promedio']:.2f} RPS (duración: {datos['duracion_segundos']:.0f} s)")
print(f"Latencia (p50/p95/p99): {datos['latencia_p50_ms']:.1f} ms / {datos['latencia_p95_ms']:.1f} ms / {datos['latencia_p99_ms']:.1f} ms")
print(f"CPU mibici-api:        prom {api['cpu_promedio_pct']:.2f}% | máx {api['cpu_maximo_pct']:.2f}%")
print(f"RAM mibici-api:        prom {api['memoria_promedio_pct']:.2f}% | máx {api['memoria_maxima_pct']:.2f}%")
print(f"CPU mibici-mlflow:     prom {mlf['cpu_promedio_pct']:.2f}% | máx {mlf['cpu_maximo_pct']:.2f}%")
PY

if [ "$EJECUTAR_CARGA_SMOKE" = true ]; then
  paso "Extra - Burst en vivo con Locust (20 s hacia carpeta temporal)"
  TMP_LOAD=$(mktemp -d /tmp/mibici_demo_load.XXXXXX)
  trap 'rm -rf "$TMP_LOAD"' EXIT
  MIBICI_LOAD_PROFILE=smoke MIBICI_API_URL="$API_URL" LOAD_OUTPUT_DIR="$TMP_LOAD" \
    ./scripts/run_load_test.sh
  echo -e "${VERDE}Burst temporal completado sin modificar reports/load_testing/.${RESET}"
fi

echo -e "\n${VERDE}${NEGRITA}DEMO OPERATIVA COMPLETADA EXITOSAMENTE.${RESET}"
