#!/usr/bin/env bash
set -euo pipefail

PROFILE="${MIBICI_LOAD_PROFILE:-standard}"
HOST="${MIBICI_API_URL:-http://localhost:8000}"
OUTPUT_DIR="${LOAD_OUTPUT_DIR:-reports/load_testing}"
INTERVAL_SECONDS="${RESOURCE_INTERVAL_SECONDS:-2}"

case "$PROFILE" in
  smoke) DURATION_SECONDS=25 ;;
  standard) DURATION_SECONDS=370 ;;
  *) echo "Perfil no válido: $PROFILE (use smoke o standard)" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
docker compose up -d --build --wait --wait-timeout 120

./scripts/monitor_resources.sh \
  "$DURATION_SECONDS" \
  "$INTERVAL_SECONDS" \
  "$OUTPUT_DIR/resources.csv" &
monitor_pid=$!

cleanup() {
  if kill -0 "$monitor_pid" >/dev/null 2>&1; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

MIBICI_LOAD_PROFILE="$PROFILE" python -m locust \
  -f locustfile.py \
  --headless \
  --host "$HOST" \
  --csv "$OUTPUT_DIR/locust" \
  --csv-full-history \
  --html "$OUTPUT_DIR/locust_report.html" \
  --json-file "$OUTPUT_DIR/locust_summary_raw"

wait "$monitor_pid" || true
trap - EXIT INT TERM

python -m mibici.load_testing.report \
  --prefix "$OUTPUT_DIR/locust" \
  --recursos "$OUTPUT_DIR/resources.csv" \
  --salida "$OUTPUT_DIR"

echo "Prueba terminada. Los contenedores continúan activos para inspección."
echo "Para detenerlos: docker compose down"
