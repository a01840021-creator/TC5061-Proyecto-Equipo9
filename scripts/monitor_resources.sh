#!/usr/bin/env bash
set -euo pipefail

DURATION_SECONDS="${1:-370}"
INTERVAL_SECONDS="${2:-2}"
OUTPUT_FILE="${3:-reports/load_testing/resources.csv}"
CONTAINERS="${CONTAINERS:-mibici-api mibici-mlflow}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
printf 'timestamp,container,cpu_percent,memory_usage,memory_percent\n' > "$OUTPUT_FILE"

end_time=$((SECONDS + DURATION_SECONDS))
while (( SECONDS < end_time )); do
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  for container in $CONTAINERS; do
    if docker inspect "$container" >/dev/null 2>&1; then
      docker stats --no-stream \
        --format "${timestamp},{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" \
        "$container" >> "$OUTPUT_FILE"
    fi
  done
  sleep "$INTERVAL_SECONDS"
done

echo "Métricas de recursos guardadas en $OUTPUT_FILE"
