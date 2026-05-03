#!/usr/bin/env bash

set -euo pipefail

PORT="${1:-8208}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
UNPIN_IDS="${UNPIN_IDS:-}"

echo "BASE_URL=${BASE_URL}"
echo

echo "== POST /v1/semantic/dummy =="
curl -sS -X POST "${BASE_URL}/v1/semantic/dummy" \
  -H "Content-Type: application/json"
echo
echo

echo "== GET /v1/semantic/healthz =="
curl -sS "${BASE_URL}/v1/semantic/healthz"
echo
echo

echo "== GET /v1/semantic/pinned =="
curl -sS "${BASE_URL}/v1/semantic/pinned"
echo
echo

echo "== GET /v1/semantic/scheduler_state =="
curl -sS "${BASE_URL}/v1/semantic/scheduler_state"
echo
echo

echo "== POST /v1/semantic/pinned/unpin =="
if [[ -n "${UNPIN_IDS}" ]]; then
  payload="["
  first=1
  IFS=',' read -r -a request_ids <<< "${UNPIN_IDS}"
  for request_id in "${request_ids[@]}"; do
    if [[ ${first} -eq 0 ]]; then
      payload+=", "
    fi
    payload+="\"${request_id}\""
    first=0
  done
  payload="{\"request_ids\": ${payload}]}"
else
  payload='{}'
fi

curl -sS -X POST "${BASE_URL}/v1/semantic/pinned/unpin" \
  -H "Content-Type: application/json" \
  -d "${payload}"
echo
