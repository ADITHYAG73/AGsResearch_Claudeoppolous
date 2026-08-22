#!/bin/bash
# Read-only: prints the RunPod account balance. Key is read from .env by the shell, never echoed.
set -a; source "$(dirname "$0")/../../.env"; set +a
KEY="${RUNPOD_API_KEY:-${RUNPOD_APIKEY:-${RUNPOD_API_TOKEN}}}"
[ -z "$KEY" ] && { echo "no RUNPOD_API_KEY in .env"; exit 1; }
curl -s https://api.runpod.io/graphql \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ myself { clientBalance spendLimit currentSpendPerHr } }"}' \
| python3 -c "import json,sys; d=json.load(sys.stdin); m=d.get('data',{}).get('myself') or {}; print(f\"balance \${m.get('clientBalance','?'):.2f}   spend/hr now \${m.get('currentSpendPerHr',0):.2f}\") if m else print('error:', d.get('errors'))"
