#!/usr/bin/env bash
# End-to-end demo: exercises all three agentic branches against the deployed API.
# Prereq: `make pf` running in another terminal (port-forwards svc to :8080).
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"

ask() {
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "Q: $1"
  curl -s "$BASE_URL/chat" -H 'Content-Type: application/json' \
    -d "{\"question\": $(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" \
    | python3 -m json.tool
}

echo "# ANSWER — confident, grounded, cited"
ask "I forgot my login, how do I get back in?"
ask "Can I get my money back?"

echo
echo "# CLARIFY — ambiguous / multi-intent"
ask "I want to cancel and get my money back."

echo
echo "# DECLINE — out of scope (generalizes beyond Q10; not string-matched)"
ask "What's the weather in Essen today?"
ask "Can you help me write code for my integration project?"

echo
echo "# Decision distribution:"
curl -s "$BASE_URL/metrics" | grep support_decisions_total
