#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/labelstudio_backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/labelstudio_backend/.env"
  set +a
fi

source .venv/bin/activate

export LABEL_STUDIO_URL="${LABEL_STUDIO_URL:-http://localhost:8080}"
export LABEL_STUDIO_API_KEY="${LABEL_STUDIO_API_KEY:-}"

exec label-studio-ml start labelstudio_backend --host 0.0.0.0 --port "${PORT:-9090}"
