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

LS_BIN="$ROOT/.venv-labelstudio/bin/label-studio"
if [[ ! -x "$LS_BIN" ]]; then
  echo "Label Studio не найден: $LS_BIN" >&2
  echo "Создайте окружение: python3 -m venv .venv-labelstudio && source .venv-labelstudio/bin/activate && pip install label-studio" >&2
  exit 1
fi

export LS_PORT="${LS_PORT:-8080}"
export LS_DATA_DIR="${LS_DATA_DIR:-$HOME/.local/share/label-studio}"
export ML_BACKEND_URL="${ML_BACKEND_URL-http://localhost:9090}"

args=(start)

if [[ -n "${LS_PROJECT_NAME:-}" ]]; then
  args+=("$LS_PROJECT_NAME")
fi

args+=(
  --port "$LS_PORT"
  --data-dir "$LS_DATA_DIR"
  --no-browser
)

if [[ -n "${LABEL_CONFIG_PATH:-}" ]]; then
  args+=(--label-config "$LABEL_CONFIG_PATH")
fi

if [[ -n "${LS_PROJECT_NAME:-}" && "${LS_INIT_PROJECT:-1}" == "1" ]]; then
  args+=(--init)
fi

if [[ -n "$ML_BACKEND_URL" ]]; then
  # --ml-backends uses nargs='+', so it must be the last option.
  args+=(--ml-backends "$ML_BACKEND_URL")
fi

exec "$LS_BIN" "${args[@]}"
