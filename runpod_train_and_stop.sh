#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  ./runpod_train_and_stop.sh <pod_id> [-- <train command...>]

Examples:
  ./runpod_train_and_stop.sh "$RUNPOD_POD_ID"
  ./runpod_train_and_stop.sh "$RUNPOD_POD_ID" -- python train.py --config configs/train.preflight.momh.stability.yaml

Behavior:
  1) Activates .venv
  2) Runs the training command (default: python train.py)
  3) Stops the pod with runpodctl, even if training fails
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: pod_id is required." >&2
  usage
  exit 2
fi

POD_ID="$1"
shift

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  TRAIN_CMD=("$@")
else
  TRAIN_CMD=(python train.py)
fi

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "ERROR: runpodctl not found in PATH." >&2
  exit 127
fi

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: .venv/bin/activate not found. Create/activate uv-managed venv first." >&2
  exit 2
fi

source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/huggingface}"

TRAIN_EXIT_CODE=0
"${TRAIN_CMD[@]}" || TRAIN_EXIT_CODE=$?

echo "Stopping RunPod pod: ${POD_ID}"
STOP_EXIT_CODE=0
runpodctl stop pod "${POD_ID}" || STOP_EXIT_CODE=$?

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
  echo "Training command failed with exit code ${TRAIN_EXIT_CODE}." >&2
fi
if [[ ${STOP_EXIT_CODE} -ne 0 ]]; then
  echo "Failed to stop pod ${POD_ID} (exit code ${STOP_EXIT_CODE})." >&2
fi

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
  exit "${TRAIN_EXIT_CODE}"
fi
if [[ ${STOP_EXIT_CODE} -ne 0 ]]; then
  exit "${STOP_EXIT_CODE}"
fi

exit 0
