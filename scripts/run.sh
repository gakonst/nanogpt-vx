#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-vx}"
OUT="${2:-$ROOT/runs/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
exec python "$ROOT/scripts/benchmark.py" --mode "$MODE" --upstream "${UPSTREAM_DIR:-$ROOT/vendor/modded-nanogpt}" --output "$OUT" --seed "${TRAIN_SEED:-42}"
