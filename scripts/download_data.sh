#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${UPSTREAM_DIR:-$ROOT/vendor/modded-nanogpt}"
python "$ROOT/scripts/fetch_upstream.py" --destination "$UPSTREAM"
cd "$UPSTREAM"
python data/cached_fineweb10B.py 9
