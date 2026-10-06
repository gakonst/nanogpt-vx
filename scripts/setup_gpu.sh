#!/usr/bin/env bash
# Run only on an already-authorized Linux x86_64 GPU host. Does not rent a GPU.
# Base image: nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 (pinned upstream Dockerfile).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || { echo 'Linux x86_64 required' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'Use Ubuntu 24.04 CUDA 13 base image' >&2; exit 1; }
SUDO=()
if [[ "$EUID" != 0 ]]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y ca-certificates curl gnupg python3-venv python3-dev git build-essential z3 libffi8
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > "$ROOT/.llvm-key.gpg"
"${SUDO[@]}" install -m 644 "$ROOT/.llvm-key.gpg" /usr/share/keyrings/llvm-snapshot.gpg
printf '%s\n' 'deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-22 main' | "${SUDO[@]}" tee /etc/apt/sources.list.d/nanogpt-vx-llvm22.list
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y llvm-22 llvm-22-dev clang-22 libmlir-22-dev mlir-22-tools
export LLVM_BIN=/usr/lib/llvm-22/bin
export PATH="$LLVM_BIN:$HOME/.vx/bin:$PATH"
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$ROOT/requirements.txt"
# Require the upstream CUDA 13 attention runtime; CUDA 12 wheels alone are insufficient.
python -c 'import ctypes; ctypes.CDLL("libcudart.so.13")'
VX_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["vx"]["commit"])' "$ROOT/toolchain.lock.json")"
export VX_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["vx"]["release"])' "$ROOT/toolchain.lock.json")"
# Pinned installer downloads the fixed release and verifies the published checksum.
curl -fsSL "https://raw.githubusercontent.com/vx-lang/Vx/$VX_COMMIT/www/install.sh" -o "$ROOT/.vx-install.sh"
env -u VX_SKIP_CHECKSUM VX_NO_CUDA=1 VX_NO_MODIFY_PATH=1 sh "$ROOT/.vx-install.sh"
python "$ROOT/vx/build.py" --vxc "$HOME/.vx/bin/vxc" --llvm-bin "$LLVM_BIN" --chip sm_90
python "$ROOT/scripts/fetch_upstream.py"
python "$ROOT/scripts/preflight.py" --output "$ROOT/setup-hardware.json"
python -m pip freeze --all > "$ROOT/setup-pip-freeze.txt"
printf 'Setup complete. Activate with: source %q/.venv/bin/activate\n' "$ROOT"
printf 'Then scripts/download_data.sh; scripts/run.sh baseline; scripts/run.sh vx; scripts/compare.py BASELINE VX\n'
