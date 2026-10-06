# nanoGPT-Vx

An experimental Vx kernel port of the current
[modded-nanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), with MPP
GPU infrastructure and a reproducible baseline/candidate runner.

**Status: partial kernel port, not a new record.** No H100 execution, full training
run, convergence result or speedup has been measured. The full model, FP8 GEMMs,
attention, optimizer and distributed runtime still use upstream PyTorch/Triton.

The first port replaces tail-weight averaging with compiled Vx kernels for bf16
and fp32 inputs and an fp32 accumulator. It preserves the upstream fused operation
`fma(rate, float(src) - dst, dst)`. Vx emits the arithmetic and loads/stores;
a small build adapter maps thread-index/FMA extern hooks to LLVM intrinsics and
marks kernel entry points before LLVM emits SM90 PTX. This currently uses raw
pointers and an explicit CUDA driver launch adapter; it does not claim Vx's native
placement/transfer verification or tensor-core lowering.

The trainer is pinned at `4ea6b937337a4889b8cfe3f38a93d120048d8f71` and vendored
with its MIT license. Vx is pinned to v0.0.2, using LLVM 22. `.vx` files use Rust
syntax highlighting on GitHub.

## Target

The pinned upstream record is **39.9 seconds**, reaching **≤3.28 FineWeb validation
loss on 8×H100**. A candidate must preserve the token streams and timing rules,
complete the full training and validation, and beat the baseline on the same
hardware. A kernel microbenchmark is not an end-to-end training record.

The local evidence checker never declares an official record. Upstream rule
review and independent validation remain necessary. Changing the learning
algorithm also requires the upstream statistical validation procedure.

## Run on an MPP GPU node

Read [MPP provisioning and cost](docs/MPP.md) first. The initial proposed allocation
is one hour on 8×H100; no allocation has been purchased.

```bash
bash scripts/setup_gpu.sh
source .venv/bin/activate
python scripts/fetch_upstream.py
bash scripts/download_data.sh
python vx/build.py --llvm-bin /usr/lib/llvm-22/bin --chip sm_90
VX_REQUIRE_CUDA=1 python -m unittest discover -s tests -p 'test_vx_gpu.py' -v
bash scripts/run.sh baseline runs/baseline
bash scripts/run.sh vx runs/vx
python scripts/compare.py runs/baseline runs/vx
```

Run the GPU differential tests before both training runs. GPU correctness and
performance are unverified until those tests run on the actual allocation.
The preflight rejects CPU, H200, fewer than eight GPUs, and shortened schedules.
Keep all run bundles, including logs, per-rank receipts, source/data hashes and
machine information. `scripts/validate_record.py` checks bundle completeness;
its hashes detect accidental edits, not fabricated measurements.

## Development

```bash
python -m unittest discover -s tests -v
```

Compiler-backed tests need `VXC` pointing to the pinned compiler and `LLVM_BIN`
pointing to LLVM 22 tools. Test results and exact remaining limitations are in
[the validation report](results/VALIDATION.md).

## Scope and next measurements

1. Execute GPU parity tests against the independent upstream Triton kernel,
   including non-default streams and graph replay.
2. Measure kernel latency and complete matched baseline/candidate training runs.
3. Profile the measured bottleneck before expanding the port to fused FP8
   quantization, optimizer or tensor-core work. Tail averaging may be too small
   a share of runtime to improve the record, and a new implementation may be slower.

The public repo is a starting point for this record attempt. No result is inferred
from compilation, local CPU checks or the availability of an MPP quote.

See [upstream rules](https://github.com/KellerJordan/modded-nanogpt#rules),
[Vx](https://vxlang.org/), [NOTICE](NOTICE) and [LICENSE](LICENSE).
