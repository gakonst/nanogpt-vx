# Validation — 2026-10-06

Status: **partial Vx kernel port; no GPU or end-to-end training result**.
No new speed record is claimed.

## Observed local results

Machine: Apple Silicon macOS development host. Compiler: vxc 0.0.2.
LLVM: Homebrew 22.1.8. No CUDA GPU was available for execution.

`python3 -m unittest discover -s tests -v` completed with:

```
Ran 23 tests in 2.103s
OK (skipped=1)
```

- 21 evidence-validator tests passed using explicitly synthetic fixtures. They
  exercise missing logs/ranks, wrong or duplicate GPU identities, changed source
  and data, failed/absent GPU correctness gates, changed Vx binaries, NaN loss,
  incomplete training, mismatched seeds and failure to beat the matched baseline.
  The synthetic times/losses in tests are not performance results.
- Two compiler-backed tests passed. One executes 126 differential cases across
  bf16/fp32 inputs, nine lengths (including partial thread blocks) and seven rates,
  comparing all output bytes and guard regions against an independent C `fmaf`
  reference. The other checks emitted PTX and source/artifact hashes.
- The GPU test class was skipped because CUDA/PyTorch was unavailable. Its five
  methods have **not run**: upstream Triton bitwise parity, non-default stream,
  CUDA graph replay, torch.compile mutation, and invalid storage contracts.
- Running the GPU suite with `VX_REQUIRE_CUDA=1` exited 1 as intended. The training
  runner requires this gate to pass for Vx candidates.

`python3 scripts/fetch_upstream.py` verified all 60 vendored files against the
pinned SHA-256 manifest. Python compile checks and shell syntax checks passed.

## What is compiled

Vx emits the arithmetic, bf16 widening, pointer access, and bounds check. A small
adapter maps four declared extern hooks to NVVM thread-index and LLVM FMA
intrinsics and marks CUDA entry points; LLVM emits SM90/PTX8 code. CPU checks use
a launch-index shim around the same Vx-emitted arithmetic.

This is an explicit pointer/driver integration. It does not exercise Vx's native
placement/transfer verification, automatic differentiation, or tensor-core APIs.
The full model remains upstream PyTorch/Triton. The only replaced operation is
fused tail-weight averaging, so a speedup is neither demonstrated nor guaranteed.

## Outstanding checks

1. Provision the approved MPP allocation and confirm eight full H100s. Neither
   successful allocation nor GPU capacity is established by a quote.
2. Run the mandatory GPU differential suite on that node. Driver launches and
   graph/compile integration are currently unverified on hardware.
3. Complete matched baseline and Vx runs, unchanged data and timing protocol.
4. Inspect convergence and numerical drift, then compare full-run time. Repeat
   to assess noise before claiming an improvement.
5. Seek independent upstream record validation if the evidence warrants it.

The published upstream comparison point is 39.9s at loss <=3.28. An official
record requires more than passing this repository's local, self-reported checks.
