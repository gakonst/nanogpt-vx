# MPP infrastructure

Training infrastructure is a Modal GPU sandbox obtained through Machine Payments
Protocol via Mercator. Local macOS checks are development checks, not record evidence.

`infra/mpp-plan.json` requests eight H100s for at most one hour. A live quote on
2026-10-06 returned **USD 31.5936** for creation. Exec and status endpoints each
cost USD 0.0001/call. This is a historical quote, not successful provisioning.
Refresh the quote before submitting; use its exact total and unchanged plan with
one persisted idempotency key. Reconcile unknown outcomes with that key.

This connection reports `mcp_challenge` and requires an approved quote or an
explicit sufficient budget before payment. No sandbox has been purchased. GPU
correctness, training convergence and performance have not been measured.

The quoted image is `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`.
Installing dependencies, downloading data,
warmup, baseline and candidate all consume allocation time. One hour is an initial
experiment budget, not a guarantee everything fits. A new image or duration needs
a fresh quote. Do not automatically renew.

After payment authorization:

1. Search/describe Modal, quote this complete plan, and submit with the approved
   total and stable key. Read the actual returned sandbox ID.
2. `/sandbox/exec` takes `{sandbox_id, command: ["bash", "-lc", "..."], timeout}`.
   Clone this public repo and follow `scripts/setup_gpu.sh`.
3. Inspect `nvidia-smi` before training. Exactly eight full H100 GPUs are required.
   Modal documents possible H100-to-H200 upgrades; reject those for this comparison.
4. Run compiled-kernel differential checks, then baseline and Vx on the same node.
   Preserve full logs, data hashes, environment and per-rank execution evidence.
5. Collect results before the deadline and terminate the sandbox when done.

Never publish microbenchmark or incomplete training results as a new record.
Official upstream validation/acceptance is independent of this repository.

Sources: [Modal GPU guide](https://modal.com/docs/guide/gpu),
[Sandbox resources](https://modal.com/docs/guide/sandbox-resources),
[MPP schema](https://mpp.tempo.xyz/openapi.json).
