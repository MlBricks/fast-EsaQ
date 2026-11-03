# fastEsa v0.3.8 — Half-Life Quality ESA CUDA Patch

- Keeps the original PROGRESS half-life Quality/Hybrid math and state_dict keys.
- Quality ESA accepts all scan backends used by Fast ESA (`auto`, `swqr`, `swqr_short`, `swqr_adaptive`, `thunder`, `original`).
- Quality `auto` uses native SWQR CUDA for training/prefill, Thunder fallback for CPU, and direct recurrence at T=1.
- Adds the previously tested **legacy** fused half-life CUDA decode (not the slower experimental exp2 gate) with learned RMSNorm.
- Adds `prepare_quality_decode()` and inference-only `CUDAGreedyGraph` with a dedicated stream for each GPU.
- Guards native SWQR CUDA launches and backward passes against using the wrong active GPU in dual-T4 runs.
- Adds original/optimized quality forward, backward, seed, checkpoint and streaming parity tests.
- Upgrades package version to 0.3.8; original PROGRESS reference sources retained unchanged.

Local validation: 141 passed, 1 CUDA-only test skipped on CPU-only PyTorch.
Actual v0.3.8 native CUDA compile, numerical parity and graph throughput require a Kaggle GPU run.
