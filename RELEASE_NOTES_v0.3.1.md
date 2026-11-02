# fastEsa v0.3.1 — SWQR-Short Experimental

- Added native CUDA one-launch CTA-local segmented scan for T <= 2048. Warps process disjoint time segments while lanes access contiguous channel elements. Shared memory holds per-segment affine summaries.
- `scan='swqr_short'` explicitly requires CUDA and T<=2048. Configure 4, 8, or 16 warps. `scan='swqr_adaptive'` uses short scan for 2..1024 by default, preserving original SWQR at >1024. One-token decoding still uses direct recurrence.
- Existing `scan='swqr'` and default `scan='auto'` unchanged, so no unintended speed regression.
- Backward pass reuses the new CUDA scan in reverse time, preserving gradients for retention, write, and cached input state. No change to FAST math or state_dict keys.
- Includes dedicated GPU parity+benchmark script against v0.3 SWQR and ESA Native Thunder, including full-model prefill. The external ESA source in the standalone Kaggle notebook is provided separately under its own license.

**Validation scope:** 79 CPU-side tests passed, including analytic-gradient CPU emulation. No real CUDA compiler/GPU was available for verifying the kernel in the development container. Native CUDA correctness and performance must be determined on Kaggle before declaring a short-context speedup. The adaptive threshold 1024 is experimental; benchmark-tune before changing production defaults.
