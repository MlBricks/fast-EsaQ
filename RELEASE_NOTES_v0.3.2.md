# fastEsa v0.3.2 — T4 batch-aware SWQR dispatch (candidate)

This release **changes execution selection only**, not FAST retention equations,
parameter names, learned checkpoints, CUDA kernel implementations, or the quality
variant. This change builds on the **experimental v0.3.1 SWQR-Short** branch;
the public GitHub `main` was last confirmed at v0.3.0 (`93b87f2`).

## Default `scan="auto"` policy

| Hardware | Batch | Sequence length | Selected backend |
|---|---:|---:|---|
| Tesla/NVIDIA T4 (SM75) | 1 | 128–512 | SWQR-Short, 16 warps |
| Tesla/NVIDIA T4 (SM75) | 2 | 128–256 | SWQR-Short, 8 warps |
| Tesla/NVIDIA T4 (SM75) | any other | 2+ | Original SWQR CUDA |
| Unprofiled CUDA GPU | any | 2+ | Original SWQR CUDA |
| CPU | any | 2+ | PyTorch Thunder |
| Any device | any | 1 | Direct one-token recurrence |

Only the specifically benchmarked T4 short-context regions use SWQR-Short.
The default does not extrapolate short-kernel gains to untested GPUs, batches,
or sequence lengths. `FASTESA_DISABLE_SWQR=1` forces Thunder fallback when
`scan="auto"` is used.

If compiling SWQR-Short fails, `auto` warns and tries original SWQR; if original
SWQR fails too, it warns and falls back to Thunder. This is an availability
fallback, not evidence that the CUDA operation was correct: strict native
modes (`swqr`, `swqr_short`, `swqr_adaptive`) do not silently fall back.

## API

```python
from fastesa import make_esa, select_swqr_backend

mixer = make_esa("fast", dim=648, heads=6, scan="auto")
print(select_swqr_backend(1, 256, "cuda:0"))  # ('swqr_short', 16) on Tesla T4

# Explicit overrides: always use the requested algorithm, or raise.
short = make_esa("fast", dim=648, heads=6, scan="swqr_short", swqr_warps=4)
long = make_esa("fast", dim=648, heads=6, scan="swqr")
portable = make_esa("fast", dim=648, heads=6, scan="thunder")

# CUDA-only adaptive mode: strict dispatch with no automatic PyTorch fallback.
adaptive = make_esa("fast", dim=648, heads=6, scan="swqr_adaptive")
```

The legacy `swqr_adaptive_scan(A,U,state,cutoff=1024,warps=8)` direct function
call remains available; providing an explicit `cutoff` requests the old
v0.3.1 length-only policy. Omitting it now selects the v0.3.2 batch-aware
policy. The user-facing `scan="swqr_adaptive"` uses the batch-aware policy.

## Validation

- 119 local CPU tests passed (including dispatcher cases, backward emulation,
  checkpoint key identity, cached-state handling, and historical tests).
- `benchmark_dispatch_v032.py --smoke` passed on CPU.
- **Actual v0.3.2 CUDA compilation and speed still require a T4 run.** Previous
  v0.3.1 benchmarks motivated the policy, but did not verify this dispatcher.

To validate on Kaggle T4, run:

```bash
python benchmark_dispatch_v032.py --device cuda:0 --out results_v032
```

The script tests output/gradient parity before claiming throughput and writes
CSV files under `results_v032`.

## Notes

- Native SWQR-Short CUDA code is unchanged from v0.3.1.
- Original SWQR CUDA code is unchanged.
- No model weights are distributed.
- Refer to the existing ESA/fastEsa noncommercial license notice.
