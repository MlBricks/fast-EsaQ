# fastEsa v0.3 candidate patch

## Provenance

The uploaded `fast-EsaQ-main(1).zip` contains metadata files, but no `fastesa/`
Python package. This release reconstructs the package from the earlier
`fastEsa_0.2.0_Thunder_source.zip`, preserves its original PROGRESS worker
scripts, and incorporates the tested SWQR scan from
`MLBricks_FAST_SWQR_Benchmark_Source.zip`. Nothing is removed from the source
license or licensing notices.

## Backend decisions

* FAST is the unchanged PROGRESS learned-retention architecture, including its
  QGV weights, raw_alpha and raw_beta state-dict names.
* `scan='auto'` (new default for FAST): native CUDA SWQR for multi-token
  training/prefill; portable PyTorch Thunder if CUDA or nvcc is unavailable;
  direct recurrence whenever sequence length is 1.
* `scan='swqr'`: explicit native CUDA SWQR; fails if unavailable, rather than
  silently measuring an alternative backend.
* `scan='thunder'`: stand-alone differentiable PyTorch Thunder scan.
* `scan='original'`: original PROGRESS learned reference, byte-for-byte source
  preserved in `fastesa/_original`.
* QUALITY (PROGRESS hybrid) is not modified by this release. Its original
  scan is retained.
* No Lightning kernel is included: earlier only-the-recurrence native
  Lightning path did not outperform FAST direct decode in the 30M benchmark.

## Safety and maturity

SWQR's native forward CUDA kernel was benchmarked on Kaggle T4. This patch
includes an experimental custom reverse-scan autograd wrapper; long-duration
training, cross-GPU CUDA compilation, and full-sequence language-model-quality
validation are still required. The native module is lazily compiled on first
CUDA SWQR use; PyTorch and CUDA SDK must match for source compilation.

Backend metadata is non-persistent; FAST checkpoint keys and weights are
unchanged. Loading the same original FAST state_dict works across all backends.

## Example

```python
from fastesa import make_esa, make_model, load_checkpoint
fast = make_esa('fast', dim=648, heads=6)   # auto dispatch
thunder = make_esa('fast', dim=648, heads=6, scan='thunder')
reference = make_esa('fast', dim=648, heads=6, scan='original')
model = make_model('fast', dim=648, heads=6, layers=6)
# Existing original PROGRESS checkpoint:
# model = load_checkpoint('latest.pt', scan='auto')
```

For GPU performance benchmarking, precompile with `from fastesa import
prepare_swqr; prepare_swqr()` before starting the timer. CUDA SWQR native mode
requires `torch.cuda.is_available()` and NVCC installed. Set
`FASTESA_DISABLE_SWQR=1` to force the auto backend to Thunder.

For a strict CUDA native parity check (including reverse gradients) on Kaggle,
run `python examples/validate_swqr_cuda.py` after installation. The test
requires CUDA and may compile the native kernel on first use.
