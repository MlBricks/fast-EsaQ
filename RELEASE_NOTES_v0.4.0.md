# MLBricks ESA v0.4.0 — Unified API for the FastESA repository

- Current default: half-life ESA (`ESA(embd=..., head=...)`).
- Historical learned-bounds: `ESA(..., variant="base")`.
- Execution: `backend="auto"|"native"|"pytorch"`, with `set_backend`.
- Checkpoint helpers: `save_esa_checkpoint` / `load_esa_checkpoint`, architecture verified against state_dict keys.
- Internals: `core.py`, `base.py`, `scan.py`, `decode.py`, `graph.py`, `cuda/`.
- Deprecated names and source import modules remain as compatibility shims.
- No changes to recurrence equations or checkpoint tensor keys.
- CUDA kernels need hardware-level validation on target GPUs before release.
