# ESA Engine v0.3.9 — internal naming cleanup

- **ESA** / **ESAModel**: current half-life architecture. Factory defaults changed from learned-bounds to half-life.
- **BaseESA** / **BaseESAModel**: historical learned-bounds architecture.
- Execution modules: `core.py`, `base.py`, `scan.py`, `decode.py`, `graph.py`, `model.py`, `config.py`.
- CUDA files renamed to `scan.cu`, `short_scan.cu`, `decode.cu` with matching bindings. Mathematical computations remain unchanged.
- Clean entrypoints: `parallel_scan`, `short_scan`, `sequential_scan`, `hierarchical_scan`, `GraphDecoder`, `prepare_decode`.
- Clean scan settings: `auto`, `parallel`, `short`, `adaptive`, `hierarchical`, `reference`.
- Old class names, import paths, scan options, environment variables, and variant names remain compatible aliases.
- `load_checkpoint` detects the architecture by parameter keys and honors old checkpoint variant metadata.
- New CUDA extension build names avoid accidentally loading cached binaries for v0.3.8.

**Integration note:** This is the refactored FastESA repository, NOT a modification of the separate MLBricks Kit source repository. Before merging into MLBricks Kit, adapt its existing `mlbricks.esa` API/native resolver and test its existing checkpoints and downstream components.

**GPU note:** CUDA kernels are source-renamed only; run native parity and graph tests on the target GPU before production deployment.
