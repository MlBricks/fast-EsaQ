# Integration into the main MLBricks Kit

This archive is a **refactoring of the uploaded FastESA v0.3.8 source**. It is not a modified checkout of the separate `MLBricks` / `mlbricks-kit` repository. Do not copy this archive over the main library root.

## Target internal module convention

- `mlbricks.esa.core` — current half-life `ESA` and its recurrent math
- `mlbricks.esa.base` — historical learned-bounds `BaseESA`
- `mlbricks.esa.model` — `ESAModel` / `BaseESAModel` wrappers
- `mlbricks.esa.config` — config and architecture identity
- `mlbricks.esa.scan` — automatic/parallel/short/hierarchical/sequential scanning
- `mlbricks.esa.decode` — half-life fused CUDA decode
- `mlbricks.esa.graph` — CUDA Graph generation
- `mlbricks.esa.cuda` — native source files

The updated source in this archive resides under `fastesa/` because that is the existing package being refactored. An actual `mlbricks.esa` merge requires inspecting the main library's existing ESA classes, model builders, dispatch policy, loader, trainer, and import exports first.

## API migration

| Previous | Canonical | Compatibility |
| --- | --- | --- |
| `QualityESA` | `ESA` | Alias retained |
| `FastESA` | `BaseESA` | Alias retained |
| `QualityESAModel` | `ESAModel` | Alias retained |
| `FastESAModel` | `BaseESAModel` | Alias retained |
| `CUDAGreedyGraph` | `GraphDecoder` | Alias retained |
| `_scan.py` | `scan.py` | Import bridge retained |
| `_quality_decode.py` | `decode.py` | Import bridge retained |
| `cuda_graph.py` | `graph.py` | Import bridge retained |
| `scan='swqr'` | `scan='parallel'` | Alias retained |
| `scan='swqr_short'` | `scan='short'` | Alias retained |
| `scan='thunder'` | `scan='hierarchical'` | Alias retained |
| `scan='original'` | `scan='reference'` | Alias retained |
| `quality` variant | `current` | Alias retained |
| `fast` variant | `base` | Alias retained |

## Important checkpoint safeguards

- A checkpoint with `log_half_life` stays half-life `ESA`.
- A checkpoint with `raw_alpha` stays historical `BaseESA`.
- Preserve the parameter names of both architectures so existing state dictionaries still load strictly.
- **Do not** change the architecture of existing `mlbricks.ESA` checkpoints merely by changing its constructor default. Add a serialized architecture identifier and checkpoint migration tests to the main library first.
- Preserve `backend='auto' | 'native' | 'pytorch'` in the main MLBricks public API; execution backend and architecture selection are independent.
- Validate downstream SOUP, VESA, MLBricks Studio, and model lifecycle on a real checkout before publishing a new `mlbricks-kit` package.

## Checks completed here

- Existing and new CPU unit tests: 154 passed, 1 GPU-only skipped.
- Python syntax and wheel packaging checked.
- Native CUDA compile, graph correctness, and performance: **not tested** in this environment. Validate on a CUDA GPU before production use.
