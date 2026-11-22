# ESA direct half-life defaults (November 2026)

- Current / quality ESA now initializes six independently trainable head
  parameters to explicit nominal half-lives `[0.125, 0.5, 2, 8, 32, 64]`.
- The exact tuple is centralized in `fastesa.core.DEFAULT_CURRENT_HALF_LIVES`.
- Applied in `ESA`, `ESAModel`, `make_model(seed=...)`, and public `CurrentESA` /
  `ESAConfig` routes. Historical `_original/` research code remains unchanged.
- FP32 `log_half_life` parameters, names, shapes, model math, fused CUDA decode,
  and all scan backends remain unchanged.
- Existing checkpoints override fresh initializer values on load; no retroactive
  changes to learned weights, and no retraining automatically occurs.
- Other ESA variants (BaseESA / learned-bounds) are unaffected.
- The new initialization comes from the TinyStories 500M-token 4-layer
  benchmarks; other model widths, tasks, or contexts need separate verification.
