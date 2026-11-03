# Current ESA (half-life) and historical Base ESA

The stable ESA mixer API uses the requested `embd`, `head`, `variant`, and `backend` names:

```python
from fastesa import ESA, ESAConfig, GraphDecoder

# Half-life ESA, the new default.
current = ESA(embd=384, head=6, backend="auto")

# Historical learned-bounds Base ESA.
base = ESA(embd=384, head=6, variant="base", backend="auto")

# Explicit current variant.
explicit = ESA(embd=384, head=6, variant="current", backend="auto")

# Other execution choices.
reference = ESA(embd=384, head=6, backend="pytorch")
```

The low-level `make_esa`, `make_model`, `FastESA`, `QualityESA`, `scan`, and
`CUDAGreedyGraph` interfaces remain importable for old notebooks.
`GraphDecoder` is the canonical graph helper. Checkpoint tensor keys do not change.

In **this FastESA repository**, use `from fastesa import ESA`. Once integrated
into the separate MLBricks Kit root package, the public import can be
`from mlbricks import ESA` with exactly the same constructor shape.
