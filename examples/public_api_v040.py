"""Unified ESA public API smoke test; no GPU required."""
import torch
from fastesa import ESA, save_esa_checkpoint, load_esa_checkpoint

for variant in ("current", "base"):
    model = ESA(embd=48, head=6, variant=variant, backend="pytorch")
    x = torch.randn(2, 16, 48)
    y, state = model(x)
    assert y.shape == x.shape
    assert state.shape == (2, 6, 8)
    print(variant, y.shape, model.resolved_backend())
