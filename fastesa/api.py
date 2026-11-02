"""Public fastEsa variants backed by the original PROGRESS implementations.

Fast aliases Learned ESA (without learned_lm extension).
Quality aliases Bounded-write Hybrid ESA (not the original half-life baseline).
No mathematical kernels or weight-key layouts are rewritten here.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import re
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ._original import progress_learned as learned
from ._original import progress_hybrid as hybrid

Variant = Literal["fast", "quality"]


def normalize_variant(variant: str) -> Variant:
    key = str(variant).strip().lower().replace("-", "_")
    if key in {"fast", "learned", "progress_learned", "fastesa"}:
        return "fast"
    if key in {"quality", "hybrid", "progress_hybrid"}:
        return "quality"
    raise ValueError(f"Unknown variant {variant!r}; select 'fast' or 'quality'.")


class FastESA(learned.ESAFullMixer):
    """Learned-retention FAST with weight-compatible execution backends.

    scan='auto' (default): T4 shape-aware SWQR CUDA (Short at measured small
    shapes, original SWQR elsewhere), Thunder CPU/compile fallback, direct for
    cached single-token decoding.
    scan='swqr': native SWQR (CUDA); scan='thunder': PyTorch hierarchical scan;
    scan='original': unchanged PROGRESS reference.
    """

    def __init__(self, dim: int, heads: int = 6, scan: str = "auto",
                 thunder_compass: int = 16, swqr_chunk: int = 128, swqr_warps: int = 8):
        super().__init__(dim=dim, heads=heads, variant="learned")
        if scan not in {"auto", "swqr", "swqr_short", "swqr_adaptive", "thunder", "original"}:
            raise ValueError("scan must be auto, swqr, swqr_short, swqr_adaptive, thunder or original")
        if not isinstance(thunder_compass, int) or thunder_compass < 1:
            raise ValueError("thunder_compass must be a positive integer")
        if swqr_chunk not in {16, 32, 64, 128, 256, 512}:
            raise ValueError("swqr_chunk must be a power of two in [16,512]")
        self.scan = scan
        self.thunder_compass = thunder_compass
        if swqr_warps not in (4, 8, 16):
            raise ValueError("swqr_warps must be 4, 8, or 16")
        self.swqr_chunk = swqr_chunk
        self.swqr_warps = swqr_warps

    def forward(self, x, state=None):
        if self.scan == "original":
            return super().forward(x, state)
        from ._scan import (auto_scan, direct_scan, swqr_scan, thunder_scan,
                            swqr_short_scan, swqr_adaptive_scan)
        if x.ndim != 3:
            raise ValueError("Expected input [batch, sequence, dim]")
        B, T, D = x.shape
        if T < 1:
            raise ValueError("Input must contain at least one token")
        if state is not None and (tuple(state.shape) != (B, self.heads, self.dh)
                                  or state.device != x.device):
            raise ValueError(f"Initial state must be {(B, self.heads, self.dh)} on {x.device}")
        q, g, vraw = self.qgv(x).chunk(3, dim=-1)
        gate = g.float().reshape(B, T, self.heads, self.dh)
        v = torch.tanh(vraw.float().reshape(B, T, self.heads, self.dh))
        alpha, beta = self.bounds()
        A = alpha[None, None, :, None] + (beta-alpha)[None, None, :, None]*torch.sigmoid(gate)
        U = (1.0-A)*v
        # No sequence-parallel scan for a one-token decode, even in SWQR mode.
        if T == 1:
            E = direct_scan(A, U, state)
        elif self.scan == "swqr":
            E = swqr_scan(A, U, state, chunk=self.swqr_chunk)
        elif self.scan == "swqr_short":
            E = swqr_short_scan(A, U, state, warps=self.swqr_warps)
        elif self.scan == "swqr_adaptive":
            E = swqr_adaptive_scan(A, U, state, warps=self.swqr_warps,
                                   chunk=self.swqr_chunk)
        elif self.scan == "thunder":
            E = thunder_scan(A, U, state, compass=self.thunder_compass)
        else:
            E = auto_scan(A, U, state, compass=self.thunder_compass, chunk=self.swqr_chunk)
        final = E[:, -1]
        z = F.rms_norm(E.reshape(B,T,D),(D,),eps=1e-5)
        y = (torch.sigmoid(q.float())*z).to(x.dtype)
        return self.W_O(y), final


class QualityESA(hybrid.HybridESA):
    """Unmodified PROGRESS `hybrid` recurrence (requires six heads)."""

    def __init__(self, dim: int, heads: int = 6, chunk_size: int = 32):
        super().__init__(dim=dim, heads=heads, chunk_size=chunk_size)


class FastESAModel(learned.KaggleRecall):
    """Original PROGRESS recall model with optional accelerated FAST mixers."""
    def __init__(self, vocab_size: int = 256, dim: int = 384,
                 heads: int = 6, layers: int = 6,
                 scan: str = "auto", thunder_compass: int = 16, swqr_chunk: int = 128, swqr_warps: int = 8):
        super().__init__(vocab_size=vocab_size, dim=dim, heads=heads,
                         layers=layers, variant="learned")
        if scan != "original":
            for block in self.blocks:
                block.esa = FastESA(dim, heads, scan=scan,
                                    thunder_compass=thunder_compass, swqr_chunk=swqr_chunk, swqr_warps=swqr_warps)
        self.scan = scan


class QualityESAModel(hybrid.RecallModel):
    """Original hybrid notebook's embedding, ESA blocks, FFN and tied LM head.

    `forward(tokens, cached_states=None)` returns (logits [B,V], states).
    """

    def __init__(self, vocab_size: int = 256, dim: int = 384,
                 heads: int = 6, layers: int = 6, chunk_size: int = 32):
        super().__init__(vocab_size=vocab_size, dim=dim, heads=heads,
                         layers=layers, variant="hybrid", chunk_size=chunk_size)


def make_esa(variant: str = "fast", *, dim: int = 384, heads: int = 6,
             chunk_size: int = 32, scan: str = "auto",
             thunder_compass: int = 16, swqr_chunk: int = 128, swqr_warps: int = 8) -> nn.Module:
    """Create standalone mixer, with original trained state_dict key layout."""
    if normalize_variant(variant) == "fast":
        if chunk_size != 32:
            raise ValueError("PROGRESS learned uses fixed chunk_size=32; cannot change it without changing the implementation")
        return FastESA(dim=dim, heads=heads, scan=scan,
                       thunder_compass=thunder_compass, swqr_chunk=swqr_chunk, swqr_warps=swqr_warps)
    if scan not in {"auto", "original"}:
        raise ValueError("Only FAST supports selecting scan backends")
    return QualityESA(dim=dim, heads=heads, chunk_size=chunk_size)


def make_model(variant: str = "fast", *, vocab_size: int = 256,
               dim: int = 384, heads: int = 6, layers: int = 6,
               chunk_size: int = 32, seed: int | None = None,
               scan: str = "auto", thunder_compass: int = 16, swqr_chunk: int = 128, swqr_warps: int = 8) -> nn.Module:
    """Build original recall/LM-shaped model.

    If seed is provided, reproduce the variant's EXACT notebook initializer.
    This is an architecture factory, not a pretrained weights download.
    """
    v = normalize_variant(variant)
    if v == "fast":
        if chunk_size != 32:
            raise ValueError("PROGRESS learned uses fixed chunk_size=32")
        model = FastESAModel(vocab_size=vocab_size, dim=dim, heads=heads, layers=layers,
                             scan=scan, thunder_compass=thunder_compass, swqr_chunk=swqr_chunk, swqr_warps=swqr_warps)
        if seed is not None:
            learned.deterministic_init(model, int(seed))
        return model
    if scan not in {"auto", "original"}:
        raise ValueError("Only FAST supports selecting scan backends")
    if seed is not None:
        args = SimpleNamespace(seed=int(seed), vocab_size=vocab_size, dim=dim,
                               heads=heads, layers=layers, chunk_size=chunk_size)
        return hybrid.build_model(args, "hybrid")
    return QualityESAModel(vocab_size=vocab_size, dim=dim, heads=heads,
                           layers=layers, chunk_size=chunk_size)


def forward_tokens(model: nn.Module, tokens: torch.Tensor, states=None):
    """Uniform streaming API for the *unchanged* notebook LM models.

    Returns (last-token logits [B,V], new_states [layers][B,H,D/H]).
    Requires tokens shaped [B,T] with T >= 1.
    """
    if isinstance(model, learned.KaggleRecall):
        return model.forward_chunk(tokens, states)
    if isinstance(model, hybrid.RecallModel):
        return model(tokens, states)
    raise TypeError("Expected a model created by fastesa.make_model")


def load_checkpoint(path: str | Path, *, variant: str | None = None,
                    device: str | torch.device = "cpu", chunk_size: int = 32,
                    trusted: bool = False, scan: str = "auto",
                    thunder_compass: int = 16, swqr_chunk: int = 128, swqr_warps: int = 8) -> nn.Module:
    """Restore a `latest.pt` model produced by an ORIGINAL PROGRESS notebook.

    Only model weights are restored, not optimizer/scaler/RNG state. Architecture
    is inferred from original weight tensors; variant is checked against the
    checkpoint's variant label, when present.

    By default deserialization runs in torch `weights_only` safe mode. For
    hybrid notebooks that save arbitrary NumPy RNG objects, only pass
    trusted=True if YOU trust the checkpoint producer and contents.
    """
    try:
        obj = torch.load(Path(path), map_location="cpu", weights_only=not trusted)
    except Exception as e:
        if not trusted:
            raise ValueError(
                "Could not safely read checkpoint; original notebook may contain "
                "NumPy/Python RNG objects. Only if you trust the file, call "
                "load_checkpoint(..., trusted=True)."
            ) from e
        raise
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint should contain a model state dict or {'model': state_dict}.")
    state = obj.get("model", obj)
    if not isinstance(state, dict) or "emb.weight" not in state:
        raise ValueError("Expected original PROGRESS recall model state_dict with emb.weight")
    indicated = obj.get("variant", None)
    if indicated is not None and variant is not None:
        if normalize_variant(indicated) != normalize_variant(variant):
            raise ValueError(f"Checkpoint is {indicated!r}, not {variant!r}")
    if variant is None:
        if indicated is None:
            if "blocks.0.esa.raw_alpha" in state:
                variant = "fast"
            elif "blocks.0.esa.log_half_life" in state:
                variant = "quality"
            else:
                raise ValueError("Unknown checkpoint architecture; specify a variant")
        else:
            variant = indicated
    v = normalize_variant(variant)
    vocab_size, dim = state["emb.weight"].shape
    layer_ids = [int(m.group(1)) for name in state
                 if (m := re.match(r"^blocks\.(\d+)\.esa\.", name))]
    if not layer_ids:
        raise ValueError("No ESA blocks found in checkpoint")
    layers = max(layer_ids) + 1
    if v == "fast":
        heads = int(state["blocks.0.esa.raw_alpha"].numel())
    else:
        heads = int(state["blocks.0.esa.log_half_life"].numel())
    model = make_model(v, vocab_size=int(vocab_size), dim=int(dim),
                       heads=heads, layers=layers, chunk_size=chunk_size,
                       scan=scan, thunder_compass=thunder_compass, swqr_chunk=swqr_chunk, swqr_warps=swqr_warps)
    model.load_state_dict(state, strict=True)
    return model.to(device)
