"""Weight-independent affine scans for FAST (E[t] = A[t]*E[t-1] + U[t]).

Thunder is a self-contained pure-PyTorch adaptation of the MLBricks ESA
hierarchical scan; SWQR is optional native FP32 CUDA with an experimental
reverse-scan autograd implementation. No JIT compilation on package import.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F


_SUPPORTED_CHUNKS = {16, 32, 64, 128, 256, 512}
_EXTENSIONS: dict[str, object] = {}
_FAILED_EXTENSIONS: set[str] = set()


def _validate(A: Tensor, U: Tensor, state: Tensor | None) -> None:
    if A.ndim != 4 or A.shape != U.shape:
        raise ValueError("A and U must have matching [batch, seq, heads, head_dim] shapes")
    if A.size(1) < 1:
        raise ValueError("Sequence must have at least one token")
    if A.dtype != torch.float32 or U.dtype != torch.float32:
        raise TypeError("Scan requires FP32 A and U")
    if state is not None and (state.shape != A.shape[:1] + A.shape[2:] or
                              state.device != A.device):
        raise ValueError("Cache must have shape [batch, heads, head_dim] and match device")


def direct_scan(A: Tensor, U: Tensor, state: Tensor | None = None) -> Tensor:
    """Single-step direct recurrence; does not launch a parallel scan."""
    _validate(A, U, state)
    if A.shape[1] != 1:
        raise ValueError("direct_scan only accepts one token")
    return U if state is None else A * state.float().unsqueeze(1) + U


def thunder_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *, compass: int = 16) -> Tensor:
    """Differentiable hierarchical Thunder scan, no ESA installation required.

    Input cache is incorporated into the initial write and never detached,
    preserving gradients through chunk boundaries. Equivalent in real arithmetic.
    """
    _validate(A, U, state)
    if not isinstance(compass, int) or compass < 1:
        raise ValueError("compass must be a positive integer")
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if state is not None:
        U = torch.cat((U[:, :1] + A[:, :1] * state.float().unsqueeze(1), U[:, 1:]), dim=1)
    B, T, H, Dh = A.shape
    pad = (-T) % compass
    if pad:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        U = F.pad(U, (0, 0, 0, 0, 0, pad), value=0.0)
    groups = A.shape[1] // compass
    As = A.reshape(B, groups, compass, H, Dh)
    Us = U.reshape(B, groups, compass, H, Dh)
    st = U.new_zeros(B, groups, H, Dh)
    trans = A.new_ones(B, groups, H, Dh)
    local = []
    prefix = []
    for t in range(compass):
        st = As[:, :, t] * st + Us[:, :, t]
        trans = As[:, :, t] * trans
        local.append(st)
        prefix.append(trans)
    local = torch.stack(local, dim=2)
    prefix = torch.stack(prefix, dim=2)
    cp = prefix[:, :, -1]
    cs = local[:, :, -1]
    shift = 1
    while shift < groups:
        prev_p, prev_s = cp, cs
        cp = torch.cat((prev_p[:, :shift], prev_p[:, shift:] * prev_p[:, :-shift]), dim=1)
        cs = torch.cat((prev_s[:, :shift], prev_p[:, shift:] * prev_s[:, :-shift] + prev_s[:, shift:]), dim=1)
        shift *= 2
    carry = torch.cat((cs.new_zeros(B, 1, H, Dh), cs[:, :-1]), dim=1)
    return (prefix * carry.unsqueeze(2) + local).reshape(B, -1, H, Dh)[:, :T]


def _extension():
    """Lazy native build on the current CUDA device; PyTorch cache reused."""
    if not torch.cuda.is_available():
        raise RuntimeError("SWQR native FastScan needs a CUDA-enabled PyTorch installation")
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / "_cuda"
    if not all((root / file).is_file() for file in ("swqr_scan_bindings.cpp", "swqr_fast_scan.cu")):
        raise RuntimeError("SWQR CUDA source files are missing from the installation")
    name = "mlbricks_fastesa_swqr_scan_030"
    if name not in _EXTENSIONS:
        os.environ.setdefault("MAX_JOBS", "2")
        # Standard torch extension cache avoids writing into read-only site-packages.
        _EXTENSIONS[name] = load(
            name=name,
            sources=[str(root / "swqr_scan_bindings.cpp"), str(root / "swqr_fast_scan.cu")],
            extra_cflags=["-O2"], extra_cuda_cflags=["-O3"],
            verbose=os.getenv("FASTESA_CUDA_VERBOSE", "0") == "1",
        )
    return _EXTENSIONS[name]


class _SWQR(Function):
    @staticmethod
    def forward(ctx, A: Tensor, U: Tensor, state: Tensor, chunk: int):
        E = _extension().scan(A.contiguous(), U.contiguous(), state.contiguous(), chunk)
        ctx.save_for_backward(A, E, state)
        ctx.chunk = chunk
        return E

    @staticmethod
    def backward(ctx, gradE: Tensor):
        A, E, state = ctx.saved_tensors
        # Reverse recurrence: r_t = gradE_t + A_{t+1} * r_{t+1}.
        Ar = torch.cat((torch.zeros_like(A[:, :1]), A[:, 1:].flip(1)), dim=1).contiguous()
        Ur = gradE.float().flip(1).contiguous()
        empty = A.new_empty(0)
        r = _extension().scan(Ar, Ur, empty, ctx.chunk).flip(1)
        initial = state.unsqueeze(1) if state.numel() else torch.zeros_like(E[:, :1])
        previous = torch.cat((initial, E[:, :-1]), dim=1)
        dA = r * previous
        dU = r
        dState = (A[:, 0] * r[:, 0]).contiguous() if state.numel() else None
        return dA, dU, dState, None


def swqr_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *, chunk: int = 128) -> Tensor:
    """Native FP32 SWQR scan; CUDA required for sequences longer than one token."""
    _validate(A, U, state)
    if chunk not in _SUPPORTED_CHUNKS:
        raise ValueError(f"SWQR chunk must be one of {sorted(_SUPPORTED_CHUNKS)}")
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if not A.is_cuda:
        raise RuntimeError("scan='swqr' requires CUDA; use scan='auto' for portable fallback")
    initial = state.float().contiguous() if state is not None else A.new_empty(0)
    return _SWQR.apply(A.contiguous(), U.contiguous(), initial, chunk)


def auto_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *,
              compass: int = 16, chunk: int = 128) -> Tensor:
    """SWQR on CUDA, Thunder otherwise; direct one-token decode.

    If extension compilation fails, warn once and fall back to Thunder;
    explicit scan='swqr' never hides a failed native build.
    """
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if A.is_cuda and os.getenv("FASTESA_DISABLE_SWQR", "0") != "1":
        if str(A.device) not in _FAILED_EXTENSIONS:
            try:
                return swqr_scan(A, U, state, chunk=chunk)
            except (RuntimeError, OSError, ImportError) as exc:
                _FAILED_EXTENSIONS.add(str(A.device))
                warnings.warn(f"SWQR native unavailable ({type(exc).__name__}: {exc}); "
                              "using PyTorch Thunder. Explicit scan='swqr' raises instead.",
                              RuntimeWarning, stacklevel=2)
    return thunder_scan(A, U, state, compass=compass)


def prepare_swqr() -> None:
    """Precompile native SWQR (optional, e.g. before timing CUDA benchmarks)."""
    _extension()
