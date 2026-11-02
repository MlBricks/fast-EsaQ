"""SWQR-Short CPU emulation verifies API, math, gradients without claiming CUDA parity."""
import pytest
import torch
from fastesa import make_esa
from fastesa import _scan


class EmulatedShort:
    @staticmethod
    def scan(A, U, state, warps):
        # Exact mathematical CUDA contract; does not simulate GPU performance.
        carry = state if state.numel() else torch.zeros_like(A[:, 0])
        seq = []
        for t in range(A.shape[1]):
            carry = A[:, t] * carry + U[:, t]
            seq.append(carry)
        return torch.stack(seq, 1)


@pytest.mark.parametrize("T", [2, 7, 33, 128, 129, 512, 1024, 1536, 2048])
@pytest.mark.parametrize("warps", [4, 8, 16])
@pytest.mark.parametrize("cached", [False, True])
def test_short_forward_backward_cpu_emulator(monkeypatch, T, warps, cached):
    monkeypatch.setattr(_scan, "_short_extension", lambda: EmulatedShort())
    torch.manual_seed(T + warps + int(cached) * 123)
    a = (.85 + .14 * torch.rand(1, T, 2, 3)).requires_grad_()
    u = (.08 * torch.randn_like(a)).requires_grad_()
    s = (.03 * torch.randn(1, 2, 3)).requires_grad_() if cached else a.new_empty(0)
    weight = torch.randn_like(a) / T**.5
    y = _scan._SWQRShort.apply(a, u, s, warps)
    g = torch.autograd.grad(y, (a, u, s) if cached else (a, u), weight)
    ar = a.detach().requires_grad_()
    ur = u.detach().requires_grad_()
    sr = s.detach().requires_grad_() if cached else s
    yr = EmulatedShort.scan(ar, ur, sr, warps)
    gr = torch.autograd.grad(yr, (ar, ur, sr) if cached else (ar, ur), weight)
    torch.testing.assert_close(y, yr, rtol=0, atol=0)
    for have, want in zip(g, gr):
        torch.testing.assert_close(have, want, rtol=1e-4, atol=1e-5)


def test_backend_selection_and_compatible_weights():
    m = make_esa('fast', dim=48, heads=6, scan='swqr_short', swqr_warps=8)
    n = make_esa('fast', dim=48, heads=6, scan='swqr')
    n.load_state_dict(m.state_dict(), strict=True)
    assert set(m.state_dict()) == set(n.state_dict())
    with pytest.raises(ValueError):
        make_esa('fast', dim=48, scan='swqr_short', swqr_warps=3)
    with pytest.raises(ValueError, match='2048'):
        _scan.swqr_short_scan(torch.ones(1, 2049, 1, 1), torch.ones(1, 2049, 1, 1))
    with pytest.raises(RuntimeError, match='CUDA'):
        _scan.swqr_short_scan(torch.ones(1, 128, 1, 1), torch.ones(1, 128, 1, 1))
