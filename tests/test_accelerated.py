"""CPU validation of v0.3 scan and checkpoint compatibility.

Native CUDA parity/performance must be run separately on a real GPU.
"""
import torch
import pytest
from fastesa import make_esa, make_model, forward_tokens, load_checkpoint
from fastesa import __version__
from fastesa._scan import thunder_scan, auto_scan, direct_scan, swqr_scan


@pytest.mark.parametrize("T", [1, 16, 31, 32, 33, 64, 128, 257])
@pytest.mark.parametrize("cache", [False, True])
def test_thunder_matches_original_and_grad(T, cache):
    torch.manual_seed(T + int(cache))
    original = make_esa('fast', dim=24, heads=6, scan='original')
    thunder = make_esa('fast', dim=24, heads=6, scan='thunder')
    thunder.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, T, 24)
    state = torch.randn(2, 6, 4) if cache else None
    xa = x.detach().clone().requires_grad_()
    xb = x.detach().clone().requires_grad_()
    sa = state.detach().clone().requires_grad_() if cache else None
    sb = state.detach().clone().requires_grad_() if cache else None
    ya, ea = original(xa, sa)
    yb, eb = thunder(xb, sb)
    torch.testing.assert_close(yb, ya, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eb, ea, rtol=1e-4, atol=1e-5)
    loss_a = ya.square().mean() + .1 * ea.square().mean()
    loss_b = yb.square().mean() + .1 * eb.square().mean()
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(xa.grad, xb.grad, rtol=2e-3, atol=1e-5)
    if cache:
        torch.testing.assert_close(sa.grad, sb.grad, rtol=2e-3, atol=1e-5)
    for (name_a, p_a), (name_b, p_b) in zip(original.named_parameters(), thunder.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(p_a.grad, p_b.grad, rtol=2e-3, atol=1e-5)


def test_auto_cpu_matches_thunder_and_direct():
    a = make_esa('fast', dim=48, scan='auto')
    b = make_esa('fast', dim=48, scan='thunder')
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(2, 70, 48)
    ya, sa = a(x)
    yb, sb = b(x)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    z = torch.randn(2, 1, 48)
    yd, sd = a(z, sa)
    yt, st = b(z, sb)
    torch.testing.assert_close(yd, yt, atol=0, rtol=0)
    torch.testing.assert_close(sd, st, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="CUDA"):
        swqr_scan(torch.ones(1, 2, 1, 1), torch.ones(1, 2, 1, 1))


def test_model_checkpoint_stable(tmp_path):
    m1 = make_model('fast', dim=48, layers=2, scan='original')
    m2 = make_model('fast', dim=48, layers=2, scan='auto')
    assert set(m1.state_dict()) == set(m2.state_dict())
    m2.load_state_dict(m1.state_dict(), strict=True)
    path = tmp_path / 'weights.pt'
    torch.save({'variant': 'learned', 'model': m1.state_dict()}, path)
    m3 = load_checkpoint(path, scan='thunder')
    tokens = torch.randint(0, 256, (2, 37))
    original = forward_tokens(m1, tokens)[0]
    torch.testing.assert_close(forward_tokens(m2, tokens)[0], original, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(forward_tokens(m3, tokens)[0], original, rtol=1e-4, atol=1e-4)
    assert __version__ == "0.3.2"


def test_state_gradient_recurrence():
    torch.manual_seed(22)
    a = torch.sigmoid(torch.randn(1, 5, 2, 4)).requires_grad_()
    u = torch.randn_like(a).requires_grad_()
    state = torch.randn(1, 2, 4).requires_grad_()
    e = thunder_scan(a, u, state)
    z = state
    full = []
    for i in range(5):
        z = a[:, i] * z + u[:, i]
        full.append(z)
    torch.testing.assert_close(e, torch.stack(full, dim=1), rtol=1e-5, atol=1e-6)
    assert e.grad_fn is not None
    e.sum().backward()
    assert state.grad is not None

@pytest.mark.parametrize("length", [2, 3, 33, 128, 513])
@pytest.mark.parametrize("cache", [False, True])
def test_swqr_experimental_custom_backward_cpu_emulation(monkeypatch, length, cache):
    """Check analytic SWQR backward; GPU kernel itself must be tested on T4."""
    from fastesa import _scan
    class Emulator:
        @staticmethod
        def scan(A, U, state, chunk):
            carry = state if state.numel() else U.new_zeros(A.size(0), A.size(2), A.size(3))
            out = []
            for t in range(A.shape[1]):
                carry = A[:, t] * carry + U[:, t]
                out.append(carry)
            return torch.stack(out, dim=1)
    monkeypatch.setattr(_scan, '_extension', lambda: Emulator())
    torch.manual_seed(length + 100 * cache)
    a = torch.sigmoid(torch.randn(2, length, 3, 4)).detach().requires_grad_()
    u = torch.randn_like(a).requires_grad_()
    initial = torch.randn(2, 3, 4).requires_grad_() if cache else a.new_empty(0)
    coeff = torch.randn_like(a)
    actual = _scan._SWQR.apply(a, u, initial, 128)
    (actual * coeff).sum().backward()
    a2 = a.detach().requires_grad_()
    u2 = u.detach().requires_grad_()
    initial2 = initial.detach().requires_grad_()
    ref = Emulator.scan(a2, u2, initial2, 128)
    (ref * coeff).sum().backward()
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, a2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(u.grad, u2.grad, rtol=1e-4, atol=1e-5)
    if cache:
        torch.testing.assert_close(initial.grad, initial2.grad, rtol=1e-4, atol=1e-5)
