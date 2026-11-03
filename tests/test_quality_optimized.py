"""Quality ESA v0.3.8: equations, streaming, gradients, seeded checkpoint keys."""
import pytest
import torch

from fastesa import QualityESA, CUDAGreedyGraph, make_esa, make_model, forward_tokens, load_checkpoint
from fastesa._original import progress_hybrid


@pytest.mark.parametrize('T', [1, 2, 31, 32, 33, 65, 128, 257, 512])
@pytest.mark.parametrize('cached', [False, True])
def test_quality_auto_cpu_reference_forward_backward(T, cached):
    torch.manual_seed(1000 + T + 13*cached)
    ref = make_esa('quality', dim=48, heads=6, scan='original')
    fast = make_esa('quality', dim=48, heads=6, scan='auto')
    fast.load_state_dict(ref.state_dict(), strict=True)
    x0 = torch.randn(2, T, 48, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_()
    s0 = torch.randn(2, 6, 8, requires_grad=True) if cached else None
    s1 = s0.detach().clone().requires_grad_() if cached else None
    a, aa = ref(x0, s0)
    b, bb = fast(x1, s1)
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(aa, bb, rtol=2e-4, atol=2e-5)
    (a.square().mean() + 0.1*aa.square().mean()).backward()
    (b.square().mean() + 0.1*bb.square().mean()).backward()
    torch.testing.assert_close(x0.grad, x1.grad, rtol=1e-3, atol=2e-5)
    if cached:
        torch.testing.assert_close(s0.grad, s1.grad, rtol=1e-3, atol=2e-5)
    for name, p in ref.named_parameters():
        torch.testing.assert_close(p.grad, dict(fast.named_parameters())[name].grad,
                                   rtol=1e-3, atol=2e-5)


@pytest.mark.parametrize('scan', ['auto', 'thunder', 'original'])
def test_seeded_full_model_compatibility(scan, tmp_path):
    m = make_model('quality', dim=48, layers=2, heads=6, seed=2026, scan=scan)
    baseline = make_model('quality', dim=48, layers=2, heads=6, seed=2026, scan='original')
    assert list(m.state_dict()) == list(baseline.state_dict())
    for k in m.state_dict():
        torch.testing.assert_close(m.state_dict()[k], baseline.state_dict()[k], atol=0, rtol=0)
    tokens = torch.randint(0, 256, (2, 129))
    a, states = forward_tokens(m, tokens)
    b, baseline_states = forward_tokens(baseline, tokens)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    for x, y in zip(states, baseline_states):
        torch.testing.assert_close(x, y, atol=2e-4, rtol=2e-4)
    ck = tmp_path / 'quality.pt'
    torch.save({'variant':'hybrid', 'model': baseline.state_dict()},ck)
    restored = load_checkpoint(ck, variant='quality', scan=scan)
    torch.testing.assert_close(forward_tokens(restored,tokens)[0],a,atol=2e-4,rtol=2e-4)


def test_quality_invalid_backends_and_gpu_fallback():
    with pytest.raises(ValueError):
        make_esa('quality', dim=48, scan='garbage')
    with pytest.raises(ValueError):
        make_esa('quality', dim=48, fused_decode='wrong')
    with pytest.raises(RuntimeError, match='CUDA'):
        make_esa('quality', dim=48, scan='swqr')(torch.randn(1, 4, 48))
    with pytest.raises(RuntimeError, match='CUDA'):
        CUDAGreedyGraph(None, torch.ones(1, 4, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA+nvcc; Kaggle T4')
def test_quality_cuda_native_parity():
    # Native fused decode, SWQR forward and backward on the same CUDA GPU.
    torch.manual_seed(91)
    dev = torch.device('cuda:0')
    quality = make_esa('quality', dim=48, scan='swqr', fused_decode='on').to(dev)
    reference = progress_hybrid.HybridESA(48,6).to(dev)
    reference.load_state_dict(quality.state_dict(), strict=True)
    x = torch.randn(2, 129, 48, device=dev)
    x0=x.detach().clone().requires_grad_()
    x1=x.detach().clone().requires_grad_()
    y0, s0=reference(x0)
    y1, s1=quality(x1)
    torch.testing.assert_close(y0,y1,rtol=1e-3,atol=2e-4)
    (y0.square().mean()+s0.square().mean()).backward()
    (y1.square().mean()+s1.square().mean()).backward()
    torch.testing.assert_close(x0.grad,x1.grad,rtol=4e-3,atol=2e-4)
    quality.eval(); reference.eval()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        tok=torch.randn(2,1,48,device=dev)
        a, sa = reference(tok,s0.detach())
        b, sb = quality(tok,s1.detach())
        torch.testing.assert_close(a.float(),b.float(),rtol=5e-3,atol=1e-3)
        torch.testing.assert_close(sa,sb,rtol=5e-3,atol=1e-3)
