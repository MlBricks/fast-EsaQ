"""Run on a CUDA GPU (e.g., Kaggle T4) to validate v0.3 native SWQR.

Usage: python examples/validate_swqr_cuda.py
This validates native gradients and cached-state parity, not trained LM quality.
"""
import time
import torch
from fastesa import make_esa, prepare_swqr


def check_close(name, actual, expected, *, atol, rtol):
    diff = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: name + ': ' + m)
    print(name, 'PASS max_abs=', diff, flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This validation requires a real CUDA device')
    print('GPU:', torch.cuda.get_device_name(), flush=True)
    prepare_swqr()
    for T in [2, 31, 128, 513, 2048, 4096]:
        for cached in (False, True):
            torch.manual_seed(500 + T + int(cached))
            old = make_esa('fast', dim=96, heads=6, scan='original').cuda().float()
            new = make_esa('fast', dim=96, heads=6, scan='swqr').cuda().float()
            new.load_state_dict(old.state_dict(), strict=True)
            x0 = torch.randn(1, T, 96, device='cuda', requires_grad=True)
            x1 = x0.detach().clone().requires_grad_()
            s = torch.randn(1, 6, 16, device='cuda') if cached else None
            s0 = s.detach().clone().requires_grad_() if cached else None
            s1 = s.detach().clone().requires_grad_() if cached else None
            y0, e0 = old(x0, s0)
            y1, e1 = new(x1, s1)
            check_close('outputs T=%d cached=%s' % (T, cached), y1, y0, atol=3e-4, rtol=3e-4)
            check_close('state T=%d cached=%s' % (T, cached), e1, e0, atol=3e-4, rtol=3e-4)
            (y0.square().mean() + e0.square().mean()*.1).backward()
            (y1.square().mean() + e1.square().mean()*.1).backward()
            check_close('input grads', x1.grad, x0.grad, atol=1e-4, rtol=1e-3)
            if cached:
                check_close('state grads', s1.grad, s0.grad, atol=1e-4, rtol=1e-3)
            for pa, pb in zip(old.parameters(), new.parameters()):
                check_close('param grads', pb.grad, pa.grad, atol=1e-4, rtol=1e-3)
    print('ALL CUDA PARITY TESTS PASSED. Benchmark speed separately.', flush=True)


if __name__ == '__main__':
    main()
