"""CUDA smoke/throughput example: python examples/benchmark_quality_optimized.py.

Requires a CUDA toolkit for on-demand fused-kernel/SWQR compilation.
Run the two models on the same GPU in separate measurements for fairness.
"""
from __future__ import annotations
import time
import torch
from fastesa import (CUDAGreedyGraph, make_model, forward_tokens,
                     prepare_quality_decode)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for this speed test')
    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    torch.manual_seed(17)
    model = make_model('quality', dim=384, heads=6, layers=6,
                       seed=17, scan='auto', fused_decode='on').to(dev).eval()
    torch.cuda.synchronize(dev)
    prepare_quality_decode()
    # Validate unfused vs fused on same exact weights and initial state.
    reference = make_model('quality', dim=384, heads=6, layers=6,
                           scan='original').to(dev).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    prompt = torch.randint(0, 256, (1, 512), device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        old, old_states = forward_tokens(reference, prompt)
        new, new_states = forward_tokens(model, prompt)
        torch.testing.assert_close(new.float(), old.float(), atol=0.02, rtol=0.02)
        for a, b in zip(old_states, new_states):
            torch.testing.assert_close(a, b, atol=0.01, rtol=0.01)
        one = torch.randint(0, 256, (1, 1), device=dev)
        old, old_states = forward_tokens(reference, one, old_states)
        new, new_states = forward_tokens(model, one, new_states)
        torch.testing.assert_close(new.float(), old.float(), atol=0.02, rtol=0.02)
        for a, b in zip(old_states, new_states):
            torch.testing.assert_close(a, b, atol=0.01, rtol=0.01)
    print('Quality ESA CUDA scan/fused decode forward parity PASS')
    # Use a smaller prefill as graph capture cost scales with prompt length.
    decoder = CUDAGreedyGraph(model, prompt[:, :16], unroll=8)
    drift = decoder.check_parity()
    n = 256
    for _ in range(3):
        decoder.generate(n)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    tokens = decoder.generate(n)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    print(f'Graph parity PASS (state max_abs={drift:.6g})')
    print(f'Quality ESA graph decode: {tokens.numel()/elapsed:,.0f} tokens/s '
          f'({tokens.size(1)} tokens, batch {tokens.size(0)})')


if __name__ == '__main__':
    main()
