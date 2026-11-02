"""fastEsa v0.3.2 T4 CUDA dispatcher validation and throughput benchmark.

Run: python benchmark_dispatch_v032.py --device cuda:0 --out results_v032
CPU-only: python benchmark_dispatch_v032.py --smoke
No pretrained weights or language-quality conclusions are produced.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from pathlib import Path
import torch
from fastesa import make_esa, select_swqr_backend
from fastesa import _scan


def measure(fn, repeats: int, graph_ops: int = 8) -> float:
    """Median CUDA-event milliseconds per op, using CUDA Graph replay."""
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / graph_ops)
    return statistics.median(times)


def cpu_smoke():
    a = 0.8 + 0.19 * torch.rand(1, 33, 6, 8)
    u = torch.randn_like(a) * .02
    s = torch.randn(1, 6, 8) * .02
    result = _scan.auto_scan(a, u, s)
    expected = _scan.thunder_scan(a, u, s)
    torch.testing.assert_close(result, expected)
    assert select_swqr_backend(1, 256, 'cuda', device_name='Tesla T4',
                               capability=(7, 5)) == ('swqr_short', 16)
    assert select_swqr_backend(8, 128, 'cuda', device_name='Tesla T4',
                               capability=(7, 5)) == ('swqr', None)
    print('CPU dispatch and recurrence smoke test PASSED')


def grad_check(batch: int, seq: int, device: torch.device, dim: int):
    """Check chosen native backend and gradients against original SWQR."""
    torch.manual_seed(201 + batch + seq)
    base_a = (0.82 + .17 * torch.rand(batch, seq, 6, dim // 6, device=device)).contiguous()
    base_u = (0.02 * torch.randn_like(base_a)).contiguous()
    base_s = (0.02 * torch.randn(batch, 6, dim // 6, device=device)).contiguous()
    upstream = torch.randn_like(base_a) / seq**0.5

    def run(mode):
        a = base_a.detach().clone().requires_grad_(True)
        u = base_u.detach().clone().requires_grad_(True)
        s = base_s.detach().clone().requires_grad_(True)
        y = (_scan.auto_scan(a, u, s) if mode == 'auto' else
             _scan.swqr_scan(a, u, s))
        g = torch.autograd.grad(y, (a, u, s), upstream)
        return y.detach(), [x.detach() for x in g]

    y0, g0 = run('swqr')
    y1, g1 = run('auto')
    output_err = (y0 - y1).abs().max().item()
    grad_err = max((x - y).abs().max().item() for x, y in zip(g0, g1))
    torch.testing.assert_close(y1, y0, atol=4e-5, rtol=4e-4)
    for one, two in zip(g0, g1):
        torch.testing.assert_close(one, two, atol=4e-5, rtol=4e-4)
    return output_err, grad_err


def run(args):
    cpu_smoke()
    if args.smoke:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required. Run with --smoke for CPU-only tests.')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print('GPU:', props.name, 'compute capability:', props.major, props.minor,
          '| torch:', torch.__version__, flush=True)
    _scan.prepare_swqr()
    _scan.prepare_swqr_short()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in args.batches:
            for seq in args.contexts:
                choice, warps = select_swqr_backend(batch, seq, device)
                torch.manual_seed(100 + batch + seq)
                a = (0.8 + 0.19 * torch.rand(batch, seq, 6, args.dim // 6,
                                              device=device)).contiguous()
                u = (torch.randn_like(a) * 0.02).contiguous()
                s = (torch.randn(batch, 6, args.dim // 6, device=device) * .01).contiguous()
                ref = _scan.swqr_scan(a, u, s)
                actual = _scan.auto_scan(a, u, s)
                # A successful auto result must truly use its selected native path.
                if choice == 'swqr_short' and (str(device), 'swqr_short') in _scan._FAILED_EXTENSIONS:
                    raise RuntimeError('Auto dispatcher silently lost the SWQR-Short backend')
                if (str(device), 'swqr') in _scan._FAILED_EXTENSIONS:
                    raise RuntimeError('Auto dispatcher unexpectedly fell back from native SWQR')
                max_abs = (ref - actual).abs().max().item()
                torch.testing.assert_close(actual, ref, atol=4e-5, rtol=4e-4)
                methods = {'original_swqr': lambda: _scan.swqr_scan(a, u, s),
                           'auto_dispatch': lambda: _scan.auto_scan(a, u, s)}
                if seq <= 2048:
                    methods.update({f'short_w{w}': (lambda w=w: _scan.swqr_short_scan(a, u, s, warps=w))
                                    for w in (4, 8, 16)})
                durations = {name: measure(fn, args.repeats, args.graph_ops)
                             for name, fn in methods.items()}
                result = dict(gpu=props.name, batch=batch, context=seq,
                              selected_backend=choice, selected_warps=warps,
                              max_output_error=max_abs,
                              **{name+'_ms': round(ms, 6) for name, ms in durations.items()},
                              **{name+'_tok_s': round(batch*seq/(ms*1e-3))
                                 for name, ms in durations.items()})
                rows.append(result)
                print(result, flush=True)
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with open(out/'scan_dispatch_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)

    print('Running selected full-mixer prefill parity and throughput checks', flush=True)
    model_rows = []
    for batch in args.batches:
        for seq in args.contexts:
            if batch * seq > args.max_model_tokens:  # Avoid undue graph memory use.
                continue
            torch.manual_seed(51)
            m0 = make_esa('fast', dim=args.dim, heads=6, scan='swqr').eval().to(device)
            m1 = make_esa('fast', dim=args.dim, heads=6, scan='auto').eval().to(device)
            m1.load_state_dict(m0.state_dict(), strict=True)
            x = torch.randn(batch, seq, args.dim, device=device)
            with torch.inference_mode():
                r0, s0 = m0(x)
                r1, s1 = m1(x)
                torch.testing.assert_close(r1, r0, atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(s1, s0, atol=5e-5, rtol=5e-4)
                ms0 = measure(lambda: m0(x), args.repeats, max(2, args.graph_ops // 2))
                ms1 = measure(lambda: m1(x), args.repeats, max(2, args.graph_ops // 2))
            res = dict(batch=batch, context=seq,
                       original_tok_s=round(batch*seq/(ms0*1e-3)),
                       auto_tok_s=round(batch*seq/(ms1*1e-3)),
                       auto_speedup=round(ms0/ms1, 4))
            model_rows.append(res)
            print('[model prefill]', res, flush=True)
            del m0, m1, x
    with open(out/'model_prefill_results.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['batch','context','original_tok_s','auto_tok_s','auto_speedup'])
        writer.writeheader();writer.writerows(model_rows)

    gradient_rows=[]
    for batch, seq in ((1,128), (1,512), (2,128), (2,256), (8,512)):
        err, graderr=grad_check(batch,seq,device,args.dim)
        gradient_rows.append(dict(batch=batch,context=seq,max_abs_error=err,
                                  max_gradient_error=graderr))
        print('[gradient parity]',gradient_rows[-1],flush=True)
    with open(out/'gradient_parity.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(gradient_rows[0]))
        writer.writeheader();writer.writerows(gradient_rows)
    with open(out/'environment.json','w') as f:
        json.dump(dict(gpu=props.name, torch=torch.__version__, cuda=torch.version.cuda,
                       fastesa='0.3.2', contexts=args.contexts, batches=args.batches), f, indent=2)
    print('PASS: native CUDA parity, gradient checks, and timing exported to',out)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',default='fastesa_v032_results')
    p.add_argument('--dim',type=int,default=648)
    p.add_argument('--contexts',type=int,nargs='+',default=[128,256,512,1024,2048])
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,8])
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--graph-ops',type=int,default=8)
    p.add_argument('--max-model-tokens',type=int,default=4096)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert a.dim % 6 == 0
    run(a)
