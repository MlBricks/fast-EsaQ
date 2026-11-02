"""Real T4 SWQR-Short vs old SWQR vs ESA native Thunder benchmark.

All backends use identical FP32 A/U, optional state. Only selected GPU native
backends are compared, never silently substituted. CUDA events and CUDA graphs
reduce host launch noise. Results are not supplied in advance.
"""
from __future__ import annotations
import argparse, csv, json, statistics, time
from pathlib import Path
import torch
from fastesa import _scan, make_model, make_esa, forward_tokens


def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def event_bench(fn, repeats=10, graph_iters=32):
    # Replay a captured sequence, normalize GPU event duration by executed ops.
    # Fail visibly if capture not supported; no hidden alternate timing backend.
    for _ in range(4):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):fn()
    graph.replay();torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    durations=[]
    for _ in range(repeats):
        start.record();graph.replay();end.record();end.synchronize()
        durations.append(start.elapsed_time(end)/graph_iters)
    return statistics.median(durations),min(durations),max(durations)


def cpu_smoke():
    torch.manual_seed(3)
    a=.80+.19*torch.rand(1,33,3,4)
    u=.1*torch.randn_like(a)
    s=.03*torch.randn(1,3,4)
    expected=_scan.thunder_scan(a,u,s,compass=16)
    carry=s; outputs=[]
    for t in range(a.shape[1]):
        carry=a[:,t]*carry+u[:,t];outputs.append(carry)
    torch.testing.assert_close(expected,torch.stack(outputs,dim=1),atol=1e-5,rtol=1e-5)
    m1=make_esa('fast',dim=48,heads=6,scan='swqr_adaptive')
    m2=make_esa('fast',dim=48,heads=6,scan='thunder')
    m2.load_state_dict(m1.state_dict(),strict=True)
    print('CPU SMOKE PASS (no GPU/kernel benchmarking performed)')


def native_thunder(a,u,state):
    from esa import native as en
    if not en.cuda_available():
        raise RuntimeError('ESA native CUDA extension not available')
    # ESA native starts from zero; adjust first write for nonzero initial state.
    if state is not None:
        u=torch.cat((u[:,:1]+a[:,:1]*state[:,None],u[:,1:]),dim=1)
    return en.thunder_scan_hierarchical(a.contiguous(),u.contiguous(),16)


def compare_forward(a,u,state,warps, reference):
    x=_scan.swqr_short_scan(a,u,state,warps=warps)
    y=reference
    err=(x-y).abs().max().item()
    ok=torch.allclose(x,y,rtol=3e-4,atol=3e-5)
    if not ok:raise AssertionError(f'SWQR-Short forward parity failed T={a.shape[1]} warps={warps} diff={err}')
    return err


def parity(device, out, H, Dh):
    rows=[]
    for B in (1,2):
      for T in (7,31,128,129,256,512,1024,1536,2048):
       torch.manual_seed(51+T+B)
       a=(.82+.17*torch.rand(B,T,H,Dh,device=device)).contiguous()
       u=(.03*torch.randn_like(a)).contiguous()
       for cached in (False, True):
        s=(.02*torch.randn(B,H,Dh,device=device)).contiguous() if cached else None
        ref=_scan.swqr_scan(a,u,s,chunk=128)
        ref_esa=native_thunder(a,u,s)
        ok=torch.allclose(ref,ref_esa,atol=5e-5,rtol=3e-4)
        if not ok: raise AssertionError('ESA native Thunder versus SWQR reference mismatch')
        for warps in (4,8,16):
            err=compare_forward(a,u,s,warps,ref)
            row=dict(batch=B,context=T,cached=cached,warps=warps,max_abs_error=err,pass_parity=True)
            rows.append(row)
            print('[forward parity]',row,flush=True)
    write_csv(out/'forward_parity.csv',rows)
    grads=[]
    for T in (128,512,1024,1536):
      for cached in (False,True):
        B=2
        torch.manual_seed(400+T+int(cached))
        A=(.82+.17*torch.rand(B,T,H,Dh,device=device)).contiguous()
        U=(.05*torch.randn_like(A)).contiguous()
        S=(.02*torch.randn(B,H,Dh,device=device)).contiguous() if cached else None
        grad=torch.randn_like(A)/T**.5
        def compute(backend,warps=8):
            a=A.detach().clone().requires_grad_()
            u=U.detach().clone().requires_grad_()
            s=S.detach().clone().requires_grad_() if cached else None
            if backend=='short':e=_scan.swqr_short_scan(a,u,s,warps=warps)
            elif backend=='swqr':e=_scan.swqr_scan(a,u,s)
            else:e=_scan.thunder_scan(a,u,s,compass=16)
            t=(a,u,s) if cached else (a,u)
            g=torch.autograd.grad(e,t,grad_outputs=grad)
            return e.detach(),tuple(z.detach() for z in g)
        er,gr=compute('swqr')
        for warps in (4,8,16):
            e,g=compute('short',warps)
            err=max((u-v).abs().max().item() for u,v in zip(g,gr))
            valid=torch.allclose(e,er,atol=5e-5,rtol=4e-4) and all(torch.allclose(u,v,atol=5e-5,rtol=4e-4) for u,v in zip(g,gr))
            row=dict(context=T,cached=cached,warps=warps,max_abs_gradient_error=err,pass_parity=valid)
            grads.append(row);print('[backward parity]',row,flush=True)
            if not valid:raise AssertionError('Short backward numerical parity failed')
    write_csv(out/'backward_parity.csv',grads)


def scan_speed(device,out,H,Dh,contexts,batches,repeats):
    rows=[]
    for B in batches:
      for T in contexts:
        torch.manual_seed(T+B)
        a=(.82+.17*torch.rand(B,T,H,Dh,device=device)).contiguous()
        u=((1-a)*torch.tanh(torch.randn_like(a))).contiguous()
        methods={'swqr_original_cuda':lambda:_scan.swqr_scan(a,u,chunk=128),
                 'esa_native_thunder_cuda':lambda:native_thunder(a,u,None)}
        if T<=2048:
            for w in (4,8,16):
                methods[f'swqr_short_w{w}']=lambda w=w:_scan.swqr_short_scan(a,u,warps=w)
        with torch.inference_mode():
          for key,fn in methods.items():
            ms,low,high=event_bench(fn,repeats=repeats)
            row=dict(batch=B,context=T,backend=key,latency_us=round(ms*1000,3),
                     tokens_per_sec=round(B*T*1000/ms),min_ms=low,max_ms=high,
                     timing='cuda_graph_events_32_ops')
            rows.append(row);print('[native scan speed]',row,flush=True)
    write_csv(out/'scan_speed.csv',rows)


def full_model_speed(device,out,dim,heads,layers,repeats):
    # All models have identical weights and architecture. Compare the scan.
    base=make_model('fast',dim=dim,heads=heads,layers=layers,scan='swqr').to(device).eval()
    variants={'swqr_original_cuda':base}
    for w in (4,8,16):
        m=make_model('fast',dim=dim,heads=heads,layers=layers,scan='swqr_adaptive',swqr_warps=w).to(device).eval()
        m.load_state_dict(base.state_dict(),strict=True)
        variants[f'swqr_adaptive_w{w}']=m
    rows=[];checks=[]
    for B in (1,2):
      for T in (128,256,512,1024,1536,2048):
        torch.manual_seed(T+B)
        toks=torch.randint(0,256,(B,T),device=device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            baseline=forward_tokens(base,toks)[0]
            for key,m in variants.items():
                got=forward_tokens(m,toks)[0]
                err=(got-baseline).abs().max().item()
                ok=torch.allclose(got,baseline,atol=.03,rtol=.02)
                checks.append(dict(batch=B,context=T,backend=key,max_abs_logit_error=err,pass_parity=ok))
                if not ok:raise AssertionError(f'Full model mismatch: {key} B{B} T{T} diff={err}')
            for key,m in variants.items():
                fn=lambda m=m:forward_tokens(m,toks)
                ms,lo,hi=event_bench(fn,repeats=max(3,repeats//2),graph_iters=8)
                rows.append(dict(batch=B,context=T,backend=key,latency_ms=ms,
                                 tokens_per_sec=round(B*T*1000/ms),min_ms=lo,max_ms=hi))
                print('[model prefill]',rows[-1],flush=True)
    write_csv(out/'full_model_prefill.csv',rows)
    write_csv(out/'full_model_parity.csv',checks)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='short_bench_results')
    ap.add_argument('--cpu-smoke',action='store_true')
    ap.add_argument('--contexts',default='128,256,512,1024,1536,2048,4096')
    ap.add_argument('--batches',default='1,2,8')
    ap.add_argument('--dim',type=int,default=648)
    ap.add_argument('--heads',type=int,default=6)
    ap.add_argument('--layers',type=int,default=6)
    ap.add_argument('--repeats',type=int,default=8)
    a=ap.parse_args()
    if a.cpu_smoke:return cpu_smoke()
    if not torch.cuda.is_available():raise RuntimeError('Native benchmark requires CUDA')
    if a.dim%a.heads:raise ValueError('dim must be divisible by heads')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    print('[GPU]',torch.cuda.get_device_name(0),'torch',torch.__version__,'cuda',torch.version.cuda,flush=True)
    # Compilation before timing, fail if either extension is unavailable.
    _scan.prepare_swqr()
    _scan.prepare_swqr_short()
    import esa.native as en
    if not en.cuda_available():raise RuntimeError('ESA native CUDA not loaded: '+str(en.import_error()))
    dev=torch.device('cuda:0')
    provenance=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                    swqr_extension=str(_scan._EXTENSIONS['mlbricks_fastesa_swqr_scan_030'].__file__),
                    short_extension=str(_scan._EXTENSIONS['mlbricks_fastesa_swqr_short_031'].__file__),
                    esa_extension=str(en._C.__file__),
                    objective='Identical learned FAST recurrence, short native scan vs old SWQR and native Thunder',
                    full_model='6 layer FAST, same weights, FP16 projections/FP32 state',
                    timing='CUDA graph replay + CUDA events; no claimed torch.compile',
                    cuda_compile_verified=True)
    (out/'native_proof.json').write_text(json.dumps(provenance,indent=2))
    contexts=[int(x) for x in a.contexts.split(',')]
    batches=[int(x) for x in a.batches.split(',')]
    parity(dev,out,a.heads,a.dim//a.heads)
    scan_speed(dev,out,a.heads,a.dim//a.heads,contexts,batches,a.repeats)
    full_model_speed(dev,out,a.dim,a.heads,a.layers,a.repeats)
    print('SUCCESS: GPU native short CUDA benchmark parity & timings complete',flush=True)


if __name__=='__main__':main()
