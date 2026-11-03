# fastEsa — MLBricks PROGRESS ESA variants

This **independent, installable Python library** includes the two original
PROGRESS model implementations alongside optimized FAST execution backends.
The original model equations, gate parameters, and checkpoint keys are kept.
Select `scan="original"` to reproduce the exact original scan; the default
`scan="auto"` selects the CUDA SWQR backend for **both Fast and Quality ESA**
(with Thunder fallback on CPU). Quality ESA also includes the original fused
half-life CUDA decode kernel and optional whole-model CUDA Graph replay.

| Public variant | Exact notebook implementation | Mixer class |
|---|---|---|
| `fast` | **PROGRESS learned** from “Learned ESA vs Learned Long-Memory ESA” | `ESAFullMixer(dim, heads, variant='learned')` |
| `quality` | **PROGRESS hybrid** from “Original ESA Long-Memory vs Bounded-Write Hybrid” | `HybridESA(dim, heads, chunk_size=32)` |

The variant names are the **requested product labels**. The notebooks are
synthetic associative-recall benchmarks; calling a variant `fast` or `quality`
does **not** establish that it is faster or has better language quality across
datasets/hardware. No trained weights were included in the uploaded material.

## Install

```bash
pip install .
# Or from the generated wheel (torch/numpy must be available):
pip install ./dist/fastesa-0.3.8-py3-none-any.whl
```

## Mixer use (the original equations, without modifications)

```python
import torch
from fastesa import make_esa

x = torch.randn(2, 128, 384)
fast = make_esa("fast", dim=384, heads=6)
quality = make_esa("quality", dim=384, heads=6)

for mixer in (fast, quality):
    y, cache = mixer(x)  # [B,T,384], [B,6,64]
    second_y, second_cache = mixer(x[:, :1], cache)
```

## Full recall model / neural language-model-shaped output

```python
import torch
from fastesa import make_model, forward_tokens

fast = make_model("fast", vocab_size=256, dim=384, heads=6, layers=6)
quality = make_model("quality", vocab_size=256, dim=384, heads=6, layers=6)
ids = torch.randint(0, 256, (2, 64))

for model in (fast, quality):
    logits, states = forward_tokens(model, ids)
    logits_next, states = forward_tokens(model, ids[:, :1], states)
    print(logits.shape, len(states))  # [2,256], 6
```

*The two full models intentionally retain their original differing `forward`
methods.* The `forward_tokens` helper is a common API for both. The model
heads return **last-token logits only**, just as the original notebooks did;
these are not drop-in replacements for a standard all-token causal LM API.

## Load actual PROGRESS checkpoints (when you have them)

```python
from fastesa import load_checkpoint

fast = load_checkpoint(".../learned/latest.pt", device="cuda")
quality = load_checkpoint(".../hybrid/latest.pt", device="cuda", trusted=True)
```

Only set `trusted=True` for a file you trust; the source hybrid notebook saves
NumPy RNG objects that can require unsafe pickle deserialization. Checkpoint
file contents and architecture must match the exact original notebook. The
loader accepts either full notebook checkpoints or raw model state dicts.

## Reproduce the notebook initializer

```python
model_fast = make_model("fast", seed=42)
model_quality = make_model("quality", seed=2026)
```

These *different* seeds reproduce their respective source notebooks'
initialization—not a shared learned-vs-hybrid initialization.

## Exact original source

Two full worker scripts were extracted directly, without editing, from cell 6
of their respective notebooks, and retained as:

- `fastesa/_original/progress_learned.py` (includes the original `PROGRESS learned` log line)
- `fastesa/_original/progress_hybrid.py` (includes the original `PROGRESS hybrid` log line)

The original CLI training and evaluation code is available, untouched:

```bash
python -m fastesa._original.progress_learned --smoke --device cpu --dim 48 --heads 6 --layers 1
python -m fastesa._original.progress_hybrid --action smoke --device cpu --dim 48 --heads 6 --layers 1
```

Original scripts use their original default Kaggle output directories and
model labels (`learned` and `hybrid`). New public aliases do not rename
checkpoint directories or modify existing experiments. The original reference and Thunder PyTorch paths need no CUDA extension.
The default Fast/Quality CUDA SWQR path lazily compiles an optional native CUDA
extension (set `scan="thunder"` to skip it). CPU and CUDA PyTorch are supported.

## Mathematical summary

`fast`: `q,g,vraw = W_qgv x`; `A = alpha_h +(beta_h-alpha_h)*sigmoid(g)`;
`U = (1-A)*tanh(vraw)`; `E = A*Eprev + U`;
`Y = W_O(sigmoid(q)*RMSNorm(E))`. `alpha/beta` are trainable bounds.
Original scan uses cumulative products/divisions over chunks of 32;
v0.3 default FAST dispatch uses SWQR CUDA or Thunder PyTorch.

`quality`: `q,g,vraw = W_qgv x`; `A = exp(-ln(2)*softplus(g+b_h)/half_life_h)`;
`U = (1-A)*tanh(vraw)`; `E = A*Eprev + U`;
`Y = W_O(sigmoid(q)*RMSNorm(E))`. `half_life_h` and bias are trainable.
Original scan uses division-free associative composition; the optimized path uses SWQR CUDA or Thunder, plus an inference-only fused CUDA decode kernel.

Both models have fixed-size recurrent state `[B, H, D/H]` per block.
The quality implementation requires six heads and the fast implementation
uses a fixed scan chunk size of 32, exactly as in the source notebooks.

## License

Contains derived ESA research code. Retains the accompanying **ESA PolyForm
Noncommercial 1.0.0** license and required commercial license notice from the
uploaded ESA repository. See `LICENSE.md` and `LICENSING_NOTICE.md`.

## Validation

```bash
python -m pytest -q tests
```

Original extraction checksums are recorded in `SOURCE_PROVENANCE.txt`.

## v0.3 fastEsa backend patch

**Current FAST default:** `scan='auto'` — SWQR native CUDA for multi-token
training/prefill, PyTorch Thunder when CUDA SWQR is unavailable, and the
one-token direct recurrence during autoregressive decoding. This is an
**execution backend change, not a new architecture or pretrained checkpoint**.
The `quality` variant continues to use its original PROGRESS hybrid scan.

```python
import torch
from fastesa import make_esa, make_model, forward_tokens, prepare_swqr

mixer = make_esa('fast', dim=384, heads=6, scan='auto')
model = make_model('fast', dim=384, heads=6, layers=6)
x = torch.randn(2, 128, 384)
y, state = mixer(x)
next_y, next_state = mixer(x[:, :1], state)  # direct recurrence

# On CUDA, prepare_swqr() compiles the optional native extension once.
# Use scan='swqr' if you want to REQUIRE native SWQR and fail on missing CUDA.
# Use scan='original' for exact historical PROGRESS training behavior.
```

Source distributions and wheels contain CUDA sources, not precompiled GPU
binaries. A C++ compiler, a matching CUDA toolkit with `nvcc`, and a CUDA
PyTorch wheel are needed to compile native SWQR on the target host. The
module is compiled lazily; CPU use does not require CUDA. See
`RELEASE_NOTES_v0.3.md` for benchmarks/validation limitations.

## Experimental SWQR-Short CUDA (v0.3.1)

The optional backend `scan="swqr_short"` implements the same learned-retention
FAST recurrence using a **one-launch, block-parallel native CUDA scan** for
sequences of 2–2048 tokens. Each warp accesses contiguous channels; different
warps handle adjacent sequence segments and exchange affine summaries in
shared memory. It is a separate kernel from the v0.3 native SWQR long scan.
No parameter names or checkpoint keys are changed.

```python
from fastesa import make_esa
short = make_esa("fast", dim=648, heads=6, scan="swqr_short", swqr_warps=8)
adaptive = make_esa("fast", dim=648, heads=6, scan="swqr_adaptive", swqr_warps=8)
```

- `swqr_short`: 2–2048 sequence lengths, explicit CUDA compilation required.
- `swqr_adaptive`: use SWQR-Short up to 1024 by default and **original** SWQR above 1024.
- `swqr`: unchanged native CUDA behavior. `auto` is also unchanged in v0.3.1.
- 1-token decode always uses the lightweight direct recurrent state update.
- `swqr_warps`: 4, 8, or 16. Benchmark all values; no universal optimum assumed.
- `prepare_swqr_short()` explicitly compiles the extension before GPU timing.
- Experimental autograd uses reverse-time SWQR-Short for `dA`, `dU`, and
  optional `dInitialState` with retained intermediate output states. This
  is not yet a fully fused training backward kernel.

**Validation:** CPU reference emulation tests are not evidence of CUDA
correctness or speed. Run the supplied T4 parity/throughput notebook before
choosing a production default. ESA native Thunder code is separately licensed
under the terms accompanying that source; this patch does not copy ESA kernels.

## v0.3.2 — Batch-aware short-context CUDA dispatch

`scan="auto"` now uses **measured T4-specific** shape selection:

| Device | Batch | Context | Scan |
|---|---:|---:|---|
| Tesla T4 / NVIDIA T4 (sm75) | 1 | 128–512 | SWQR-Short, 16 warps |
| Tesla T4 / NVIDIA T4 (sm75) | 2 | 128–256 | SWQR-Short, 8 warps |
| Any CUDA device | other multi-token shapes | 2+ | Original SWQR |
| CPU | any | 2+ | PyTorch Thunder |
| Any device | any | 1 | Direct recurrent update |

The `swqr_short`, `swqr`, `swqr_adaptive`, `thunder`, and `original` explicit
options remain available, including `swqr_warps` for manual CUDA experiments.
`swqr_adaptive` is now **batch-aware** by default; direct calls to
`swqr_adaptive_scan(..., cutoff=N)` retain the older length-only policy.

On unprofiled GPUs, no short-path speedup is assumed: `auto` uses original SWQR.
A failed short CUDA build falls back to original SWQR and finally Thunder when
using `scan="auto"`; strict native modes raise on failure.

To inspect the selection without launching a kernel:

```python
from fastesa import select_swqr_backend
print(select_swqr_backend(1, 256, "cuda:0"))
```

Run `python benchmark_dispatch_v032.py --smoke` for CPU checks, or execute
`python benchmark_dispatch_v032.py --device cuda:0` on Kaggle T4 to test parity,
gradients and throughput. Native v0.3.2 CUDA has **not yet been benchmarked**
in the package-build environment.


## v0.3.8 patch: accelerated half-life Quality ESA

The original PROGRESS `HybridESA` remains in `fastesa/_original/progress_hybrid.py`.
New public `QualityESA` and `QualityESAModel` **reuse its exact parameters**:
`qgv`, `W_O`, `state_norm`, `log_half_life`, `raw_decay_bias`.
No weights need to be converted, and the mathematical recurrence is unchanged:

```text
A = exp(-ln(2) * softplus(G + raw_decay_bias) / exp(log_half_life))
U = (1 - A) * tanh(V)
E = A * E_previous + U
read = sigmoid(Q) * RMSNorm(E, state_norm.weight)
Y = W_O(read)
```

Implementation changes:

- `scan='auto'`: CUDA SWQR forward/backward for multi-token training and prefill; `thunder` on CPU. `scan='original'` is still the unchanged reference.
- `fused_decode='auto'`: for `T=1` during CUDA inference, a C++/CUDA kernel computes the **original half-life gate**, bounded write, FP32 state, learned RMSNorm and sigmoid-query readout in one launch. `fused_decode='on'` strictly requires native CUDA; `off` always uses normal PyTorch operations. The experimental cached reciprocal/`exp2f` gate is intentionally **not** the default; it was slower in the earlier T4 run.
- `CUDAGreedyGraph`: optional **per-GPU explicit-stream** autoregressive capture, supporting graph-unrolled generation without CPU launch per token. Capture after training with model `.eval()`; recapture after optimizer steps/weight reloads.
- Native code is **lazily compiled** and requires nvcc only for CUDA acceleration; CPU usage requires no CUDA build.

```python
import torch
from fastesa import make_model, forward_tokens, CUDAGreedyGraph, prepare_quality_decode

# Use gpu cuda:1 in a dual-T4 setup or cuda:0 on a single GPU.
device = 'cuda:1'
model = make_model('quality', layers=6, dim=384, heads=6,
                   scan='auto', fused_decode='on', seed=2026).to(device).eval()
prepare_quality_decode()  # compile once before graph capture
prompt = torch.tensor([[79, 110, 99, 101]], device=device)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    last_logits, states = forward_tokens(model, prompt)
    one_token = last_logits.argmax(-1).view(-1, 1)
    _, updated_states = forward_tokens(model, one_token, states)

# Opt-in CUDA Graph; exact 128-token greedy continuation; no retraining.
decoder = CUDAGreedyGraph(model, prompt, unroll=8)
print('CUDA Graph max state difference:', decoder.check_parity())
print(decoder.generate(128).shape)  # [1,128] on GPU
```

Reference/checkpoint usage:

```python
baseline = make_model('quality', scan='original', seed=2026)
accelerated = make_model('quality', scan='auto', seed=2026)
accelerated.load_state_dict(baseline.state_dict(), strict=True)
# Existing original PROGRESS checkpoint format can be restored directly:
# load_checkpoint('/path/to/latest.pt', variant='quality', scan='auto')
```

CUDA validation on Kaggle T4:

```bash
pip install -e .
python -m pytest -q tests
python examples/benchmark_quality_optimized.py
```

The 141 CPU tests and one CUDA-skipped test were executed locally. CUDA
extension compilation, GPU numerical parity and graph throughput have **not**
been run against this packaged v0.3.8 patch in this CPU-only environment.
The previously reported ~1,727 byte-tokens/s Quality ESA graph decode was
from the separate Kaggle notebook (v0.3.7.1), not a measurement of this patch.

**License:** the upstream noncommercial licensing notice remains in force.


## Unified ESA API (v0.4.0)

The current ESA is the half-life recurrent variant. The historical learned-bounds
architecture remains available as `variant="base"`. These are architecture choices;
`backend` chooses an execution policy only.

```python
from fastesa import ESA, ESAConfig, save_esa_checkpoint, load_esa_checkpoint

current = ESA(embd=384, head=6, backend="auto")
base = ESA(embd=384, head=6, variant="base", backend="auto")
explicit = ESA(embd=384, head=6, variant="current", backend="auto")

# state_dict tensor keys are unchanged; architecture metadata prevents accidental conversion.
save_esa_checkpoint(current, "current_esa.pt")
restored = load_esa_checkpoint("current_esa.pt", backend="auto")

# Force reference PyTorch computation, or require CUDA:
current.set_backend("pytorch")
current.set_backend("native")
```

`from fastesa import ESA` is the API of **this repository**. The equivalent
`from mlbricks import ESA` import becomes available only after this component is
merged into the separate MLBricks Kit distribution.

The former `FastESA`/`QualityESA`, SWQR scan names, and CUDA-graph names remain
importable for compatibility. Source files use `core`, `base`, `scan`, `decode`,
`graph` and `cuda/`. Existing trained checkpoint weights are not converted.
