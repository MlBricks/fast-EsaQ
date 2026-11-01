# fastEsa — MLBricks PROGRESS ESA variants

This **independent, installable Python library** packages the two exact model
implementations from the supplied Kaggle PROGRESS notebooks without changing
any of their recurrence equations, scans, gate parameters, or checkpoint keys.

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
pip install ./fastEsa
# Or from the generated wheel (torch/numpy must be available):
pip install ./fastesa-0.1.0-py3-none-any.whl
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
checkpoint directories or modify existing experiments. No Triton or CUDA
extension compilation is required. CPU and CUDA PyTorch are supported.

## Mathematical summary

`fast`: `q,g,vraw = W_qgv x`; `A = alpha_h +(beta_h-alpha_h)*sigmoid(g)`;
`U = (1-A)*tanh(vraw)`; `E = A*Eprev + U`;
`Y = W_O(sigmoid(q)*RMSNorm(E))`. `alpha/beta` are trainable bounds.
Scan uses cumulative products/divisions over chunks of 32.

`quality`: `q,g,vraw = W_qgv x`; `A = exp(-ln(2)*softplus(g+b_h)/half_life_h)`;
`U = (1-A)*tanh(vraw)`; `E = A*Eprev + U`;
`Y = W_O(sigmoid(q)*RMSNorm(E))`. `half_life_h` and bias are trainable.
Scan uses the original division-free associative composition in chunks.

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
