"""Regression coverage for explicit 2026 ESA default half-life values."""
import torch
import pytest

from fastesa import ESA, ESAConfig, make_esa, make_model, load_checkpoint
from fastesa.core import DEFAULT_CURRENT_HALF_LIVES
from fastesa._original import progress_hybrid

EXPECTED = torch.tensor([0.125, 0.5, 2.0, 8.0, 32.0, 64.0])


def check_heads(mixer):
    p = mixer.log_half_life
    assert p.requires_grad and p.dtype == torch.float32
    torch.testing.assert_close(p.detach().exp(), EXPECTED, rtol=1e-6, atol=1e-6)
    assert "log_half_life" in mixer.state_dict()


def test_explicit_tuple():
    assert tuple(DEFAULT_CURRENT_HALF_LIVES) == tuple(EXPECTED.tolist())


@pytest.mark.parametrize("scan", ["reference", "auto", "hierarchical"])
def test_standalone_mixer_defaults(scan):
    m = make_esa("current", dim=48, heads=6, scan=scan)
    check_heads(m)
    # A continuation step updates the learnable log-half-lives.
    x = torch.randn(2, 8, 48)
    y, _ = m(x)
    y.square().mean().backward()
    assert m.log_half_life.grad is not None
    assert torch.isfinite(m.log_half_life.grad).all()


@pytest.mark.parametrize("scan", ["reference", "auto"])
@pytest.mark.parametrize("seed", [None, 20261103])
def test_full_model_defaults(scan, seed):
    model = make_model("current", dim=48, heads=6, layers=2,
                       vocab_size=256, scan=scan, seed=seed)
    for block in model.blocks:
        check_heads(block.esa)
    assert model.blocks[0].esa.log_half_life is not model.blocks[1].esa.log_half_life


def test_public_api_and_config_defaults():
    check_heads(ESA(embd=48, head=6, backend="pytorch"))
    check_heads(ESAConfig(embd=48, head=6, backend="pytorch").build())


def test_base_unchanged_and_historical_unchanged():
    assert "raw_alpha" in make_esa("base", dim=48, heads=6).state_dict()
    historical = progress_hybrid.HybridESA(48, 6)
    torch.testing.assert_close(historical.log_half_life.exp().detach(),
                               torch.tensor([16., 64., 256., 1024., 4096., 8192.]),
                               rtol=1e-6, atol=0.)


def test_checkpoint_restores_saved_weights(tmp_path):
    model = make_model("current", dim=48, heads=6, layers=1, vocab_size=256)
    with torch.no_grad():
        model.blocks[0].esa.log_half_life.copy_(torch.log(torch.tensor([1.,2.,3.,4.,5.,6.])))
    path = tmp_path / "old_model.pt"
    torch.save({"variant": "hybrid", "model": model.state_dict()}, path)
    restored = load_checkpoint(path, scan="reference")
    torch.testing.assert_close(restored.blocks[0].esa.log_half_life.exp(),
                               torch.tensor([1.,2.,3.,4.,5.,6.]))
