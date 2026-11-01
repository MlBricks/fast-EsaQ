import tempfile
import pytest
import torch
import torch.nn.functional as F

from fastesa import (FastESA, QualityESA, make_esa, make_model,
                     forward_tokens, load_checkpoint)
from fastesa._original import progress_learned, progress_hybrid


@pytest.mark.parametrize("variant", ["fast", "quality"])
def test_exact_mixer_matches_unmodified_source(variant):
    torch.manual_seed(123)
    m = make_esa(variant, dim=48, heads=6, scan="original")
    native = (progress_learned.ESAFullMixer(48, 6, "learned")
              if variant == "fast" else progress_hybrid.HybridESA(48, 6))
    native.load_state_dict(m.state_dict(), strict=True)
    x = torch.randn(2, 67, 48)
    a, sa = m(x)
    b, sb = native(x)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    assert sa.shape == (2, 6, 8)
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("variant", ["fast", "quality"])
def test_mixer_streaming_and_gradients(variant):
    torch.manual_seed(1234)
    mixer = make_esa(variant, dim=48, heads=6, scan="original").eval()
    x = torch.randn(1, 70, 48)
    full, s = mixer(x)
    one, prev = mixer(x[:, :32])
    two, end = mixer(x[:, 32:], prev)
    torch.testing.assert_close(torch.cat([one, two], 1), full, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(s, end, rtol=5e-4, atol=5e-4)
    loss = full.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mixer.parameters())


@pytest.mark.parametrize("variant", ["fast", "quality"])
def test_full_model_exact_and_checkpoint(variant, tmp_path):
    torch.manual_seed(21)
    model = make_model(variant, dim=48, heads=6, layers=1, seed=7, scan="original")
    original = (progress_learned.KaggleRecall(256, 48, 6, 1, "learned") if variant == "fast"
                else progress_hybrid.RecallModel(256, 48, 6, 1, "hybrid", 32))
    original.load_state_dict(model.state_dict(), strict=True)
    ids = torch.randint(0, 256, (2, 65))
    y, states = forward_tokens(model, ids)
    ref = (original.forward_chunk(ids) if variant == "fast" else original(ids))
    torch.testing.assert_close(y, ref[0], atol=0, rtol=0)
    assert len(states) == 1 and states[0].shape == (2, 6, 8)
    ck = tmp_path / "latest.pt"
    torch.save({"variant": "learned" if variant == "fast" else "hybrid",
                "model": model.state_dict()}, ck)
    restored = load_checkpoint(ck, scan="original")
    torch.testing.assert_close(forward_tokens(restored, ids)[0], y, atol=0, rtol=0)


def test_constraints():
    with pytest.raises(ValueError):
        make_esa("fast", dim=48, chunk_size=16)
    with pytest.raises(AssertionError):
        make_esa("quality", dim=48, heads=4)
    with pytest.raises(ValueError):
        make_esa("invalid", dim=48)
