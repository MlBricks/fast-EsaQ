from pathlib import Path
import torch
import pytest
from fastesa import ESA, CurrentESA, BaseESA, save_esa_checkpoint, load_esa_checkpoint
from fastesa.core import ESA as SourceCurrent, BaseESA as SourceBase

@pytest.mark.parametrize("variant,cls,internal,retention", [
    ("current",CurrentESA,SourceCurrent,"log_half_life"),
    ("base",BaseESA,SourceBase,"raw_alpha"),
])
def test_factory_and_checkpoint_layout(variant,cls,internal,retention):
    m=ESA(embd=48,head=6,variant=variant,backend="pytorch")
    assert isinstance(m, ESA) and isinstance(m, cls) and isinstance(m, internal)
    assert m.esa_variant==variant
    assert retention in m.state_dict()
    assert not any(k.startswith("engine.") or k.startswith("impl.") for k in m.state_dict())
    if variant=="base":
        assert m.variant=="learned"  # Must remain unchanged for reference forward.
    x=torch.randn(2,7,48)
    y,s=m(x)
    assert y.shape==(2,7,48) and s.shape==(2,6,8)
    z,t=m.decode_step(x[:,0],s)
    assert z.shape==(2,1,48) and t.shape==(2,6,8)


def test_default_and_explicit_current():
    a=ESA(embd=48, head=6, backend="pytorch")
    b=ESA(embd=48, head=6, variant="current",backend="pytorch")
    assert type(a) is type(b) is CurrentESA


def test_reference_weight_parity():
    for v,cl in (("current",SourceCurrent),("base",SourceBase)):
        torch.manual_seed(7)
        reference=cl(dim=48,heads=6,scan="reference")
        proposed=ESA(embd=48,head=6,variant=v,backend="pytorch")
        proposed.load_state_dict(reference.state_dict(),strict=True)
        x=torch.randn(2,16,48,requires_grad=True)
        y1,s1=reference(x)
        y2,s2=proposed(x)
        torch.testing.assert_close(y1,y2,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(s1,s2,atol=1e-6,rtol=1e-6)
        (y1.float().square().mean()).backward(retain_graph=True)
        g1=x.grad.clone();x.grad.zero_()
        (y2.float().square().mean()).backward()
        torch.testing.assert_close(g1,x.grad,atol=1e-6,rtol=1e-6)


def test_backend_selection():
    for variant in ("current","base"):
        model=ESA(embd=48,head=6,variant=variant,backend="auto")
        assert model.resolved_backend()=="auto"
        before=list(model.state_dict().keys())
        model.set_backend("pytorch")
        assert model.scan=="reference"
        model.set_backend("auto")
        assert model.scan=="auto"
        assert list(model.state_dict().keys())==before
        model.set_backend("native")
        with pytest.raises(RuntimeError,match="CUDA"):
            model(torch.randn(1,8,48))
    with pytest.raises(ValueError,match="backend"):
        ESA(embd=48,head=6,backend="fast")
    with pytest.raises(ValueError,match="variant"):
        ESA(embd=48,head=6,variant="unknown")


@pytest.mark.parametrize("variant", ["current","base"])
def test_checkpoint_roundtrip(tmp_path,variant):
    model=ESA(embd=48,head=6,variant=variant,backend="pytorch")
    path=tmp_path/"model.pt"
    save_esa_checkpoint(model,path)
    restored=load_esa_checkpoint(path,backend="pytorch")
    assert restored.esa_variant==variant
    for (k,x) in model.state_dict().items():
        torch.testing.assert_close(x, restored.state_dict()[k])
    bad="base" if variant=="current" else "current"
    with pytest.raises(ValueError,match="Requested"):
        load_esa_checkpoint(path,variant=bad)
    plain=tmp_path/"plain.pt"
    torch.save(model.state_dict(),plain)
    assert load_esa_checkpoint(plain,backend="pytorch").esa_variant==variant


def test_meta_cannot_mislabel(tmp_path):
    model=ESA(embd=48,head=6,variant="base",backend="pytorch")
    bad=tmp_path/"bad.pt"
    torch.save({"architecture":"current", "state_dict":model.state_dict()},bad)
    with pytest.raises(ValueError,match="metadata"):
        load_esa_checkpoint(bad)


def test_esa_config():
    from fastesa import ESAConfig
    cfg = ESAConfig(embd=48, head=6, variant='current', backend='auto')
    m = cfg.build()
    assert isinstance(m, CurrentESA)
    assert ESAConfig(embd=48, head=6, variant='base').build().esa_variant == 'base'
    with pytest.raises(ValueError, match='head=6'):
        ESAConfig(embd=64, head=8, variant='current')
