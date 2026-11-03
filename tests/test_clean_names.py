"""Stable public and internal naming and checkpoint migration."""
import torch
import pytest
from fastesa import (ESA, BaseESA, ESAModel, BaseESAModel, ESAConfig,
                     GraphDecoder, CUDAGreedyGraph, CurrentESA, make_esa, make_model,
                     load_checkpoint, normalize_variant, normalize_scan)
import importlib
scan_module = importlib.import_module("fastesa.scan")
from fastesa import _scan as legacy_scan
from fastesa.core import ESA as EngineESA, BaseESA as EngineBaseESA


def test_current_default_is_half_life():
    assert isinstance(make_esa(dim=48,heads=6,scan='reference'), EngineESA)
    assert isinstance(make_model(dim=48,heads=6,layers=1,scan='reference'), ESAModel)
    assert isinstance(make_esa('base',dim=48,heads=6,scan='reference'), EngineBaseESA)
    assert isinstance(make_model('base',dim=48,heads=6,layers=1,scan='reference'), BaseESAModel)
    assert normalize_variant('fast')=='base'
    assert normalize_variant('quality')=='current'
    assert normalize_scan('swqr')=='parallel'
    assert normalize_scan('original')=='reference'
    assert GraphDecoder is CUDAGreedyGraph
    assert callable(scan_module.scan)
    assert scan_module.scan is scan_module.auto_scan
    assert ESAConfig(embd=48).build().__class__ is CurrentESA


def test_new_scan_aliases_reference_equivalence():
    torch.manual_seed(7)
    current=make_esa(dim=48,heads=6,scan='reference')
    fast=make_esa(dim=48,heads=6,scan='hierarchical')
    fast.load_state_dict(current.state_dict())
    x=torch.randn(2,96,48)
    a,s=current(x); b,t=fast(x)
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(s,t,atol=2e-4,rtol=2e-4)


def test_old_checkpoint_auto_detects_variant(tmp_path):
    for variant in ('base','current'):
        model=make_model(variant,dim=48,heads=6,layers=1,scan='reference')
        target=tmp_path/f'{variant}.pt'
        torch.save({'model':model.state_dict()},target)
        loaded=load_checkpoint(target,scan='reference')
        assert isinstance(loaded, BaseESAModel if variant=='base' else ESAModel)
        assert list(model.state_dict())==list(loaded.state_dict())


def test_reject_invalid_sequential_multi_token():
    m=EngineESA(dim=48,heads=6,scan='sequential')
    with pytest.raises(ValueError,match='one token'):
        m(torch.randn(1,2,48))
