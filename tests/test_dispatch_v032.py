"""Tests for shape-aware FAST v0.3.2 dispatch, without requiring a CUDA GPU."""
import pytest
import torch
from fastesa import make_esa, select_swqr_backend
from fastesa import _scan


@pytest.mark.parametrize('batch,seq,expected', [
    (1, 1, ('direct', None)),
    (1, 127, ('swqr', None)),
    (1, 128, ('swqr_short', 16)),
    (1, 256, ('swqr_short', 16)),
    (1, 512, ('swqr_short', 16)),
    (1, 513, ('swqr', None)),
    (1, 1024, ('swqr', None)),
    (2, 128, ('swqr_short', 8)),
    (2, 256, ('swqr_short', 8)),
    (2, 257, ('swqr', None)),
    (2, 512, ('swqr', None)),
    (8, 128, ('swqr', None)),
    (8, 256, ('swqr', None)),
    (8, 1024, ('swqr', None)),
    (1, 131072, ('swqr', None)),
])
def test_t4_dispatch_policy(batch, seq, expected):
    assert select_swqr_backend(batch, seq, 'cuda:0', device_name='Tesla T4',
                               capability=(7, 5)) == expected


@pytest.mark.parametrize('name,sm', [
    ('NVIDIA RTX 2050', (8, 6)), ('Tesla T4', (8, 0)),
    ('NVIDIA A100-SXM4-80GB', (8, 0)), ('H200', (9, 0)),
    ('Unknown sm75', (7, 5)),
])
def test_other_gpu_uses_original_swqr(name, sm):
    assert select_swqr_backend(1, 128, 'cuda:0', device_name=name,
                               capability=sm) == ('swqr', None)


def test_cpu_and_one_token_dispatch():
    assert select_swqr_backend(1, 128, 'cpu') == ('thunder', None)
    assert select_swqr_backend(1, 1, 'cpu') == ('direct', None)
    assert select_swqr_backend(1, 1, 'cuda:0', device_name='Tesla T4',
                               capability=(7, 5)) == ('direct', None)


@pytest.mark.parametrize('b,t', [(0, 128), (1, 0), (-1, 512)])
def test_invalid_shape(b,t):
    with pytest.raises(ValueError):
        select_swqr_backend(b,t, 'cpu')


def test_default_backend_preserves_checkpoint_keys_and_cpu_parity():
    torch.manual_seed(11)
    model = make_esa('fast', dim=48, heads=6, scan='auto')
    ref = make_esa('fast', dim=48, heads=6, scan='thunder')
    ref.load_state_dict(model.state_dict(),strict=True)
    model.eval();ref.eval()
    x = torch.randn(2, 33, 48)
    y, s = model(x)
    yr, sr = ref(x)
    torch.testing.assert_close(y, yr, atol=0, rtol=0)
    torch.testing.assert_close(s, sr, atol=0, rtol=0)
    assert set(ref.state_dict()) == set(model.state_dict())


def test_explicit_adaptive_cpu_rejects_multitoken():
    a=torch.ones(1, 128, 1, 2)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        _scan.swqr_adaptive_scan(a, a)


def test_explicit_legacy_cutoff_validation():
    a=torch.ones(1, 128, 1, 2)
    with pytest.raises(ValueError,match='cutoff'):
        _scan.swqr_adaptive_scan(a, a, cutoff=2049)
    with pytest.raises(ValueError,match='warps'):
        _scan.swqr_adaptive_scan(a, a, warps=3)


def test_direct_recurrence_always_uses_single_token_path_on_cpu():
    a=torch.full((2,1,2,3),.9)
    u=torch.randn_like(a)*.1
    s=torch.randn(2,2,3)*.1
    got=_scan.swqr_adaptive_scan(a,u,s)
    expected=a*s[:,None]+u
    torch.testing.assert_close(got,expected)
