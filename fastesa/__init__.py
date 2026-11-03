"""fastEsa — PROGRESS learned-bounds and half-life hybrid ESA variants.

Quality ESA supports optional CUDA SWQR scan, fused half-life decode and
CUDA Graph generation. Original mathematical recurrences stay unchanged.
"""
from ._quality_decode import prepare_quality_decode
from .cuda_graph import CUDAGreedyGraph
from ._scan import prepare_swqr, prepare_swqr_short, select_swqr_backend
from .api import (FastESA, QualityESA, FastESAModel, QualityESAModel,
                  make_esa, make_model, forward_tokens, load_checkpoint,
                  normalize_variant)
__all__ = ["FastESA", "QualityESA", "FastESAModel", "QualityESAModel",
           "make_esa", "make_model", "forward_tokens", "load_checkpoint",
           "normalize_variant", "prepare_quality_decode", "CUDAGreedyGraph", "prepare_swqr", "prepare_swqr_short", "select_swqr_backend"]
__version__ = "0.3.8"
