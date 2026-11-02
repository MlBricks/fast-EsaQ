"""fastEsa — independently packaged PROGRESS Learned and Hybrid ESA variants.

The kernels come verbatim from the user-supplied PROGRESS Kaggle notebooks.
`fast` means PROGRESS learned. `quality` means PROGRESS hybrid.
These are variant names, NOT verified benchmark claims.
"""
from ._scan import prepare_swqr, prepare_swqr_short, select_swqr_backend
from .api import (FastESA, QualityESA, FastESAModel, QualityESAModel,
                  make_esa, make_model, forward_tokens, load_checkpoint,
                  normalize_variant)
__all__ = ["FastESA", "QualityESA", "FastESAModel", "QualityESAModel",
           "make_esa", "make_model", "forward_tokens", "load_checkpoint",
           "normalize_variant", "prepare_swqr", "prepare_swqr_short", "select_swqr_backend"]
__version__ = "0.3.2"
