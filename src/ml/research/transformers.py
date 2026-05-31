"""
src/ml/research/transformers.py

Custom sklearn transformers used inside challenger pipelines. These live in
their own module (NOT in train_candidate.py) on purpose: train_candidate.py is
run via `python -m`, which executes it as `__main__`, and a class defined there
would pickle as `__main__.ColumnMask` — unloadable from any other process
(e.g. paper_trader's shadow scan). Defining it here means the pickled pipeline
always references `src.ml.research.transformers.ColumnMask`, which is importable
anywhere.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnMask(BaseEstimator, TransformerMixin):
    """
    Select a fixed subset of columns (by index into AUGMENTED_FEATURES) from the
    full 15-feature matrix. This lets a feature-subset challenger still ACCEPT
    the same 15-column input the shadow scan feeds — unused columns are dropped
    internally rather than at the call site. Only added to a pipeline when a
    real subset is requested (the full-15 case skips it and stays pure-sklearn).
    """

    def __init__(self, keep_idx: List[int]):
        self.keep_idx = list(keep_idx)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_idx]
