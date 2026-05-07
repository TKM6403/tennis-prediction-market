"""
calibration_methods.py

Custom calibration wrappers used in the training pipeline.

Lives in its own module (rather than train.py) so pickled models can
unpickle cleanly from any entry point. If a class is defined in a module
that runs as `__main__`, its pickle is bound to `__main__.<ClassName>`,
which fails to load from a separate process — which is exactly what
paper_trader.py and weekly_report.py do.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from betacal import BetaCalibration


class BetaCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrap a base classifier with Beta calibration (Kull, Silva Filho & Flach 2017).

    sklearn's CalibratedClassifierCV supports only 'sigmoid' (Platt) and
    'isotonic'. Beta calibration is parametric like Platt but its function
    family includes the identity, so an already-calibrated model isn't
    distorted. Reference choice for small-to-medium-data calibration.

    Implementation:
        1. cross_val_predict to get out-of-fold base probabilities (no leak).
        2. Fit BetaCalibration on (oof_probs, y).
        3. Refit base estimator on full data for inference.
        4. predict_proba: base.predict_proba → calibrator.predict.
    """

    def __init__(self, base_estimator, cv: int = 5):
        self.base_estimator = base_estimator
        self.cv = cv

    def fit(self, X, y):
        oof = cross_val_predict(
            clone(self.base_estimator), X, y, cv=self.cv, method="predict_proba"
        )[:, 1]
        self.calibrator_ = BetaCalibration(parameters="abm")
        self.calibrator_.fit(oof, y)
        self.base_estimator_ = clone(self.base_estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        raw = self.base_estimator_.predict_proba(X)[:, 1]
        cal = np.asarray(self.calibrator_.predict(raw)).ravel()
        cal = np.clip(cal, 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
