"""
ensemble.py
-----------
A minimal, picklable wrapper that combines the predict_proba outputs of
several fitted classifiers via a weighted average ("soft voting" is just
the special case of equal weights). Kept in its own module (rather than as
a local class/lambda inside train.py) so joblib/pickle can locate the class
by import path when predict.py loads model.pkl.
"""

import numpy as np


class EnsembleModel:
    def __init__(self, models, weights=None, names=None):
        self.models = models
        self.weights = np.array(weights, dtype=float) if weights is not None \
            else np.ones(len(models)) / len(models)
        self.weights = self.weights / self.weights.sum()
        self.names = names or [f"model_{i}" for i in range(len(models))]

    def predict_proba(self, X):
        probs = np.zeros(X.shape[0])
        for w, m in zip(self.weights, self.models):
            probs += w * m.predict_proba(X)[:, 1]
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)