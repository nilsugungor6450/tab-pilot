"""TabPFN classifier used as a driving policy (cached context, one estimator for real-time use)."""
import numpy as np
import pandas as pd

from sim import ACTION_NAMES


class TabPFNPolicy:
    def __init__(self, max_context=2000, n_estimators=1):
        self.max_context = max_context
        self.n_estimators = n_estimators
        self.model = None
        self.n_context = 0

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if len(X) > self.max_context:
            idx = np.linspace(0, len(X) - 1, self.max_context).astype(int)
            X, y = X[idx], y[idx]
        self.model = TabPFNClassifier(fit_mode="fit_with_cache", n_estimators=self.n_estimators)
        self.model.fit(X, y)
        self.n_context = len(X)
        return self

    def fit_csv(self, path):
        df = pd.read_csv(path)
        return self.fit(df.drop(columns="action").to_numpy(), df["action"].to_numpy())

    def probs(self, sensors):
        S = np.atleast_2d(np.asarray(sensors, dtype=float))
        p = self.model.predict_proba(S)
        out = np.zeros((len(S), len(ACTION_NAMES)))
        for j, c in enumerate(self.model.classes_):
            out[:, ACTION_NAMES.index(c)] = p[:, j]
        return out

    def act(self, sensors):
        p = self.probs(sensors)
        return p.argmax(axis=1), p
