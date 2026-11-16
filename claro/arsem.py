"""Lagged structural equation models on the dependency graph and window residuals.

``LaggedSEM(adj, autoregressive=True)`` is CLARO's AR-SEM: each standardized channel
is regressed by OLS on its own lag-1 value and on its graph parents at lag 2.
``autoregressive=False`` is the base lag-2 SEM (parents only), which is one of the
module options compared during validation-prefix selection.
"""
import numpy as np
from sklearn.linear_model import LinearRegression


class LaggedSEM:
    """Per-channel OLS on lagged parents, with standardized absolute residuals."""

    def __init__(self, adj, autoregressive=True, lag=2):
        self.adj = adj
        self.autoregressive = autoregressive
        self.lag = lag
        self.F = adj.shape[0]
        self.models = {}
        self.residual_std = np.ones(self.F)

    def _design(self, z, i, parents):
        return np.c_[z[self.lag - 1:-1, [i]], z[:-self.lag][:, parents]]

    def fit(self, x):
        """Fit on normal training data ``x`` (N x F, raw units)."""
        if self.autoregressive:
            x = np.asarray(x, dtype=float)
        # The base SEM is fitted in the input precision (float32 training array).
        self.mu = x.mean(0)
        self.std = x.std(0) + 1e-8
        z = (x - self.mu) / self.std
        for i in range(self.F):
            parents = np.flatnonzero(self.adj[:, i] > 0)
            if self.autoregressive:
                design = self._design(z, i, parents)
                model = LinearRegression().fit(design, z[self.lag:, i])
                self.models[i] = (model, parents)
                self.residual_std[i] = np.std(z[self.lag:, i] - model.predict(design)) + 1e-8
            elif len(parents) == 0:
                # Without parents the residual is the channel's own z-score.
                self.residual_std[i] = z[self.lag:, i].std() + 1e-8
            else:
                design, target = z[:-self.lag][:, parents], z[self.lag:, i]
                model = LinearRegression(fit_intercept=True).fit(design, target)
                self.models[i] = (model, parents)
                self.residual_std[i] = (target - model.predict(design)).std() + 1e-8
        return self

    def point_errors(self, x):
        """Standardized absolute residuals; row k ends at raw sample k + lag."""
        z = (np.asarray(x, dtype=float) - self.mu) / self.std
        out = np.empty((len(z) - self.lag, self.F))
        if self.autoregressive:
            for i, (model, parents) in self.models.items():
                out[:, i] = np.abs(z[self.lag:, i] - model.predict(self._design(z, i, parents))) / self.residual_std[i]
            return out
        for i in range(self.F):
            if i in self.models:
                model, parents = self.models[i]
                prediction = z[:-self.lag, parents] @ model.coef_ + model.intercept_
                out[:, i] = np.abs(z[self.lag:, i] - prediction) / self.residual_std[i]
            else:
                out[:, i] = np.abs(z[self.lag:, i]) / self.residual_std[i]
        return out


def window_means(errors, lag, window, first_end):
    """Mean of the ``window - lag`` point residuals ending at each raw index >= ``first_end``.

    All window lengths are aligned on the same raw endpoint, so row ``k`` of the
    result ends at raw test sample ``first_end + k``.
    """
    width = window - lag
    sums = np.concatenate((np.zeros((1, errors.shape[1])), np.cumsum(errors, axis=0)))
    ends = np.arange(first_end, len(errors) + lag) - lag + 1   # exclusive point index
    return (sums[ends] - sums[ends - width]) / width
