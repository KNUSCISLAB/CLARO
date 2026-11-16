"""Alarm detector: score, prefix-only calibration, threshold and alarm emission.

The detector score is the maximum over channels and window lengths of the window
residuals. Its orientation, min-max scale and threshold are fitted on the validation
prefix only and are never refitted on the evaluation suffix.
"""
import numpy as np
from sklearn.metrics import f1_score

from . import config


def channel_quantiles(values, labels, valid, boundary):
    """Per-channel normal-time quantile of each window length on the prefix.

    Used only by the optional calibrated detector and calibrated ranker. ``values``
    maps window length to its (T x F) window residuals.
    """
    cal = np.arange(boundary - config.OFFSET)
    cal = cal[valid[cal] & (labels[cal + config.OFFSET] == 0)]
    step = max(1, len(cal) // 60000)
    return {w: np.quantile(v[cal[::step]], config.CALIBRATED_QUANTILE, axis=0) + 1e-9 for w, v in values.items()}


def detector_score(values, scales, quantiles=None):
    """Maximum window residual over channels and window lengths.

    With ``quantiles`` (calibrated detector), each residual is first divided by its
    per-channel prefix quantile.
    """
    if quantiles is None:
        return np.max([values[w].max(1) for w in scales], axis=0)
    return np.max([(values[w] / quantiles[w]).max(1) for w in scales], axis=0)


def fit_transform(scores, labels):
    """Orientation and min-max scale of the detector score on calibration data."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError('Calibration requires finite scores and both label classes')
    sign = -1. if x[y].mean() < x[~y].mean() else 1.
    oriented = sign * x
    return dict(sign=sign, minimum=float(oriented.min()), span=float(max(np.ptp(oriented), 1e-9)))


def transform(scores, state):
    """Apply a fitted transform; values are not clipped to the calibration range."""
    return (state['sign'] * np.asarray(scores) - state['minimum']) / state['span']


def fit_threshold(scores, labels):
    """Threshold maximizing pointwise F1 on calibration data (first maximum wins)."""
    candidates = np.unique(np.r_[np.nextafter(np.min(scores), -np.inf),
                                 np.quantile(scores, np.linspace(0, 1, 201))])
    values = [f1_score(labels, scores > t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(values))])


def constrained_threshold(scores, labels, normal_limit):
    """Maximum pointwise F1 among thresholds whose normal-time alarm fraction <= limit."""
    normal = ~labels.astype(bool)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 401)))
    feasible = [t for t in candidates if (scores[normal] > t).mean() <= normal_limit]
    values = [f1_score(labels, scores > t, zero_division=0) for t in feasible]
    return float(feasible[int(np.argmax(values))])


def calibrate(score, labels, valid, boundary, fa_limit=None):
    """Fit transform and threshold on the prefix ``[0, boundary)`` of the raw timeline.

    Returns the transformed score for the whole stream and the threshold.
    """
    cal_end = boundary - config.OFFSET
    mask = valid[:cal_end]
    cal_labels = labels[config.OFFSET:boundary][mask]
    state = fit_transform(score[:cal_end][mask], cal_labels)
    transformed = transform(score, state)
    if fa_limit:
        threshold = constrained_threshold(transformed[:cal_end][mask], cal_labels, fa_limit)
    else:
        threshold = fit_threshold(transformed[:cal_end][mask], cal_labels)
    return transformed, threshold


def emit(scores, threshold, valid, period=config.SNAPSHOT_PERIOD):
    """Ranking times: each alarm onset and every ``period`` samples while alarming.

    Returns indices into ``scores``. An invalid row clears the alarm state, so the
    next alarm is a new onset.
    """
    out = []
    was = False
    last = -period
    for i, s in enumerate(scores):
        if not valid[i]:
            was = False
            continue
        alarm = s > threshold
        if alarm and (not was or i - last >= period):
            out.append(i)
            last = i
        was = alarm
    return np.array(out, dtype=int)
