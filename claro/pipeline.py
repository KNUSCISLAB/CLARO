"""CLARO online pipeline and validation-prefix module selection.

A configuration fixes four optional modules:
  sem       'ar' (AR-SEM) or 'ols' (lag-2 SEM without the channel's own lag)
  scales    detector window lengths, (45,) or (5, 45)
  detector  'raw' (maximum window residual) or 'calibrated' (residuals divided by
            per-channel prefix quantiles, threshold with <= 2% prefix normal alarms)
  ranker    'score' (attribution score) or 'calibrated' (calibrated 45-s residual)
The base configuration is ('ols', (45,), 'raw', 'score'). All 16 combinations are scored
on the validation prefix; the suffix is scored only for the frozen selection.
"""
import itertools

import numpy as np

from . import config, data
from .arsem import LaggedSEM, window_means
from .detector import calibrate, channel_quantiles, detector_score, emit
from .evaluation import build_episodes, load_evidence, normal_fraction, score_episodes
from .ranking import CausalRanking, candidate_indices, rank_by_scores

BASE = dict(sem='ols', scales=(45,), detector='raw', ranker='score')
WINDOWS = (5, 45)


def grid():
    """All 16 module combinations, in the order used for reporting."""
    for sem, scales, detector, ranker in itertools.product(('ols', 'ar'), ((45,), (5, 45)), ('raw', 'calibrated'),
                                                           ('score', 'calibrated')):
        yield dict(sem=sem, scales=scales, detector=detector, ranker=ranker)


def changes(cfg):
    """Number of modules that differ from the base configuration."""
    return sum(cfg[k] != BASE[k] for k in ('sem', 'scales', 'detector', 'ranker'))


class Stream:
    """Window residuals of the test stream for one SEM on one graph."""

    def __init__(self, sem='ar', graph=None):
        adj = np.load(config.LLM_PRUNED_GRAPH if graph is None else graph)
        model = LaggedSEM(adj, autoregressive=(sem == 'ar'), lag=config.LAG).fit(data.load_train())
        errors = model.point_errors(data.load_test())
        self.values = {w: window_means(errors, config.LAG, w, config.OFFSET) for w in WINDOWS}
        self.names = data.channel_names()
        self.labels = data.load_labels()
        self.valid = data.valid_mask(len(self.values[config.RANK_WINDOW]))
        evidence = load_evidence()
        self.eligible = candidate_indices(self.names, evidence)
        self.entries = build_episodes(self.names, self.eligible, evidence)
        self.ranker = CausalRanking(adj, self.eligible)

    def boundary(self, fraction=config.VALIDATION_FRACTION):
        """Raw index where the validation prefix ends."""
        return int(len(self.labels) * fraction)

    def run(self, cfg, boundary):
        """Frozen detector and alarm-time rankings; calibration uses raw samples [0, boundary).

        Returns ranking endpoints (window rows), top-5 rankings, transformed detector
        score and threshold.
        """
        q = channel_quantiles(self.values, self.labels, self.valid, boundary)
        calibrated = cfg['detector'] == 'calibrated'
        score = detector_score(self.values, cfg['scales'], q if calibrated else None)
        transformed, threshold = calibrate(score, self.labels, self.valid, boundary,
                                           config.FA_LIMIT if calibrated else None)
        ends = emit(transformed, threshold, self.valid)
        r45 = self.values[config.RANK_WINDOW][ends]
        if cfg['ranker'] == 'score':
            rankings = [self.ranker.rank(v) for v in r45]
        else:
            rankings = [rank_by_scores(v, self.eligible) for v in r45 / q[config.RANK_WINDOW]]
        return ends, rankings, transformed, threshold

    def prefix(self, cfg, boundary):
        """Prefix J@k and normal-time alarm fraction used for selection."""
        ends, rankings, transformed, threshold = self.run(cfg, boundary)
        summary = score_episodes(ends, rankings, self.entries, config.OFFSET, boundary)
        summary['normal_alarm_fraction'] = normal_fraction(transformed, threshold, self.labels, self.valid,
                                                           config.OFFSET, boundary)
        return summary


def selection_key(summary, cfg):
    """Larger is better: J@1, J@5, fewer prefix normal-time alarms, fewer changes."""
    return (summary['hit1_successes'], summary['hit5_successes'],
            -round(summary['normal_alarm_fraction'], 3), -changes(cfg))
