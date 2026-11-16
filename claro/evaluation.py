"""Evaluation: attack episodes, source-listed targets, joint J@k and detection metrics.

Labels and targets are used only to score outputs that were already produced; the
online stream never receives them. Time conventions: window row ``i`` ends at raw
test sample ``i + OFFSET`` (44-step purge between the prefix and the suffix).
"""
import csv
import json
import re

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from . import config


def segments(labels):
    """[start, stop) intervals of consecutive positive labels."""
    changes = np.diff(np.r_[0, np.asarray(labels, dtype=int), 0])
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def metrics(scores, labels, threshold):
    """Detection metrics of a thresholded score against point labels."""
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) > threshold
    adjusted = pred.copy()
    episodes = segments(labels)
    for start, stop in episodes:
        if pred[start:stop].any():
            adjusted[start:stop] = True
    tp = sum(pred[a:b].any() for a, b in episodes)
    # Attack segments with an alarm within 60 samples (1 Hz) of onset.
    timely = sum(pred[a:min(b, a + 61)].any() for a, b in episodes)
    false_episodes = len(segments(pred & ~labels))
    denom = len(episodes) + tp + false_episodes
    two_classes = len(np.unique(labels)) == 2
    return dict(
        seg_f1=float(2 * tp / denom) if denom else 0.,
        point_f1=float(f1_score(labels, pred, zero_division=0)),
        pa_f1=float(f1_score(labels, adjusted, zero_division=0)),
        bal_acc=float(balanced_accuracy_score(labels, pred)),
        pa_bal_acc=float(balanced_accuracy_score(labels, adjusted)),
        auroc=float(roc_auc_score(labels, scores)) if two_classes else None,
        auprc=float(average_precision_score(labels, scores)) if two_classes else None,
        det60=float(timely / len(episodes)) if episodes else None,
        normal_alarm=float(pred[~labels].mean()) if (~labels).any() else None,
        attack_segments=len(episodes), threshold=float(threshold),
    )


# ---------------------------------------------------------------- episodes and targets

def canonical(name):
    """Formatting-insensitive point name (upper case, alphanumerics only)."""
    return re.sub(r'[^A-Z0-9]', '', name.split('\\')[-1].upper())


def load_evidence():
    """Documented source-point aliases and episode target repairs."""
    return json.loads((config.GROUND_TRUTH_DIR / 'channel_alias_evidence.json').read_text())


def repaired_points(attack_id, points, evidence):
    """Apply the documented corrections of truncated target points in the attack table."""
    result = list(points)
    for row in evidence.get('episode_target_repairs', []):
        if row['attack_id'] == attack_id:
            result = [row['replace'].get(p, p) for p in result]
            result += [p for p in row['append'] if p not in result]
    return result


def target_indices(points, names, eligible, evidence):
    """Map source-listed target points to candidate channels.

    Only explicit (verified) aliases are applied. Returns the matched channel indices
    and the points that could not be matched to a single candidate channel.
    """
    aliases = {canonical(r['source_point']): r['channel'] for r in evidence['verified']}
    indices, unresolved = [], []
    for point in points:
        name = aliases.get(canonical(point), point)
        match = [i for i, n in enumerate(names) if canonical(n) == canonical(name)]
        if len(match) == 1 and match[0] in eligible:
            indices.append(match[0])
        else:
            unresolved.append(point)
    return sorted(set(indices)), unresolved


def build_episodes(names, eligible, evidence):
    """Attack episodes with their target channels and coverage flag.

    An episode is fully covered when every source-listed target point maps to a
    candidate channel; only fully covered episodes enter J@k.
    """
    with (config.GROUND_TRUTH_DIR / 'hai23_target_points.csv').open() as f:
        catalog = {r['attack_id']: r['target_points'].split(';') for r in csv.DictReader(f)}
    episodes = json.loads((config.GROUND_TRUTH_DIR / 'attack_episodes.json').read_text())
    entries = []
    for row in episodes:
        points = repaired_points(row['attack_id'], catalog[row['attack_id']], evidence)
        targets, missing = target_indices(points, names, eligible, evidence)
        entries.append(dict(row, targets=targets, fully_covered=not missing,
                            source_target_points=points, unresolved_points=missing))
    return entries


# ---------------------------------------------------------------- joint localization

def score_episodes(endpoints, rankings, entries, range_start, range_stop, deadline=config.DEADLINE):
    """Joint J@k: an alarm ranking within ``deadline`` s of onset has a target in its top k.

    Considers fully covered episodes inside raw interval [range_start, range_stop).
    ``endpoints`` are window-row indices of the rankings.
    """
    times = endpoints + config.OFFSET
    rows = []
    for ep in entries:
        if not (ep['fully_covered'] and ep['raw_start'] >= range_start and ep['raw_stop'] <= range_stop):
            continue
        targets = set(ep['targets'])
        inside = np.flatnonzero((times >= ep['raw_start']) & (times < ep['raw_stop']) & (times - ep['raw_start'] <= deadline))
        row = {'attack_id': ep['attack_id'], 'raw_start': ep['raw_start'], 'n_targets': len(targets),
               'alarm_within_deadline': bool(len(inside)), 'snapshots_within_deadline': int(len(inside))}
        for k in (1, 5):
            row[f'hit{k}'] = int(any(set(rankings[i][:k]) & targets for i in inside))
            row[f'recall{k}'] = max([len(set(rankings[i][:k]) & targets) / len(targets) for i in inside], default=0.)
        ranks = [min([p + 1 for p, c in enumerate(rankings[i][:5]) if c in targets], default=None) for i in inside]
        ranks = [r for r in ranks if r is not None]
        row['best_rank_top5'] = min(ranks) if ranks else None
        rows.append(row)
    summary = {'N': len(rows)}
    for key in ('hit1', 'hit5'):
        summary[key + '_successes'] = int(sum(r[key] for r in rows))
    for key in ('recall1', 'recall5'):
        summary[key] = float(np.mean([r[key] for r in rows])) if rows else None
    summary['alarm_within_deadline'] = int(sum(r['alarm_within_deadline'] for r in rows))
    summary['episodes'] = rows
    return summary


def normal_fraction(transformed, threshold, labels, valid, lo, hi):
    """Fraction of normal valid samples in raw interval [lo, hi) that are alarmed."""
    seg = slice(lo - config.OFFSET, hi - config.OFFSET)
    normal = (labels[lo:hi] == 0) & valid[seg]
    return float((transformed[seg] > threshold)[normal].mean()) if normal.any() else 0.


def suffix_detection(transformed, threshold, labels, boundary):
    """Detection, timeliness and false-alarm summary on the suffix [boundary + OFFSET, end)."""
    suffix_labels = labels[boundary + config.OFFSET:]
    score = transformed[boundary:]
    pred = score > threshold
    normal = suffix_labels == 0
    runs = segments(pred & normal)
    lengths = [e - s for s, e in runs] or [0]
    attacks = segments(suffix_labels)
    timely = sum(pred[s:min(e, s + 61)].any() for s, e in attacks)
    return {'normal_alarm_fraction': float(pred[normal].mean()),
            'false_runs_per_hour': len(runs) / (len(suffix_labels) / 3600),
            'false_run_median_s': float(np.median(lengths)),
            'attacks_alarmed_60s': int(timely), 'attacks': len(attacks),
            'detection': metrics(score, suffix_labels, threshold)}


def alarm_delays(transformed, threshold, labels, boundary):
    """Onset-to-first-alarm delay (s) of every suffix attack (None: never alarmed)."""
    suffix_labels = labels[boundary + config.OFFSET:]
    alarm = transformed[boundary:] > threshold
    rows = []
    for start, stop in segments(suffix_labels):
        hit = np.flatnonzero(alarm[start:stop])
        rows.append({'raw_start': int(start + boundary + config.OFFSET), 'duration': int(stop - start),
                     'delay': int(hit[0]) if len(hit) else None})
    return rows
