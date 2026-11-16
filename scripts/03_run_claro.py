"""Step 3: select CLARO's optional modules on the validation prefix, then evaluate on the suffix.

``--step select`` scores all 16 module combinations on the prefix (first 40% of the
test timeline) and writes results/selected_configuration.json. ``--step evaluate``
reads that file and scores the frozen configuration once on the suffix, writing
results/evaluation.json and results/alarm_delays.json (Figure, panel a).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claro import config  # noqa: E402
from claro.evaluation import alarm_delays, score_episodes, suffix_detection  # noqa: E402
from claro.pipeline import BASE, Stream, grid, selection_key  # noqa: E402

SELECTED = config.RESULTS_DIR / 'selected_configuration.json'


def brief(summary):
    return {k: v for k, v in summary.items() if k != 'episodes'}


def select(streams):
    rows = []
    for cfg in grid():
        stream = streams[cfg['sem']]
        summary = stream.prefix(cfg, stream.boundary())
        rows.append({'config': cfg, 'prefix': brief(summary), 'prefix_episodes': summary['episodes']})
        print('prefix', cfg, summary['hit1_successes'], summary['hit5_successes'], summary['N'],
              f"FA {summary['normal_alarm_fraction']:.4f}", flush=True)
    best = max(rows, key=lambda r: selection_key(r['prefix'], r['config']))
    record = {'selected': best['config'],
              'criterion': 'prefix J@1, then J@5, then lower prefix normal-time alarm fraction (3 decimals), '
                           'then fewer changes from the base configuration',
              'base': BASE, 'validation_grid': rows}
    SELECTED.write_text(json.dumps(record, indent=2) + '\n')
    print('selected', best['config'])


def evaluate(streams):
    cfg = json.loads(SELECTED.read_text())['selected']
    cfg['scales'] = tuple(cfg['scales'])
    out = {'configuration': cfg}
    delays = {}
    for name, c in (('claro', cfg), ('base', BASE)):
        stream = streams[c['sem']]
        b = stream.boundary()
        n = len(stream.labels)
        ends, rankings, transformed, threshold = stream.run(c, b)
        delays[name] = alarm_delays(transformed, threshold, stream.labels, b)
        if name != 'claro':
            continue
        prefix = score_episodes(ends, rankings, stream.entries, config.OFFSET, b)
        suffix = score_episodes(ends, rankings, stream.entries, b + config.OFFSET, n)
        out.update({'threshold': threshold, 'prefix': prefix, 'suffix': suffix,
                    'suffix_no_alarm_within_60s': sum(not r['alarm_within_deadline'] for r in suffix['episodes']),
                    'suffix_detection': suffix_detection(transformed, threshold, stream.labels, b),
                    'rankings_suffix': [{'raw_endpoint': int(e + config.OFFSET), 'top5': [stream.names[i] for i in r]}
                                        for e, r in zip(ends, rankings) if e + config.OFFSET >= b + config.OFFSET]})
        det = out['suffix_detection']
        print(f"suffix J@1 {suffix['hit1_successes']}/{suffix['N']}, J@5 {suffix['hit5_successes']}/{suffix['N']}, "
              f"attacks alarmed within 60 s {det['attacks_alarmed_60s']}/{det['attacks']}, "
              f"normal-time alarm fraction {det['normal_alarm_fraction']:.4f}")
    (config.RESULTS_DIR / 'evaluation.json').write_text(json.dumps(out, indent=2) + '\n')
    (config.RESULTS_DIR / 'alarm_delays.json').write_text(json.dumps(delays, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['select', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    streams = {sem: Stream(sem) for sem in ('ols', 'ar')}
    if args.step in ('select', 'all'):
        select(streams)
    if args.step in ('evaluate', 'all'):
        evaluate(streams)


if __name__ == '__main__':
    main()
