"""Step 2: dependency graph, rule pruning and LLM screening.

  1. Granger edges on the first 5,000 normal training rows (``--granger``; slow).
     Without the flag the shipped data/graphs/granger_adj.npy is used.
  2. Process-prior edges expanded from <CLARO_RAW>/graph/boiler/phy_boiler.json when
     available, otherwise the shipped data/graphs/prior_edges.json.
  3. Union graph -> union_adj.npy; rule pruning -> rule_pruned_adj.npy.
  4. LLM screening of the remaining cross-subsystem edges -> llm_pruned_adj.npy. By
     default the stored LLM decision (results/graph_screening.json) is replayed;
     ``--query-llm`` asks the local Ollama server again (the original request had no
     seed, so a new response may differ).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claro import config, data, graph  # noqa: E402

SCREENING = config.RESULTS_DIR / 'graph_screening.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--granger', action='store_true', help='recompute the Granger graph from X_train.npy')
    parser.add_argument('--query-llm', action='store_true', help='query the LLM instead of replaying its stored decision')
    parser.add_argument('--out', type=Path, default=config.GRAPH_DIR, help='output directory for the graphs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = data.channel_names()

    if args.granger:
        granger = graph.granger_graph(data.load_train(), names,
                                      progress=lambda i, n, e: print(f'  granger {i}/{n} sources, {e} edges', flush=True))
        np.save(args.out / 'granger_adj.npy', granger)
    else:
        granger = np.load(config.GRANGER_GRAPH)

    phy = config.RAW_DIR / 'graph' / 'boiler' / 'phy_boiler.json'
    if phy.exists():
        edges = graph.prior_edges(json.loads(phy.read_text()))
        (args.out / 'prior_edges.json').write_text(json.dumps({'sensor_edges': edges}, indent=1) + '\n')
    else:
        edges = [tuple(e) for e in json.loads(config.PRIOR_EDGES.read_text())['sensor_edges']]

    union = graph.union_graph(granger, edges, names)
    keep, removed, screened = graph.classify_edges(union, names)
    rule_pruned = graph.rule_prune(union, removed)
    retained = set(names)
    n_prior = sum(s in retained and t in retained for s, t in edges)
    print(f'granger {int(granger.sum())}, prior {n_prior} (on retained channels), union {int(union.sum())}')
    print(f'intra-subsystem {len(keep)}, rule-removed {len(removed)}, screened by the LLM {len(screened)}')

    if args.query_llm:
        raw = graph.query_screening(screened)
        record = {'model': config.LLM_MODEL, 'llm_raw': raw, 'llm_parsed': graph.parse_screening(raw)}
        SCREENING.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    record = json.loads(SCREENING.read_text())
    llm_pruned = graph.apply_screening(rule_pruned, screened, record['llm_parsed'])
    print(f'rule-pruned {int(rule_pruned.sum())}, LLM-pruned {int(llm_pruned.sum())}')

    np.save(args.out / 'union_adj.npy', union)
    np.save(args.out / 'rule_pruned_adj.npy', rule_pruned)
    np.save(args.out / 'llm_pruned_adj.npy', llm_pruned)


if __name__ == '__main__':
    main()
