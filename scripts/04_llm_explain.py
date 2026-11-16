"""Step 4: LLM re-ranking of CLARO's top-10 candidates and checked explanations.

Only alarm snapshots that can affect J@k (first 60 s of fully covered episodes) are
sent to the LLM. Re-ranking variants (none, LLM, rank fusion) are compared on the
prefix with seed 11 and the suffix is scored with seeds 11-13. Explanations are
requested under three prompt contexts and their claims are checked against the HAI
documentation.

By default every response is read from results/llm_cache.json and a cache miss is an
error; ``--query-llm`` sends missing requests to the local Ollama server.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claro import config, llm  # noqa: E402
from claro.evaluation import score_episodes  # noqa: E402
from claro.pipeline import Stream  # noqa: E402


def brief(summary):
    return {k: v for k, v in summary.items() if k != 'episodes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--query-llm', action='store_true', help='query Ollama on cache misses')
    args = parser.parse_args()

    cfg = json.loads((config.RESULTS_DIR / 'selected_configuration.json').read_text())['selected']
    cfg['scales'] = tuple(cfg['scales'])
    stream = Stream(cfg['sem'])
    descriptions = json.loads((config.DOC_DIR / 'channel_descriptions.json').read_text())
    pairs = llm.documented_pairs(json.loads((config.DOC_DIR / 'documented_relations.json').read_text()))
    client = llm.CachedLLM(offline=not args.query_llm)

    n = len(stream.labels)
    b = stream.boundary()
    ends, base, _, _ = stream.run(cfg, b)
    prefix_idx = llm.relevant(stream.entries, ends, config.OFFSET, b)
    suffix_idx = llm.relevant(stream.entries, ends, b + config.OFFSET, n)
    print('LLM snapshots: prefix', len(prefix_idx), 'suffix', len(suffix_idx), flush=True)

    # Re-ranking: select the variant on the prefix, then score all variants on the suffix.
    validation = {}
    for v in llm.RERANK_VARIANTS:
        r = llm.rerank(stream, ends, base, prefix_idx, v, config.LLM_SEED, client, descriptions, [])
        validation[v] = brief(score_episodes(ends, r, stream.entries, config.OFFSET, b))
        print('prefix', v, validation[v]['hit1_successes'], validation[v]['hit5_successes'], validation[v]['N'], flush=True)
    selected = max(llm.RERANK_VARIANTS, key=lambda v: (validation[v]['hit1_successes'], validation[v]['hit5_successes'],
                                                        -llm.RERANK_VARIANTS.index(v)))
    log = []
    suffix = {}
    for v in llm.RERANK_VARIANTS:
        for seed in ((config.LLM_SEED,) if v == 'claro' else config.LLM_SEEDS):
            r = llm.rerank(stream, ends, base, suffix_idx, v, seed, client, descriptions, log)
            suffix[f'{v}/seed{seed}'] = score_episodes(ends, r, stream.entries, b + config.OFFSET, n)
            s = suffix[f'{v}/seed{seed}']
            print('suffix', v, seed, s['hit1_successes'], s['hit5_successes'], s['N'], flush=True)
    rerank = {'selected': selected, 'model': config.LLM_MODEL, 'system_prompt': llm.RERANK_SYSTEM,
              'validation': validation, 'suffix': suffix, 'log': log}
    (config.RESULTS_DIR / 'llm_rerank.json').write_text(json.dumps(rerank, indent=2) + '\n')

    # Explanations and their documentation check.
    indices = prefix_idx + suffix_idx
    conditions, all_items = {}, []
    for condition in llm.EXPLANATION_CONDITIONS:
        summary, items, rankings = llm.explain(stream, ends, base, indices, condition, client, descriptions, pairs)
        val = score_episodes(ends, rankings, stream.entries, config.OFFSET, b)
        suf = score_episodes(ends, rankings, stream.entries, b + config.OFFSET, n)
        summary.update(validation_J1=val['hit1_successes'], validation_J5=val['hit5_successes'],
                       suffix_J1=suf['hit1_successes'], suffix_J5=suf['hit5_successes'])
        conditions[condition] = summary
        all_items += items
        print(condition, summary, flush=True)
    (config.RESULTS_DIR / 'explanation_check.json').write_text(
        json.dumps({'conditions': conditions, 'items': all_items}, indent=2) + '\n')
    explanations = [{k: it[k] for k in ('snapshot', 'raw_endpoint', 'channel', 'reason')}
                    for it in all_items if it['condition'] == 'manual']
    (config.RESULTS_DIR / 'llm_explanations.json').write_text(json.dumps(explanations, indent=2) + '\n')


if __name__ == '__main__':
    main()
