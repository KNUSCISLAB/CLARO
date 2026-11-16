"""Stages 1-2: dependency graph (Granger + process prior), rule pruning and LLM screening.

Adjacency convention: ``adj[j, i] = 1`` is an edge j -> i (j is a parent of i).
"""
import json
import re
from collections import defaultdict

import numpy as np

from . import config

# HAI boiler components (graph/boiler/phy_boiler.json) expanded to channel names.
COMPONENT_CHANNELS = {
    'TK01': ['P1_LIT01', 'P1_PIT01'], 'TK02': ['P1_TIT02', 'P1_PIT02'], 'TK03': ['P1_LIT01'], 'TK04': [],
    'HEX01T': ['P1_TIT01'], 'HEX01S': ['P1_TIT01'], 'COND01': [], 'HT01': [],
    'PP01A': ['P1_PP01AD', 'P1_PP01AR'], 'PP01B': ['P1_PP01BD', 'P1_PP01BR'], 'PP02': ['P1_PP02D', 'P1_PP02R'],
    'PP04': ['P1_PP04', 'P1_PP04D'],
    'FCV01': ['P1_FCV01D', 'P1_FCV01Z'], 'FCV02': ['P1_FCV02D', 'P1_FCV02Z'], 'FCV03': ['P1_FCV03D', 'P1_FCV03Z'],
    'PCV01': ['P1_PCV01D', 'P1_PCV01Z'], 'PCV02': ['P1_PCV02D', 'P1_PCV02Z'], 'LCV01': ['P1_LCV01D', 'P1_LCV01Z'],
    'FT01': ['P1_FT01', 'P1_FT01Z'], 'FT02': ['P1_FT02', 'P1_FT02Z'], 'FT03': ['P1_FT03', 'P1_FT03Z'],
    'PIT01': ['P1_PIT01'], 'PIT02': ['P1_PIT02'], 'TIT01': ['P1_TIT01'], 'TIT02': ['P1_TIT02'], 'LIT01': ['P1_LIT01'],
}


def granger_graph(x_train, names, rows=config.GRANGER_ROWS, max_lag=config.GRANGER_MAX_LAG,
                  alpha=config.GRANGER_ALPHA, progress=None):
    """Pairwise Granger predictive edges on the first ``rows`` standardized training rows.

    Edge j -> i is kept when the SSR F-test of "j does not Granger-cause i" has
    p < alpha at any lag 1..max_lag (no multiple-testing correction).
    """
    import pandas as pd
    from statsmodels.tsa.stattools import grangercausalitytests
    x_train = np.asarray(x_train)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    frame = pd.DataFrame(((x_train - mean) / std)[:rows], columns=names)
    n = len(names)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                # Column order [effect, cause]: tests whether names[i] helps predict names[j].
                res = grangercausalitytests(frame[[names[j], names[i]]], maxlag=max_lag, verbose=False)
                if any(res[lag][0]['ssr_ftest'][1] < alpha for lag in range(1, max_lag + 1)):
                    adj[i, j] = 1.0
            except Exception:   # degenerate pairs are left without an edge
                pass
        if progress:
            progress(i + 1, n, int(adj.sum()))
    return adj


def prior_edges(phy_boiler):
    """Channel-level process-prior edges expanded from the boiler component graph."""
    edges = set()
    for link in phy_boiler['links']:
        for s in COMPONENT_CHANNELS.get(link['source'], []):
            for t in COMPONENT_CHANNELS.get(link['target'], []):
                if s != t:
                    edges.add((s, t))
    return sorted(edges)


def union_graph(granger, edges, names):
    """Union of the Granger graph and the prior edges (edges on unknown channels are skipped)."""
    index = {c: i for i, c in enumerate(names)}
    prior = np.zeros((len(names), len(names)), dtype=np.float32)
    for src, tgt in edges:
        if src in index and tgt in index:
            prior[index[src], index[tgt]] = 1.0
    return ((granger + prior) > 0).astype(np.float32)


# ---------------------------------------------------------------- Stage 2

def subsystem(name):
    """Subsystem label used by the pruning rules (DCS points and P4 are 'oth')."""
    for p in ('P1', 'P2', 'P3'):
        if name.startswith(p):
            return p
    return 'oth'


def classify_edges(adj, names):
    """Split edges into intra-subsystem (kept), rule-removed and LLM-screened edges.

    Rules: P3 -> P1/P2 contradicts the assumed flow direction, and a process channel
    -> 'oth' would make a measurement drive a setpoint. Screened edges are numbered
    in row-major order; these numbers are the indices shown to the LLM.
    """
    keep, removed, screened = [], [], []
    for i in range(len(names)):
        for j in range(len(names)):
            if adj[i, j] > 0:
                sf, st = subsystem(names[i]), subsystem(names[j])
                if sf == st:
                    keep.append((i, j))
                elif (sf == 'P3' and st in ('P1', 'P2')) or (sf in ('P1', 'P2', 'P3') and st == 'oth'):
                    removed.append((i, j))
                else:
                    screened.append((len(screened), i, j, sf, st, names[i], names[j]))
    return keep, removed, screened


def rule_prune(adj, removed):
    """Union graph without the rule-removed edges (float32)."""
    pruned = adj.copy().astype(np.float32)
    for i, j in removed:
        pruned[i, j] = 0
    return pruned


def screening_prompt(screened):
    """User prompt listing all screened edges, grouped by subsystem pair."""
    groups = defaultdict(list)
    for gidx, i, j, sf, st, nf, nt in screened:
        groups[(sf, st)].append((gidx, nf, nt))
    sections = []
    for (sf, st), edges in sorted(groups.items()):
        lines = [f"\n[{sf}→{st}] {len(edges)} edges:"]
        lines += [f"  {gidx}: {nf} → {nt}" for gidx, nf, nt in edges]
        sections.append('\n'.join(lines))
    body = '\n'.join(sections)
    return f"""Review these cross-subsystem edges from the data-learned causal graph.
Output the INDEX LIST of edges to REMOVE (physically unjustified).

{body}

Return ONLY valid JSON (list all indices to remove):
{{"remove": [3, 7, 12, ...], "kept_valid": [0, 2, ...], "notes": "brief summary"}}"""


def parse_screening(raw):
    """JSON object of the screening response (tolerates code fences and trailing commas)."""
    text = re.sub(r'```(?:json)?', '', raw).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    for candidate in (match.group(), re.sub(r',\s*\]', ']', match.group())):
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    return None


def query_screening(screened):
    """Ask the LLM which screened edges to remove; returns the raw response text."""
    from .llm import SCREENING_SYSTEM, request
    raw, _ = request(SCREENING_SYSTEM, screening_prompt(screened),
                     {'temperature': config.LLM_TEMPERATURE, 'num_predict': 8192})
    return raw.strip()


def apply_screening(rule_pruned, screened, parsed):
    """Remove the screened edges whose indices the LLM returned in ``parsed['remove']``."""
    pruned = rule_pruned.copy()
    if not parsed or 'remove' not in parsed:
        return pruned
    index = {e[0]: (e[1], e[2]) for e in screened}
    for gidx in {int(x) for x in parsed['remove']}:
        if gidx in index:
            pruned[index[gidx]] = 0
    return pruned
