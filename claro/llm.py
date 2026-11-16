"""LLM stages: Ollama client with response cache, prompts, re-ranking and explanation check.

All requests go to a local Ollama server. Re-ranking and explanation requests are
cached under sha256(json.dumps([model, system, user, seed])); with ``offline=True``
a cache miss raises an error instead of contacting the server.
"""
import hashlib
import json
import re
import time
from collections import defaultdict

import numpy as np

from . import config

# ---------------------------------------------------------------- prompts

# Process description written from the HAI technical-details document (v4.0).
HAI_MANUAL_CONTEXT = """You are an industrial control engineer reviewing a data-learned dependency graph for the HAI 23.05 testbed.
Use ONLY the following process description, taken from the HAI technical-details document (v4.0).

Subsystems
  P1 Boiler (Emerson Ovation DCS): water-to-water heat transfer. Main water tank -> feedwater pump -> heat-transfer system -> return water tank;
     heating system and cooling system. Control loops:
       P1-PC: PCV01D/PCV02D keep pressure PIT01 at setpoint B2016.
       P1-LC: LCV01D keeps return-tank level LIT01 at setpoint B3004, with feed-forward from outflow valve FCV03D.
       P1-FC: FCV03D keeps return-tank outflow FT03 at setpoint B3005.
       P1-TC: FCV01D/FCV02D keep temperature TIT01 at setpoint B4022 (cascade with heating flow FT02).
       P1-CC: cooling pump PP04 runs at setpoint PP04SP when main-tank temperature TIT03 is in range.
     *D is a valve/pump command, *Z its measured position; FT0nZ is converted from FT0n.
  DCS internal points x1001_*, x1002_*, x1003_*: outputs of boiler DCS logic blocks 1001 (pressure control),
     1002 (level and outflow control) and 1003 (temperature control); x1001_15_ASSIGN_OUT is the pressure demand B2016.
  P2 Turbine (GE Mark VIe): rotor kit. Speed controller output SCO drives the motor so that speed SIT01 follows the
     speed demand AutoSD; VIBTR01-04 are vibration probes; switches, lamps and 24Vdc are status signals.
  P3 Water treatment (Siemens PLC): pumps water to the upper tank; level LIT01 is controlled with valve LCV01D and pump LCP01D;
     FIT01 and PIT01 are inflow rate and pressure.
  P4 HIL simulator: steam-turbine model (P4_ST_*), hydro-turbine model (P4_HT_*) and grid load (P4_LD).
Documented couplings between subsystems
  - Boiler pressure and temperature (P1_PIT01, P1_TIT01) determine STM steam pressure/temperature (P4_ST_PT01, P4_ST_TT01);
    boiler pressure, temperature and level are sent to the HIL simulator.
  - The boiler main-valve opening follows the STM steam-valve/gate opening (P4_ST_GOV).
  - The STM generates the turbine speed-setpoint trajectory (P2 AutoSD); P2_SCST follows the STM frequency change.
  - P3 flow, pressure and level are sent to the HTM; HTM discharge/pumping demands drive P3 LCV01D/LCP01D.
  - There is no water connection between the boiler (P1) and the water-treatment process (P3); P2 is a rotor kit with no water flow.
"""

# System prompt of the original graph-screening run (Stage 2). Its process description
# (an intake-treatment-distribution water plant) does not match HAI; it is kept
# verbatim because the shipped LLM-pruned graph was produced with it.
SCREENING_SYSTEM = """You are a water treatment plant instrumentation and process control expert.
You are reviewing a data-learned causal graph for the HAI plant.
Your task: identify which cross-subsystem edges are physically unjustified (spurious correlations).

HAI plant water flow direction: P1 Intake → P2 Treatment → P3 Distribution

VALID cross-subsystem edges (keep):
  - x1001/x1002/x1003 setpoints → P1/P2/P3 actuator demand signals (*D or *SP suffix)
    e.g. x1001_05_SETPOINT_OUT → P1_FCV01D: VALID (setpoint drives valve demand)
  - P1_FT03 or P1_FT03Z → P2_SIT01: VALID (main intake flow → treatment throughput)
  - P1_FT03 or P1_FT03Z → P2_AutoGO, P2_AutoSD: VALID (flow triggers treatment start/stop)
  - P2_SIT01 → P3_FIT01: VALID (treatment output → distribution inlet flow)
  - P1_LIT01 → P2_* or P3_*: NOT valid (level doesn't directly control treatment)
  - P4_* → anything: NOT valid (P4 is heating/storage, separate system)

NOT VALID (remove):
  - Any P1 measurement (FT/PIT/LIT/TIT) → P2 control signals (AutoGO/SCO/MASW etc.)
  - Any P1 sensor → P3 sensor (no direct P1→P3 physical path, goes through P2)
  - P4_* → P1/P2/P3 sensors
  - oth setpoints → measurement sensors (FT/PIT/LIT/SIT/VIBTR): setpoints affect actuators, not measurements directly
  - P2 treatment sensors → P3 distribution actuators (except SIT01→FIT01 path)

When unsure: prefer to REMOVE (less risky than keeping spurious edges)."""

RERANK_SYSTEM = HAI_MANUAL_CONTEXT + """Your task: an anomaly detector has just raised an alarm. From the candidate channels,
rank the ones most likely to have been DIRECTLY manipulated by an attacker (a setpoint, command,
or measurement that was altered), as opposed to channels that only respond to the manipulation
through the process or control loops. Use the residual evidence and the process description.
Return ONLY JSON: {"ranking": ["CHANNEL", ...]} with the five most likely channels, best first,
using channel names exactly as given."""

# Same request, but each ranked channel comes with a one-sentence reason.
EXPLAIN_SYSTEM = RERANK_SYSTEM.replace(
    'Return ONLY JSON: {"ranking": ["CHANNEL", ...]} with the five most likely channels, best first,\n'
    'using channel names exactly as given.',
    'Return ONLY JSON: {"ranking": [{"channel": "CHANNEL", "reason": "one sentence"}, ...]} with the five most likely\n'
    'channels, best first, using channel names exactly as given; each reason must cite the residual\n'
    'evidence or the control loop.')
EXPLAIN_TASK = EXPLAIN_SYSTEM[EXPLAIN_SYSTEM.index('Your task:'):]

# Prompt contexts of the explanation check: (system prompt, include channel descriptions).
EXPLANATION_CONDITIONS = {
    'manual': (EXPLAIN_SYSTEM, True),
    'generic': (SCREENING_SYSTEM.split('Your task:')[0] + EXPLAIN_TASK, True),
    'names_only': ('You are an industrial control engineer reviewing an alarm in an industrial control system.\n'
                   + EXPLAIN_TASK, False),
}

# ---------------------------------------------------------------- client and cache


def request(system, user, options):
    """One non-streaming chat request to the local Ollama server; returns (content, response)."""
    import requests
    payload = {'model': config.LLM_MODEL, 'messages': [{'role': 'system', 'content': system},
                                                       {'role': 'user', 'content': user}],
               'stream': False, 'think': False, 'options': options}
    response = requests.post(config.OLLAMA_URL, json=payload, timeout=600).json()
    return response.get('message', {}).get('content', ''), response


class CachedLLM:
    """Re-ranking/explanation requests with a persistent response cache."""

    def __init__(self, path=config.LLM_CACHE, offline=True):
        self.path = path
        self.offline = offline
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.used = set()

    @staticmethod
    def key(system, user, seed):
        return hashlib.sha256(json.dumps([config.LLM_MODEL, system, user, seed]).encode()).hexdigest()

    def ask(self, system, user, seed):
        """Cached response text for (system, user, seed)."""
        key = self.key(system, user, seed)
        if key not in self.cache:
            if self.offline:
                raise KeyError(f'LLM cache miss ({key}); rerun with --query-llm and a running Ollama server')
            t0 = time.time()
            raw, _ = request(system, user, {'temperature': config.LLM_TEMPERATURE, 'seed': seed, 'num_predict': 400})
            self.cache[key] = {'raw': raw, 'seconds': time.time() - t0}
            self.path.write_text(json.dumps(self.cache))
        self.used.add(key)
        return self.cache[key]['raw']


# ---------------------------------------------------------------- re-ranking

def relevant(entries, ends, lo, hi):
    """Ranking snapshots inside the first 60 s of fully covered episodes in raw [lo, hi).

    Only these snapshots can affect J@k, so only they are sent to the LLM.
    """
    times = ends + config.OFFSET
    keep = set()
    for ep in entries:
        if ep['fully_covered'] and ep['raw_start'] >= lo and ep['raw_stop'] <= hi:
            keep.update(np.flatnonzero((times >= ep['raw_start']) & (times < ep['raw_stop'])
                                       & (times - ep['raw_start'] <= config.DEADLINE)).tolist())
    return sorted(keep)


def evidence(stream, end):
    """Top-10 candidates at window row ``end`` with their residual evidence."""
    r45 = stream.values[45][end]
    r5 = stream.values[5][end]
    ranker = stream.ranker
    total = ranker.components(r45)[-1]
    order = np.asarray(stream.eligible)[np.argsort(-total[stream.eligible], kind='stable')][:config.LLM_TOP]
    parents = ranker.parent_weights
    rows = []
    for rank, i in enumerate(order, 1):
        parent_mean = float(r45 @ parents[:, i]) if parents[:, i].sum() else 0.
        rows.append({'rank': rank, 'channel': stream.names[i], 'index': int(i), 'r45': float(r45[i]),
                     'r5': float(r5[i]), 'parent_mean': parent_mean, 'score': float(total[i])})
    return rows


def user_prompt(rows, descriptions=None):
    """Evidence table; ``descriptions=None`` omits the channel-description column."""
    header = 'rank | channel | ' + ('description | ' if descriptions is not None else '') + \
             '45-s residual | 5-s residual | mean parent residual | CLARO score'
    lines = [header]
    for r in rows:
        described = f"{descriptions.get(r['channel'], '')} | " if descriptions is not None else ''
        lines.append(f"{r['rank']} | {r['channel']} | " + described +
                     f"{r['r45']:.2f} | {r['r5']:.2f} | {r['parent_mean']:.2f} | {r['score']:.2f}")
    return ('Residuals are in units of normal-operation residual standard deviations.\n'
            'Candidates (CLARO ranking order):\n' + '\n'.join(lines))


def parse_ranking(raw, rows):
    """LLM order of valid candidate names, followed by the remaining candidates in score order.

    Returns (order, parsed). Names outside the candidates are ignored; an unparseable
    response keeps the score order.
    """
    names = [r['channel'] for r in rows]
    match = re.search(r'\{.*\}', raw, re.S)
    order = []
    if match:
        try:
            order = [n for n in json.loads(match.group())['ranking'] if n in names]
        except (ValueError, KeyError, TypeError):
            order = []
    seen = []
    for n in order:
        if n not in seen:
            seen.append(n)
    return seen + [n for n in names if n not in seen], bool(order)


RERANK_VARIANTS = ('claro', 'llm', 'fusion')


def rerank(stream, ends, base, indices, variant, seed, llm, descriptions, log):
    """Rankings after applying a re-ranking variant at the given snapshot indices.

    'claro' keeps the score ranking; 'llm' uses the LLM order; 'fusion' orders the ten
    candidates by score rank + LLM rank, ties broken by score rank.
    """
    out = [list(r) for r in base]
    if variant == 'claro':
        return out
    for i in indices:
        rows = evidence(stream, ends[i])
        order, ok = parse_ranking(llm.ask(RERANK_SYSTEM, user_prompt(rows, descriptions), seed), rows)
        index = {r['channel']: r['index'] for r in rows}
        llm_order = [index[n] for n in order]
        if variant == 'llm':
            out[i] = llm_order[:5]
        else:
            claro = [r['index'] for r in rows]
            score = {c: claro.index(c) + llm_order.index(c) for c in claro}
            out[i] = sorted(claro, key=lambda c: (score[c], claro.index(c)))[:5]
        log.append({'snapshot': int(i), 'end': int(ends[i]), 'variant': variant, 'seed': seed, 'parsed': ok,
                    'llm_order': order[:5]})
    return out


# ---------------------------------------------------------------- explanation check

# Control loops of the HAI manual (printed pp. 7-11); DCS block outputs belong to their loop.
LOOPS = {'P1-PC': {'P1_PIT01', 'P1_PCV01D', 'P1_PCV02D', 'P1_PCV01Z', 'P1_PCV02Z', 'x1001_05_SETPOINT_OUT',
                   'x1001_15_ASSIGN_OUT'},
         'P1-LC': {'P1_LIT01', 'P1_LCV01D', 'P1_LCV01Z', 'P1_FCV03D', 'x1002_07_SETPOINT_OUT'},
         'P1-FC': {'P1_FT03', 'P1_FT03Z', 'P1_FCV03D', 'P1_FCV03Z', 'x1002_08_SETPOINT_OUT'},
         'P1-TC': {'P1_TIT01', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FT02', 'P1_FT02Z',
                   'x1003_10_SETPOINT_OUT', 'x1003_18_SETPOINT_OUT', 'x1003_24_SUM_OUT'},
         'P1-CC': {'P1_PP04', 'P1_PP04D', 'P1_PP04SP', 'P1_TIT03'},
         'P2-SC': {'P2_AutoSD', 'P2_SCO', 'P2_SIT01', 'P2_SCST'},
         'P3-LC': {'P3_LIT01', 'P3_LCV01D', 'P3_LCP01D', 'P3_FIT01', 'P3_PIT01'}}
# Ordered: the first matching phrase is the claimed role ("position command" is a command).
ROLE_WORDS = [('position command', 'command'), ('command', 'command'), ('control output', 'command'),
              ('actuator', 'command'), ('feedback', 'feedback'), ('current position', 'feedback'),
              ('position', 'feedback'), ('setpoint', 'setpoint'), ('set point', 'setpoint'), ('demand', 'setpoint'),
              ('measure', 'measurement'), ('sensor', 'measurement'), ('transmitter', 'measurement')]


def role_of(name, description):
    """Documented role of a channel from its description in the point list."""
    d = (description or '').lower()
    if 'position' in d and 'current' in d:
        return 'feedback'
    if 'converted' in d or 'conversion' in d:
        return 'measurement'
    if 'setpoint' in d or 'set-point' in d or 'demand' in d or name.startswith('x100'):
        return 'setpoint'
    if 'command' in d or 'control out' in d or 'control output' in d:
        return 'command'
    if any(w in d for w in ('measured', 'level', 'pressure', 'temperature', 'flow', 'rpm', 'vibration',
                            'displacement', 'frequency', 'power')):
        return 'measurement'
    return None


def documented_pairs(relations):
    """Undirected channel pairs documented as related (manual relations and control loops)."""
    pairs = {(r['source'], r['target']) for r in relations['relations']} | \
            {(r['source'], r['target']) for r in relations['coarse_relations']}
    pairs |= {(b, a) for a, b in pairs}
    for members in LOOPS.values():
        pairs |= {(a, b) for a in members for b in members if a != b}
    return pairs


def check_reason(entry, rows, descriptions, pairs, names):
    """Check the claims of one {'channel', 'reason'} item against the documentation.

    Returns lists of per-claim outcomes for the claim kinds found in the reason:
    numbers (quoted residuals equal the evidence, |diff| <= 0.05), loop (the cited
    loop contains the channel), relation (another cited channel is documented as
    related) and role (claimed command/feedback/setpoint/measurement role).
    """
    ch = entry.get('channel')
    reason = str(entry.get('reason', ''))
    low = reason.lower()
    res = {}
    evid = {r['channel']: r for r in rows}
    nums = [float(x) for x in re.findall(r'(?<![\w.])-?\d+\.\d+', reason)]
    if nums and ch in evid:
        values = [evid[ch][k] for k in ('r45', 'r5', 'parent_mean', 'score')] + \
                 [r[k] for r in rows for k in ('r45', 'r5', 'parent_mean', 'score')]
        res['numbers'] = [any(abs(v - x) <= .05 for v in values) for x in nums]
    loops = re.findall(r'\b(P\d-[A-Z]{2})\b', reason)
    if loops and ch in names:
        res['loop'] = [ch in LOOPS.get(lp, set()) for lp in loops]
    others = [n for n in set(re.findall(r'\b(P\d_[A-Za-z0-9_]+|x100\d_[A-Z0-9_]+)\b', reason)) if n != ch and n in names]
    if others:
        res['relation'] = [(ch, o) in pairs for o in others]
    else:   # a role word is attributed to the ranked channel only when no other channel is named
        claimed = next((r for w, r in ROLE_WORDS if w in low), None)
        truth = role_of(ch, descriptions.get(ch))
        if claimed and truth:
            res['role'] = [claimed == truth]
    return res


def parse_explanations(raw):
    """List of {'channel', 'reason'} items, or None if the response cannot be parsed."""
    try:
        items = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])['ranking']
        return [x for x in items if isinstance(x, dict)]
    except (ValueError, KeyError, TypeError):
        return None


def explain(stream, ends, base, indices, condition, llm, descriptions, pairs):
    """Explanations under one prompt condition, their checks and the resulting rankings."""
    system, with_descriptions = EXPLANATION_CONDITIONS[condition]
    names = set(stream.names)
    tally = defaultdict(list)
    parsed = 0
    rankings = [list(r) for r in base]
    items = []
    for i in indices:
        rows = evidence(stream, ends[i])
        raw = llm.ask(system, user_prompt(rows, descriptions if with_descriptions else None), config.LLM_SEED)
        explained = parse_explanations(raw)
        if explained is None:
            explained = []
        else:
            parsed += 1
        candidates = {r['channel'] for r in rows}
        order = [x.get('channel') for x in explained if x.get('channel') in candidates]
        index = {r['channel']: r['index'] for r in rows}
        rankings[i] = ([index[c] for c in order] + [r['index'] for r in rows if r['channel'] not in order])[:5]
        for x in explained:
            res = check_reason(x, rows, descriptions, pairs, names)
            for k, v in res.items():
                tally[k] += v
            items.append({'condition': condition, 'snapshot': int(i), 'raw_endpoint': int(ends[i] + config.OFFSET),
                          'channel': x.get('channel'), 'reason': x.get('reason'), 'checks': res})
    summary = {'snapshots': len(indices), 'parsed': parsed, 'reasons': len(items),
               **{k: {'checked': len(v), 'correct': int(sum(v)), 'rate': round(float(np.mean(v)), 3) if v else None}
                  for k, v in tally.items()}}
    return summary, items, rankings
