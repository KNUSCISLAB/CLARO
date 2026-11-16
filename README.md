# Alarm-Conditioned Root Cause Attribution with Autoregressive Residuals and LLM Explanations for Industrial Control Systems

CLARO detects attacks on an industrial control system from its sensor streams and, at each
alarm, ranks the channels most likely to have been manipulated directly. It builds a
dependency graph from Granger predictive edges and process-prior edges, prunes
cross-subsystem edges with programmatic rules and a local large language model (LLM), fits an
autoregressive structural equation model (AR-SEM) on the pruned graph, raises alarms from
multi-scale AR-SEM residuals, ranks candidate channels at alarm time with a graph-based
attribution score, and finally lets the LLM re-rank the top candidates and explain each choice
using the plant documentation. This repository contains the implementation and the saved
outputs for the HAI 23.05 dataset.

The framework is shown in [`figures/fig_framework.pdf`](figures/fig_framework.pdf).

## Method summary

1. **Dependency graph** (`claro/graph.py`). Pairwise Granger edges on the first 5,000
   standardized normal training rows (statsmodels SSR F-test, maximum lag 5, p < 0.01; 1,087
   edges), unioned with 61 process-prior edges expanded from the HAI boiler component graph
   (1,103 edges).
2. **Graph pruning** (`claro/graph.py`, `claro/llm.py`). Cross-subsystem edges that
   contradict the assumed flow direction or would let a measurement drive a setpoint are
   removed (254 edges); the remaining 401 cross-subsystem edges are screened by `gemma4:31b`
   in one request (396 removed), giving the 453-edge pruned graph.
3. **AR-SEM** (`claro/arsem.py`). Each standardized channel is regressed by ordinary least
   squares on its own lag-1 value and its graph parents at lag 2; absolute residuals are
   divided by the training residual standard deviation and averaged over 5-s and 45-s windows
   aligned on the same endpoint.
4. **Detector** (`claro/detector.py`). The maximum window residual over channels and window
   lengths; orientation, min-max scale and the threshold maximizing pointwise F1 are fitted on
   the validation prefix (first 40% of the test timeline) only.
5. **Alarm-time ranking** (`claro/ranking.py`). At each alarm onset and every 45 samples while
   the alarm persists, candidate channels (P1/P2/P3 and boiler DCS outputs) are ranked by
   `rc(i) = r_i - 0.5 * mean parent residual + 0.5 * log(1 + |descendants within distance 4|)`
   on the current 45-s residual vector.
6. **LLM re-ranking and explanations** (`claro/llm.py`). The ten highest-scoring candidates,
   with channel descriptions from the HAI technical document and their residual evidence, are
   sent to the LLM, which returns five channels with a one-sentence reason each. The claims in
   the reasons are checked automatically against the documentation.

Optional modules (AR term, 5-s window, per-channel calibrated detector, calibrated-residual
ranking; 16 combinations) are selected on the validation prefix by joint J@1, then J@5, then
lower prefix normal-time alarm fraction, then fewer changes from the base configuration
(lag-2 SEM, 45-s window, maximum-residual detector, attribution score). The selection is
written to disk before the suffix is scored. In the original experiments this criterion
pooled the HAI prefix with the prefix of a second dataset that is not part of this
repository; here it is applied to the HAI prefix alone and selects the same configuration
(its prefix J@1 of 13/19 is the unique maximum on HAI).

## Repository layout

```
claro/                 package
  config.py            paths (CLARO_RAW, CLARO_DATA) and constants
  data.py              HAI 23.05 preprocessing and loading
  graph.py             Granger graph, prior edges, union, rule pruning, LLM screening
  arsem.py             AR-SEM (and base lag-2 SEM), window residuals
  detector.py          detector score, prefix calibration, thresholds, alarm emission
  ranking.py           candidate set and attribution score
  evaluation.py        episodes and targets, joint J@k, detection and timeliness metrics
  pipeline.py          online stream, module grid and selection key
  llm.py               Ollama client and cache, prompts, re-ranking, explanation check
scripts/               01_preprocess.py ... 05_figures.py (run in order)
data/graphs/           sensor_cols.txt, granger_adj.npy, prior_edges.json, union_adj.npy,
                       rule_pruned_adj.npy, llm_pruned_adj.npy
data/ground_truth/     attack episodes, source-listed target points, documented aliases
data/documentation/    channel descriptions and documented channel relations (HAI manual)
results/               saved outputs (see below)
figures/               fig_framework.pdf, fig_delay_rank.pdf
```

## Installation

```bash
pip install -r requirements.txt
```

The code was run with Python 3.9 (numpy 2.0.2, scikit-learn 1.5.0) for steps 3 and 4,
Python 3.9 with numpy 1.25, pandas 1.4.3 and statsmodels 0.13.2 for steps 1 and 2, and
Python 3.10 with matplotlib 3.10.9 for step 5.

**LLM (optional).** The LLM stages use a local [Ollama](https://ollama.com) server
(`http://localhost:11434`, override with `OLLAMA_URL`) and the model tag `gemma4:31b`
(temperature 0.05; seed 11, and seeds 12 and 13 for the repeated suffix runs). All 328
re-ranking and explanation responses are stored in `results/llm_cache.json`, keyed by the
SHA-256 of `json.dumps([model, system prompt, user prompt, seed])`, so step 4 runs without a
GPU or an Ollama server. The graph-screening response is stored in
`results/graph_screening.json`.

## Data

The HAI dataset is not redistributed here. Obtain it from the official repository,
<https://github.com/icsdataset/hai> (data files are stored with Git LFS; run `git lfs pull`),
and point `CLARO_RAW` to the clone (default `data/raw/hai`). Step 1 reads the official
comma-separated files

```
<CLARO_RAW>/hai-23.05/hai-train{1,2,3,4}.csv, hai-test{1,2}.csv, label-test{1,2}.csv
```

(`--suffix` selects another file extension if the files were renamed). Step 2 additionally
reads `<CLARO_RAW>/graph/boiler/phy_boiler.json` when it is present; otherwise the shipped
`data/graphs/prior_edges.json` is used.

Preprocessed arrays are written to `CLARO_DATA` (default `data/processed`):
`X_train.npy` (896,400 x 66, float32), `X_test.npy` (284,400 x 66, float32), `y_test.npy`
(284,400, int8) and `sensor_cols.txt`.

## Usage

```bash
export CLARO_RAW=/path/to/hai          # clone of the official HAI repository
export CLARO_DATA=data/processed

python scripts/01_preprocess.py        # HAI 23.05 -> numpy arrays
python scripts/02_build_graph.py       # union, rule-pruned and LLM-pruned graphs
                                       #   --granger    recompute the Granger graph (slow)
                                       #   --query-llm  ask the LLM again instead of replaying
python scripts/03_run_claro.py         # select modules on the prefix, then evaluate the suffix
                                       #   --step select | evaluate | all
python scripts/04_llm_explain.py       # LLM re-ranking and explanation check (cache only)
                                       #   --query-llm  send cache misses to Ollama
MPLCONFIGDIR=/tmp/claro-matplotlib python scripts/05_figures.py
```

Without `--query-llm`, step 4 never contacts the LLM and stops with an error on a cache
miss. Step 2 without `--query-llm` replays the stored screening decision; the original
screening request had no seed, so a new query may return a different graph.

## Outputs

| File | Written by | Content |
|---|---|---|
| `data/graphs/*.npy` | 02 | Granger, union, rule-pruned and LLM-pruned graphs (`A[j, i] = 1`: edge j -> i) |
| `results/graph_screening.json` | 02 (`--query-llm`) | raw and parsed LLM graph-screening response |
| `results/selected_configuration.json` | 03 | selected modules and prefix results of all 16 combinations |
| `results/evaluation.json` | 03 | prefix and suffix J@k per episode, suffix detection metrics, suffix rankings |
| `results/alarm_delays.json` | 03 | onset-to-first-alarm delay per suffix attack (CLARO and base) |
| `results/llm_rerank.json` | 04 | re-ranking variants on the prefix (selection) and suffix (seeds 11-13) |
| `results/llm_explanations.json` | 04 | one-sentence explanations of the manual-grounded prompt |
| `results/explanation_check.json` | 04 | documentation-agreement check for three prompt contexts |
| `results/llm_cache.json` | 04 (`--query-llm`) | LLM response cache |
| `results/oracle_target_ranks.json` | stored | rank of the first target on oracle attack segments (Figure, panel b) |
| `figures/fig_delay_rank.pdf` | 05 | onset-to-alarm delay and target rank |

`results/oracle_target_ranks.json` is a stored input of the figure: it contains the
per-episode ranks of CLARO and of comparison methods (SEM baselines and neural detectors)
from the oracle-segment evaluation of the manuscript, which is not part of this repository.

## Results

Running steps 1-5 on HAI 23.05 with this code gives the following results. The suffix is
the evaluation part of the test timeline after the 40% validation prefix and a 44-sample
purge. J@k counts fully covered attack episodes (all source-listed target points map to a
candidate channel) in which an alarm-time ranking within 60 s of onset has a target channel
in its top k.

| | Prefix (validation) | Suffix (evaluation) |
|---|---|---|
| Selected modules | AR-SEM, 5-s and 45-s windows, maximum-residual detector, attribution score | |
| J@1 | 13 / 19 | 5 / 9 |
| J@5 | 15 / 19 | 5 / 9 |
| Attacks alarmed within 60 s of onset | | 16 / 25 |
| Normal-time alarm fraction | 2.63% | 2.53% |
| LLM re-ranking J@1 (selected on prefix: LLM) | 14 / 19 | 5 / 9 (seeds 11, 12, 13) |

Agreement of the LLM explanations with the HAI documentation (70 alarm snapshots, 350
reasons per context; in parentheses the number of reasons containing that kind of claim):

| Context | Loop | Role | Relation | Numbers |
|---|---|---|---|---|
| Manual + channel descriptions (CLARO) | 97.9% (239) | 87.0% (269) | 95.8% (24) | 100.0% (35) |
| Mismatched process description + channel descriptions | none | 91.7% (218) | 89.1% (46) | 100.0% (79) |
| Channel names only | none | 66.5% (224) | 81.5% (27) | 100.0% (79) |

### Reproduction check

The refactored code was run on HAI 23.05 and its outputs were compared with those of the
original research code:

- Step 1 reproduces the preprocessed arrays and channel list exactly.
- Step 2 reproduces the Granger graph (1,087 edges), the union graph (1,103), the rule-pruned
  graph (849) and, by replaying the stored LLM decision, the LLM-pruned graph (453) exactly.
  The screening user prompt is byte-identical to the original one.
- Step 3 reproduces the selected configuration, the prefix results of all 16 module
  combinations, the per-episode suffix results and all suffix detection metrics.
- Step 4, using only the cache (all 328 requests hit), reproduces the re-ranking selection,
  all suffix results per variant and seed, and every checked explanation item of the three
  prompt contexts.
- Step 5 reproduces the manuscript figure; the rasterized pages (200 dpi) are identical.

The graph-screening LLM response itself (step 2 with `--query-llm`) was not re-queried.
