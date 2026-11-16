"""Paths and fixed constants of the CLARO pipeline.

Raw and preprocessed HAI data are not part of the repository. Their locations can be
set with the environment variables ``CLARO_RAW`` (official HAI repository clone) and
``CLARO_DATA`` (preprocessed arrays written by ``scripts/01_preprocess.py``).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = Path(os.environ.get('CLARO_RAW', ROOT / 'data' / 'raw' / 'hai'))
DATA_DIR = Path(os.environ.get('CLARO_DATA', ROOT / 'data' / 'processed'))
GRAPH_DIR = ROOT / 'data' / 'graphs'
GROUND_TRUTH_DIR = ROOT / 'data' / 'ground_truth'
DOC_DIR = ROOT / 'data' / 'documentation'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'

# Graph files (adjacency A[j, i] = 1 means an edge j -> i).
GRANGER_GRAPH = GRAPH_DIR / 'granger_adj.npy'
PRIOR_EDGES = GRAPH_DIR / 'prior_edges.json'
UNION_GRAPH = GRAPH_DIR / 'union_adj.npy'
RULE_PRUNED_GRAPH = GRAPH_DIR / 'rule_pruned_adj.npy'
LLM_PRUNED_GRAPH = GRAPH_DIR / 'llm_pruned_adj.npy'
SENSOR_COLS = GRAPH_DIR / 'sensor_cols.txt'

# Stage 1: Granger predictive edges.
GRANGER_ROWS = 5000
GRANGER_MAX_LAG = 5
GRANGER_ALPHA = 0.01

# Stage 3: AR-SEM and window residuals.
LAG = 2                      # parents enter at lag 2; the channel itself at lag 1
RANK_WINDOW = 45             # window (s) of the residual vector used for ranking
OFFSET = RANK_WINDOW - 1     # residual-window row i ends at raw test sample i + OFFSET

# Stage 4: online protocol.
VALIDATION_FRACTION = 0.4    # chronological validation prefix of the test timeline
SNAPSHOT_PERIOD = 45         # ranking at alarm onset and every 45 samples while alarming
DEADLINE = 60                # joint J@k deadline (s) after attack onset
TEST_FILE_BOUNDARY = 54000   # first sample of hai-test2 in the concatenated test set
CALIBRATED_QUANTILE = 0.99   # per-channel quantile of the optional calibrated detector/ranker
FA_LIMIT = 0.02              # prefix normal-time alarm limit of the calibrated detector

# LLM (local Ollama server).
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/chat')
LLM_MODEL = 'gemma4:31b'
LLM_TEMPERATURE = 0.05
LLM_SEED = 11
LLM_SEEDS = (11, 12, 13)
LLM_TOP = 10                 # candidates passed to the LLM
LLM_CACHE = RESULTS_DIR / 'llm_cache.json'
