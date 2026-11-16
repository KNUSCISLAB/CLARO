"""Step 5: Figure of onset-to-alarm delay and target rank (figures/fig_delay_rank.pdf).

Panel (a) reads results/alarm_delays.json (written by 03_run_claro.py). Panel (b)
reads results/oracle_target_ranks.json, the per-episode rank of the first
source-listed target on oracle attack segments as reported in the manuscript; the
comparison methods in that panel are not part of this repository.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claro import config  # noqa: E402

CLARO_C, BASE_C, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a63', '#e4e3de'
RANK_C = ['#0d366b', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef']
NONE_C = '#efeeea'
METHODS = [('CLARO-AR', 'CLARO'), ('SEM+LLM', 'SEM+LLM'), ('SEM-RCA', 'SEM-RCA'), ('timesnet', 'TimesNet'),
           ('tranad', 'TranAD'), ('gdn', 'GDN')]


def delay_panel(ax, delays):
    base = [r['delay'] for r in delays['base']]
    claro = [r['delay'] for r in delays['claro']]
    top = 1500

    def plot(v):
        return top if v is None else max(v, 1)

    order = sorted(range(len(claro)), key=lambda i: (plot(claro[i]), plot(base[i])))
    x = np.arange(len(order))
    for j, i in enumerate(order):
        ax.plot([j, j], [plot(base[i]), plot(claro[i])], color=GRID, lw=1.2, zorder=1)
    ax.scatter(x, [plot(base[i]) for i in order], s=16, color=BASE_C, edgecolor='white', linewidth=.6, zorder=3,
               label='Base (lag-2 SEM, 45 s)')
    ax.scatter(x, [plot(claro[i]) for i in order], s=16, color=CLARO_C, marker='D', edgecolor='white', linewidth=.6,
               zorder=4, label='CLARO')
    ax.axhline(60, color=INK, lw=.8, ls=(0, (3, 2)))
    ax.text(len(order) - .5, 66, '60 s', ha='right', va='bottom', color=INK, fontsize=6.5)
    ax.set_yscale('log')
    ax.set_ylim(.8, top * 1.5)
    ax.set_yticks([1, 10, 60, 300, top])
    ax.set_yticklabels(['1', '10', '60', '300', 'none'])
    ax.set_xticks([])
    ax.set_xlabel(f'Suffix attacks ({len(order)}), ordered by CLARO delay', color=INK)
    ax.set_ylabel('Onset to first alarm (s)')
    ax.set_title('(a) Delay to first alarm', loc='center', color=INK)
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=6.5, handletextpad=.3, borderaxespad=.2)


def rank_panel(bx, ranks):
    n = len(ranks['CLARO-AR'])
    for row, (key, _) in enumerate(METHODS):
        for col, r in enumerate(ranks[key]):
            bx.add_patch(plt.Rectangle((col + .06, row + .06), .88, .88, color=RANK_C[r - 1] if r else NONE_C, lw=0))
            if r:
                bx.text(col + .5, row + .52, str(r), ha='center', va='center', fontsize=5.2,
                        color='white' if r <= 3 else INK)
        top1 = sum(v == 1 for v in ranks[key])
        bx.text(n + .25, row + .5, f'{top1}/{n}', va='center', fontsize=6.3, color=INK,
                fontweight='bold' if key == 'CLARO-AR' else 'normal')
    bx.set_xlim(0, n + 2.3)
    bx.set_ylim(len(METHODS), 0)
    bx.set_yticks(np.arange(len(METHODS)) + .5, [label for _, label in METHODS])
    for t, (key, _) in zip(bx.get_yticklabels(), METHODS):
        t.set_color(INK)
        t.set_fontweight('bold' if key == 'CLARO-AR' else 'normal')
    bx.set_xticks([])
    bx.set_xlabel(f'Fully covered episodes ({n}); cell = rank of first target (blank: >5)', color=INK)
    bx.text(n + .25, -.25, 'Top-1', fontsize=6.3, color=MUTED, va='bottom')
    bx.set_title('(b) Target rank on oracle segments', loc='center', color=INK)
    for side in bx.spines.values():
        side.set_visible(False)
    bx.tick_params(length=0)


def main():
    delays = json.loads((config.RESULTS_DIR / 'alarm_delays.json').read_text())
    ranks = json.loads((config.RESULTS_DIR / 'oracle_target_ranks.json').read_text())['methods']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                         'font.size': 7, 'axes.titlesize': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={'width_ratios': [1.05, 1], 'wspace': .32})
    delay_panel(ax, delays)
    rank_panel(bx, ranks)
    fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.17)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / 'fig_delay_rank.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
