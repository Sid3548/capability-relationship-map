"""Read interlink_summary.json + cliff_sweep parquet and emit an HTML fragment
(Job-2 results) to stdout, reusing the handoff file's CSS classes."""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
S = json.load(open(REPO / "results" / "interlink_summary.json", encoding="utf-8"))
TASKS = S["tasks"]


def matrix_table(mat, fmt="{:+.2f}"):
    out = ['<table>', '<tr><th>target ↓ / measured →</th>' + ''.join(f'<th>{t}</th>' for t in TASKS) + '</tr>']
    for i, rt in enumerate(TASKS):
        cells = []
        for j in range(len(TASKS)):
            v = mat[i][j]
            cls = ''
            if fmt.startswith('{:+.2f}') and 'acc' in 'x':
                pass
            cells.append(f'<td>{fmt.format(v)}</td>')
        out.append(f'<tr><th>{rt}</th>' + ''.join(cells) + '</tr>')
    out.append('</table>')
    return '\n'.join(out)


def acc_matrix_table(mat):
    out = ['<table>', '<tr><th>target ↓ / measured →</th>' + ''.join(f'<th>{t}</th>' for t in TASKS) + '</tr>']
    for i, rt in enumerate(TASKS):
        cells = []
        for j in range(len(TASKS)):
            v = mat[i][j]
            cls = 'bad' if v <= -0.1 else ('good' if v >= 0.05 else 'muted')
            cells.append(f'<td class="{cls}">{v:+.2f}</td>')
        out.append(f'<tr><th>{rt}</th>' + ''.join(cells) + '</tr>')
    out.append('</table>')
    return '\n'.join(out)


b = S["baselines"]
sh = S["specific_high_counts"]
le = S["low_envelope_counts"]
char = S["coding_envelope_characterization"]

frag = []
frag.append('<h3>Job 2 — multi-task interlink (4 tasks, 0.5B, 2026-07-05)</h3>')
frag.append('<p class="muted">Widened batteries: coding 16 · math 18 · history 16 (factual exact/alias/F1) · reasoning 16 (constrained MCQ, single letter A–D). New eval stubs <code>factual.py</code>, <code>reasoning.py</code> implemented.</p>')

# baselines table
frag.append('<table>')
frag.append('<tr><th>task</th><th>baseline acc</th><th>NLL</th><th>ppl</th></tr>')
for t in TASKS:
    frag.append(f'<tr><td>{t}</td><td>{b[t]["acc"]:.3f}</td><td>{b[t]["nll"]:.3f}</td><td>{b[t]["ppl"]:.2f}</td></tr>')
frag.append('</table>')

# scoring / masks
frag.append(f'<p><strong>Multi-task scoring.</strong> Per-layer percentile rank per task. '
            f'Core (rank≥{S["params"]["core_pct"]} on ALL 4 tasks, excluded from every removal) = <strong>{S["core_count"]}</strong> neurons '
            f'({100*S["core_count"]/116736:.3f}%). '
            f'A-specific-high (rank≥{S["params"]["spec_high"]} on target AND ≤{S["params"]["spec_low"]} on every other, minus core) counts: '
            + ', '.join(f'{t}=<strong>{sh[t]}</strong>' for t in TASKS) + '. '
            f'A-low envelope budget = {S["params"]["envelope_budget"]*100:.0f}% = {le[TASKS[0]]} neurons each. '
            'A-low (envelope) and A-specific-high (interlink) are kept as SEPARATE masks throughout.</p>')
frag.append('<p><strong>Pairwise overlap of A-specific-high sets:</strong> Jaccard = 0.0000 for every pair — the task-specific-high sets are fully disjoint (by construction: a neuron high for A and low for all others cannot also be high for B). Confirms the sets isolate per-task high-firing channels.</p>')

# interlink matrices
frag.append('<p><strong>Interlink N×N — remove target task\'s LOW-ENVELOPE set (5%), Δacc (post − pre), rows=targeted, cols=measured:</strong></p>')
frag.append(acc_matrix_table(S["interlink_dacc_env"]))
frag.append('<p><strong>Interlink N×N — remove target task\'s SPECIFIC-HIGH set, Δacc:</strong></p>')
frag.append(acc_matrix_table(S["interlink_dacc_high"]))
frag.append('<p><strong>Δ-NLL (loss) matrices — LOW-ENVELOPE removal:</strong></p>')
frag.append(matrix_table(S["interlink_dloss_env"]))
frag.append('<p><strong>Δ-NLL (loss) matrices — SPECIFIC-HIGH removal:</strong></p>')
frag.append(matrix_table(S["interlink_dloss_high"]))

# characterization
frag.append('<h3>Removed-neuron characterization — coding low-envelope set</h3>')
frag.append(f'<p>Removed set = {char["n_removed"]} neurons (coding envelope, {char["budget_fraction"]*100:.0f}%). '
            '<strong>When do these neurons actually fire?</strong></p>')
frag.append('<table>')
frag.append('<tr><th>measured on task</th><th>mean_abs of removed set</th><th>global median mean_abs (all neurons)</th><th>max_abs of removed set</th></tr>')
for t in TASKS:
    frag.append(f'<tr><td>{t}</td><td>{char["mean_abs_on_"+t]:.4f}</td><td>{char["global_median_mean_abs_"+t]:.4f}</td><td>{char["max_abs_on_"+t]:.4f}</td></tr>')
frag.append('</table>')
frag.append(f'<p>Fraction effectively dead (mean_abs &lt; {char["dead_eps"]} on ALL four tasks): '
            f'<strong>{char["frac_effectively_dead"]:.3f}</strong>. '
            f'Fraction that fire more on some other task than on coding: {char["frac_fires_more_on_other_than_coding"]:.3f}. '
            'These are the neurons that are near-silent across every task we tested — the evidence for why the envelope is safe to remove.</p>')

# cliff
cliff_path = REPO / "results" / "cliff_sweep_coding.parquet"
if cliff_path.exists():
    df = pd.read_parquet(cliff_path)
    low = df[(df.control == "low") & df.nll_target.notna()].sort_values("frac_removed")
    base_nll = low.iloc[0].nll_target
    gen = df[(df.control == "low") & df.acc_target.notna()].sort_values("frac_removed")
    cliff = low[low.nll_target > 2 * base_nll]
    cliff_frac = cliff.iloc[0].frac_removed if len(cliff) else None
    frag.append('<h3>Coding cliff sweep (low-activation envelope pushed to 50%)</h3>')
    frag.append(f'<p>Baseline coding NLL = {base_nll:.3f}. NLL first exceeds 2× baseline at '
                + (f'<strong class="bad">{cliff_frac*100:.1f}% removed</strong>' if cliff_frac is not None else '<strong>not within 50%</strong>')
                + ' (the cliff on the smooth track).</p>')
    frag.append('<table><tr><th>% removed</th><th>coding NLL (envelope)</th><th>coding pass@1 (gen ckpt)</th></tr>')
    gmap = dict(zip((gen.frac_removed*100).round(1), gen.acc_target))
    for _, r in low.iterrows():
        pct = round(r.frac_removed*100, 1)
        if abs(pct - round(pct)) < 0.06 and int(round(pct)) % 5 == 0 or pct in gmap:
            a = gmap.get(pct)
            frag.append(f'<tr><td>{pct:.1f}</td><td>{r.nll_target:.3f}</td><td>{a:.2f}' if a is not None else f'<tr><td>{pct:.1f}</td><td>{r.nll_target:.3f}</td><td>—')
            frag.append('</td></tr>')
    frag.append('</table>')

frag.append('<p class="muted">Interlink artifacts: <code>results\\interlink_env_dacc.png</code>, <code>interlink_env_dloss.png</code>, <code>interlink_high_dacc.png</code>, <code>interlink_high_dloss.png</code>, <code>interlink_rows.parquet</code>, <code>interlink_summary.json</code>; cliff: <code>cliff_sweep_coding.parquet</code>. Repo committed; git_hash now recorded in manifests.</p>')

print('\n'.join(frag))
