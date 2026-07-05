"""Emit an HTML <h2> section from a comprehensive_* results dir.
Usage: python scripts/emit_section.py <resultsdir> <section_num> <model_label> <out_html_fragment>
Reads baselines.json, A_overlap.json, B_allocation.json, C_stress.json, D_interlink.json.
"""
import json, sys
from pathlib import Path

d = Path(sys.argv[1]); sec = sys.argv[2]; label = sys.argv[3]; out = Path(sys.argv[4])
def J(n):
    p = d / n
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

base = J("baselines.json"); A = J("A_overlap.json"); B = J("B_allocation.json")
C = J("C_stress.json"); D = J("D_interlink.json")
CAPS = (C or A or {}).get("caps") or (B and list(B["per_cap_exclusive_count"].keys())) or []
if not CAPS and base: CAPS = list(base.keys())

f = []
f.append(f'<h2 id="comp{sec}">{sec} · Comprehensive capability↔neuron map — {label}</h2>')
f.append('<p class="sub">15 capabilities × 40 prompts = 600 total. Universal primary metric = teacher-forced NLL over gold (comparable across all 15). Active-neuron threshold X = mean_abs rank in the TOP 20% within each layer (per-layer percentile, because SiLU activation scale is unbounded and layer-dependent, so a raw fixed cutoff is not comparable across layers; sensitivity also computed at top-10% and top-30%). Honest framing: these are active-set overlaps under this activation basis, NOT literal claims that a cognitive function "lives" in specific neurons.</p>')

# Baselines
if base:
    f.append('<h3>Baselines (per capability)</h3><table><tr><th>capability</th><th>accuracy</th><th>NLL (primary)</th><th>eval type</th></tr>')
    for c in CAPS:
        b = base[c]; acc = "—" if b["acc"] is None else f'{b["acc"]:.3f}'
        f.append(f'<tr><td>{c}</td><td>{acc}</td><td>{b["nll"]:.3f}</td><td class="muted">{b["eval_type"]}</td></tr>')
    f.append('</table>')

# A overlap
if A:
    f.append('<h3>Deliverable A — active-set overlap (15×15)</h3>')
    f.append('<p>Headline metric = <strong>overlap coefficient</strong> |A∩B|/min(|A|,|B|) at top-20% (Jaccard also computed; overlap-coeff is the headline because active-set sizes differ across capabilities and Jaccard is deflated by that). Conclusions checked stable across top-10/20/30%.</p>')
    f.append('<p><strong>Most entangled pairs</strong> (overlap-coeff, Jaccard):</p><ul>')
    for a,b,o,jc in A["most_entangled"]:
        f.append(f'<li>{a} ↔ {b}: overlap={o:.3f}, jaccard={jc:.3f}</li>')
    f.append('</ul><p><strong>Least entangled pairs:</strong></p><ul>')
    for a,b,o,jc in A["least_entangled"]:
        f.append(f'<li>{a} ↔ {b}: overlap={o:.3f}, jaccard={jc:.3f}</li>')
    f.append('</ul><p class="muted">Heatmap: <code>A_overlap_coeff.png</code>; matrix + rows in <code>A_overlap.json</code> / <code>A_overlap_rows.parquet</code>.</p>')

# B allocation
if B:
    bk = B["buckets"]; bp = B["buckets_pct"]; tot = B["total_neurons"]
    f.append('<h3>Deliverable B — neuron allocation map</h3>')
    f.append(f'<p>All {tot:,} MLP neurons classified by how many of 15 capabilities they are active for (top-20%):</p>')
    f.append('<table><tr><th>bucket</th><th>definition</th><th>neurons</th><th>% of all</th></tr>')
    f.append(f'<tr><td>DEAD</td><td>active for 0 caps</td><td>{bk["dead"]:,}</td><td>{100*bp["dead"]:.1f}%</td></tr>')
    f.append(f'<tr><td>EXCLUSIVE</td><td>active for exactly 1</td><td>{bk["exclusive"]:,}</td><td>{100*bp["exclusive"]:.1f}%</td></tr>')
    f.append(f'<tr><td>SHARED</td><td>active for 2–12</td><td>{bk["shared"]:,}</td><td>{100*bp["shared"]:.1f}%</td></tr>')
    f.append(f'<tr><td>CORE</td><td>active for ≥13/15</td><td>{bk["core"]:,}</td><td>{100*bp["core"]:.1f}%</td></tr>')
    f.append('</table>')
    ex = B["per_cap_exclusive_count"]; expct = B["per_cap_exclusive_pct"]
    f.append('<p><strong>Exclusive neurons per capability</strong> (active for only that capability):</p>')
    f.append('<table><tr><th>capability</th><th>exclusive neurons</th><th>% of all neurons</th></tr>')
    for c in sorted(CAPS, key=lambda x: -ex[x]):
        f.append(f'<tr><td>{c}</td><td>{ex[c]:,}</td><td>{100*expct[c]:.2f}%</td></tr>')
    f.append('</table><p class="muted">Chart: <code>B_allocation.png</code>.</p>')

# C stress
if C:
    f.append('<h3>Deliverable C — global stress sweep (where it breaks)</h3>')
    f.append(f'<p>Neurons removed ascending by GLOBAL importance (least-used across all 15 caps first), mean-ablation, 0.1% steps to {C["max_fraction"]*100:.0f}% removed; random control (5 seeds) as reference band; NLL for all 15 every step. FAILURE per capability = NLL ratio &gt; 2× its own baseline. Fragility ranking = order in which capabilities fail.</p>')
    ff = C["failure_fraction_pct"]; rank = C["fragility_ranking_most_to_least_fragile"]
    f.append('<p><strong>FRAGILITY RANKING (most fragile first) — % of neurons removed at failure:</strong></p>')
    f.append('<table><tr><th>rank</th><th>capability</th><th>% removed at 2× NLL failure</th></tr>')
    for i,c in enumerate(rank):
        v = ff[c]
        vs = f'{v*100:.1f}%' if v is not None else f'&gt;{C["max_fraction"]*100:.0f}% (never failed)'
        f.append(f'<tr><td>{i+1}</td><td>{c}</td><td>{vs}</td></tr>')
    f.append('</table>')
    firsts = [c for c in rank if ff[c] is not None]
    if firsts:
        f.append(f'<div class="callout crux"><strong>Headline:</strong> The most fragile cognitive function is <strong>{rank[0]}</strong> (fails at {ff[rank[0]]*100:.1f}% removed). The most robust survive past {C["max_fraction"]*100:.0f}% removed. Curves: <code>C_retention_curves.png</code>, <code>C_fragility_bar.png</code>.</div>')
    f.append('<p class="muted">Full per-step rows: <code>C_stress_rows.parquet</code>.</p>')

# D interlink
if D:
    f.append('<h3>Deliverable D — 15×15 interlink</h3>')
    f.append(f'<p>For each capability, removed (i) its low-envelope and (ii) its specific-high set at a fixed {D["budget"]*100:.0f}% budget; evaluated all 15. Δacc and Δloss 15×15 matrices (rows = targeted, cols = collateral). Heatmaps: <code>D_env_dacc.png</code>, <code>D_env_dloss.png</code>, <code>D_high_dacc.png</code>, <code>D_high_dloss.png</code>; data in <code>D_interlink.json</code>.</p>')
    # summarize: mean self vs off-diagonal for specific-high dloss
    import numpy as np
    dh = np.array(D["dloss_high"]); n=len(CAPS)
    self_d = np.mean([dh[i,i] for i in range(n)])
    off = np.mean([dh[i,j] for i in range(n) for j in range(n) if i!=j])
    f.append(f'<p>Specific-high removal Δloss: mean self-damage = {self_d:+.3f}, mean off-diagonal (collateral) = {off:+.3f} → specific-high neurons are {"more" if self_d>off else "not more"} selectively important to their own capability.</p>')

out.write_text("\n".join(f), encoding="utf-8")
print(f"wrote {out} ({len(''.join(f))} chars)")
