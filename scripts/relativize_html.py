# -*- coding: utf-8 -*-
"""Make all HTML file references relative (capability-relationship-map/...) and
append a complete 0.5B data manifest to section 13."""
import re
from pathlib import Path

REPO = Path("C:/Users/user/capability-relationship-map")
HTMLF = Path("C:/Users/user/Desktop/Capability Removal-Envelope Experiment — Context Handoff.html")
raw = HTMLF.read_text(encoding="utf-8")

# 1) absolute repo prefix -> relative (both slash styles)
raw = raw.replace("C:\\Users\\user\\capability-relationship-map", "capability-relationship-map")
raw = raw.replace("C:/Users/user/capability-relationship-map", "capability-relationship-map")
# 2) normalize backslashes to forward slashes inside any capability-relationship-map/... path span
raw = re.sub(r"capability-relationship-map[^\s<>\"']*", lambda m: m.group(0).replace("\\", "/"), raw)

# 3) build complete 0.5B data manifest (relative paths)
DESC = {
 "baselines.json":"per-task baseline acc + NLL (15 caps, n=40)",
 "aggregates_mean_abs.npz":"captured mean_abs activations [24,4864] per capability (the raw data A/B/C/D derive from)",
 "A_overlap.json":"Deliverable A — 15x15 overlap matrices (overlap-coeff + Jaccard, top-10/20/30%) + most/least entangled",
 "C_stress.json":"Deliverable C — fragility ranking + per-cap NLL failure fractions",
 "C_stress_rows.jsonl":"Deliverable C — full sweep rows (global+random, NLL every 0.1% step, acc at gen checkpoints)",
 "D_interlink.json":"Deliverable D — 15x15 dacc/dloss matrices (low-envelope & specific-high removal)",
 "manifest.json":"run manifest (git hash, model, seed, threshold, dims)",
 "C_fragility_bar.png":"plot: % removed at NLL-failure per capability",
 "C_retention_curves.png":"plot: per-capability NLL degradation vs % removed",
 "D_env_dacc.png":"heatmap: interlink Δacc, low-envelope removal",
 "D_env_dloss.png":"heatmap: interlink Δloss, low-envelope removal",
 "D_high_dacc.png":"heatmap: interlink Δacc, specific-high removal",
 "D_high_dloss.png":"heatmap: interlink Δloss, specific-high removal",
 # earlier 0.5B (smoke/cliff/4-task) in results/
 "sweep_coding.parquet":"smoke run: 0->10% coding sweep, 3 controls (700 rows)",
 "run_manifest.json":"smoke run manifest",
 "accuracy_retention_curve.png":"smoke run: accuracy retention, 3 controls",
 "loss_retention_curve.png":"smoke run: loss retention, 3 controls",
 "cliff_sweep_coding.parquet":"coding cliff sweep to ~50% (1986 rows)",
 "cliff_retention_to50.png":"coding cliff plot (pass@1 cliff ~25%, NLL never doubles)",
 "interlink_rows.parquet":"4-task interlink rows",
 "interlink_summary.json":"4-task interlink summary",
 "interlink_manifest.json":"4-task interlink manifest",
 "interlink_env_dacc.png":"4-task interlink heatmap (env Δacc)",
 "interlink_env_dloss.png":"4-task interlink heatmap (env Δloss)",
 "interlink_high_dacc.png":"4-task interlink heatmap (high Δacc)",
 "interlink_high_dloss.png":"4-task interlink heatmap (high Δloss)",
 "job2_fragment.html":"4-task interlink HTML fragment",
}
comp = sorted(p.name for p in (REPO/"results/comprehensive_qwen0.5b").glob("*"))
older = sorted(p.name for p in (REPO/"results").glob("*") if p.is_file())

def rows(files, base):
    out=[]
    for f in files:
        d=DESC.get(f,"")
        out.append(f'<tr><td><code>capability-relationship-map/{base}/{f}</code></td><td class="muted">{d}</td></tr>')
    return "\n".join(out)

manifest = f"""<h3 id="data05b">0.5B — complete data manifest (all files, relative paths)</h3>
<p>Every file produced/used for the 0.5B model. All paths are relative to the repo folder <code>capability-relationship-map/</code> (repo root on this machine: your home dir).</p>
<p><strong>Comprehensive 15-capability run</strong> — <code>results/comprehensive_qwen0.5b/</code>:</p>
<table><tr><th>file</th><th>what</th></tr>
{rows(comp,'results/comprehensive_qwen0.5b')}
</table>
<p><strong>Earlier 0.5B runs (smoke envelope, coding cliff, 4-task interlink)</strong> — <code>results/</code>:</p>
<table><tr><th>file</th><th>what</th></tr>
{rows(older,'results')}
</table>
<p class="muted">Batteries: <code>capability-relationship-map/data/batteries/comprehensive/*.jsonl</code> (15×40). Pipeline: <code>capability-relationship-map/scripts/run_full.py</code>, <code>run_cd.py</code>, <code>build_comprehensive.py</code>, <code>harvest_05b_cd.py</code>.</p>"""

anchor = '<p class="good"><strong>0.5B comprehensive run COMPLETE</strong>'
if 'id="data05b"' not in raw and anchor in raw:
    raw = raw.replace(anchor, manifest + "\n" + anchor, 1)
    act="manifest added"
elif 'id="data05b"' in raw:
    act="manifest already present"
else:
    act="anchor not found (manifest NOT added)"

HTMLF.write_text(raw, encoding="utf-8")
print("relativized paths;", act, "; len", len(raw))
print("remaining absolute repo refs:", raw.count("C:/Users/user/capability-relationship-map")+raw.count("C:\\Users\\user\\capability-relationship-map"))
