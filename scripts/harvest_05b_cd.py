# -*- coding: utf-8 -*-
"""Harvest 0.5B C (fragility) + D (interlink) JSON into HTML section 13, and make
torch-free PNGs. Replaces the '...pending...' placeholder in section 13."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path("C:/Users/user/capability-relationship-map/results/comprehensive_qwen0.5b")
HTMLF = Path("C:/Users/user/Desktop/Capability Removal-Envelope Experiment — Context Handoff.html")
CAPS = ["coding","math","formal_logic","grammar","translation","reading_comprehension","history_facts","philosophy","science_facts","commonsense","problem_solving","creative_writing","summarization","spatial_pattern","ethics"]
SH = {"coding":"code","math":"math","formal_logic":"logic","grammar":"gram","translation":"transl","reading_comprehension":"readc","history_facts":"hist","philosophy":"phil","science_facts":"sci","commonsense":"cmnsn","problem_solving":"probs","creative_writing":"creat","summarization":"summ","spatial_pattern":"spat","ethics":"ethic"}
sh=[SH[c] for c in CAPS]

base=json.load(open(D/"baselines.json"))
C=json.load(open(D/"C_stress.json"))
Dl=json.load(open(D/"D_interlink.json"))
rows=[json.loads(l) for l in open(D/"C_stress_rows.jsonl")]
g=[r for r in rows if r["control"]=="global"]
gen=[r for r in g if any(k.startswith("acc_") for k in r)]

ff=C["failure_fraction_pct"]; ranking=C["fragility_ranking_most_to_least_fragile"]

# ---- accuracy retention at ~20/35/50% ----
def near(frac):
    return min(gen,key=lambda r:abs(r["frac"]-frac))
ckpts=[near(0.2),near(0.35),near(0.5)]

# ---- PNGs ----
# fragility bar
fig,ax=plt.subplots(figsize=(11,4.5))
vals=[(ff[c] if ff[c] is not None else 0.72)*100 for c in ranking]
ax.bar(range(15),vals,color=["#ff7a7a" if ff[c] is not None else "#5fd0a0" for c in ranking])
ax.set_xticks(range(15)); ax.set_xticklabels(ranking,rotation=60,ha="right",fontsize=8)
ax.set_ylabel("% removed at NLL-failure (2x)"); ax.set_title("0.5B fragility: NLL-doubling under global least-used-first removal (green=never failed)")
fig.tight_layout(); fig.savefig(D/"C_fragility_bar.png",dpi=140); plt.close(fig)
# retention curves (nll ratio)
gg=sorted([r for r in g],key=lambda r:r["frac"])
fig,ax=plt.subplots(figsize=(11,6))
for c in CAPS: ax.plot([r["frac"]*100 for r in gg],[r[f"nllratio_{c}"] for r in gg],lw=1,label=c)
ax.axhline(2,color="k",ls="--",lw=.8); ax.set_yscale("log"); ax.set_xlabel("% neurons removed (global)"); ax.set_ylabel("NLL ratio vs baseline (log)")
ax.set_title("0.5B global stress: per-capability NLL degradation"); ax.legend(fontsize=7,ncol=3)
fig.tight_layout(); fig.savefig(D/"C_retention_curves.png",dpi=140); plt.close(fig)
# interlink heatmaps
def heat(mat,name,title):
    m=np.array(mat); fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(m,cmap="RdBu",vmin=-max(0.01,np.abs(m).max()),vmax=max(0.01,np.abs(m).max()))
    ax.set_xticks(range(15)); ax.set_xticklabels(sh,rotation=60,ha="right",fontsize=7); ax.set_yticks(range(15)); ax.set_yticklabels(sh,fontsize=7)
    ax.set_xlabel("measured"); ax.set_ylabel("removed target"); ax.set_title(title); fig.colorbar(im); fig.tight_layout(); fig.savefig(D/name,dpi=140); plt.close(fig)
heat(Dl["dacc_env"],"D_env_dacc.png","Interlink Δacc: remove target LOW-ENVELOPE (5%)")
heat(Dl["dacc_high"],"D_high_dacc.png","Interlink Δacc: remove target SPECIFIC-HIGH (5%)")
heat(Dl["dloss_env"],"D_env_dloss.png","Interlink Δloss: remove LOW-ENVELOPE")
heat(Dl["dloss_high"],"D_high_dloss.png","Interlink Δloss: remove SPECIFIC-HIGH")

# ---- interlink highlights ----
de=np.array(Dl["dacc_env"]); dh=np.array(Dl["dacc_high"])
diag_hi=[(CAPS[i],dh[i,i]) for i in range(15)]
diag_hi.sort(key=lambda x:x[1])
off=[]
for i in range(15):
    for j in range(15):
        if i!=j: off.append((CAPS[i],CAPS[j],dh[i,j]))
off.sort(key=lambda x:x[2])

# ---- build HTML ----
o=[]; w=o.append
w('<h3>Deliverable C — Global stress sweep to 70% removed (fragility: which capability fails first)</h3>')
w('<p>Neurons removed in ascending order of GLOBAL importance (least-used across all 15 caps first), 0.1% steps, mean-ablation, + random control (5 seeds). Two failure lenses — and <strong>they disagree, which is itself the result</strong>.</p>')
w('<div class="callout warn"><strong>The two rankings flip.</strong> By smooth loss (NLL doubling), the low-baseline-NLL <em>verbal/shared-wiring</em> cluster fails first (grammar 10%, commonsense 12%, translation 20%); math/science/problem_solving NEVER hit 2× even at 70%. But by <strong>accuracy</strong>, the exact-answer tasks crater fastest — math & coding are near-0 by ~35% removed. Reason: NLL-doubling is confounded by baseline confidence (a task already at high NLL has headroom), and exact-answer tasks fail on a single wrong token while 4-way MCQ tasks degrade gracefully. Honest global break point: <strong>~35% of least-used neurons removed collapses generation competence across the board.</strong></div>')
w('<table><tr><th>rank</th><th>capability</th><th>NLL-fails @ (2× baseline)</th><th>baseline NLL</th></tr>')
for k,c in enumerate(ranking):
    fv = "never (&gt;70%)" if ff[c] is None else f"{ff[c]*100:.1f}%"
    cls=' class="bad"' if ff[c] is not None and ff[c]<0.25 else ''
    w(f'<tr><td>{k+1}</td><td{cls}>{c}</td><td>{fv}</td><td class="muted">{base[c]["nll"]:.2f}</td></tr>')
w('</table>')
w('<p><strong>Accuracy retention (acc/baseline) at global-removal checkpoints — complementary lens:</strong></p>')
w('<div style="overflow-x:auto"><table style="font-size:12px"><tr><th>% removed</th>'+''.join(f'<th>{s}</th>' for s in sh)+'</tr>')
for r in ckpts:
    w(f'<tr><td>{r["frac"]*100:.0f}%</td>')
    for c in CAPS:
        b=base[c]["acc"]; a=r.get(f"acc_{c}")
        if b and a is not None:
            ret=a/b; cls=' class="bad"' if ret<0.5 else (' class="good"' if ret>0.9 else '')
            w(f'<td{cls}>{ret:.2f}</td>')
        else: w('<td class="muted">—</td>')
    w('</tr>')
w('</table></div>')
w('<p class="muted">Plots: <code>results\\comprehensive_qwen0.5b\\C_fragility_bar.png</code>, <code>C_retention_curves.png</code>. Raw: <code>C_stress.json</code>, <code>C_stress_rows.jsonl</code>.</p>')

w('<h3>Deliverable D — 15×15 interlink (remove each capability\'s neurons, measure collateral)</h3>')
w('<p>Diagonal = self-damage; off-diagonal = collateral on other capabilities. Δacc = accuracy change vs baseline (post−pre), 5% budget.</p>')
w('<p><strong>Biggest self-damage (remove a cap\'s SPECIFIC-HIGH set, its own Δacc):</strong> '+"; ".join(f"{c} {v:+.2f}" for c,v in diag_hi[:5])+'</p>')
w('<p><strong>Strongest cross-capability collateral (remove row → hurts col):</strong> '+"; ".join(f"{a}→{b} {v:+.2f}" for a,b,v in off[:6])+'</p>')
for mat,label in [(Dl["dacc_env"],"Δacc — remove LOW-ENVELOPE"),(Dl["dacc_high"],"Δacc — remove SPECIFIC-HIGH")]:
    M=np.array(mat)
    w(f'<p><strong>{label}</strong> (rows=removed target, cols=measured):</p>')
    w('<div style="overflow-x:auto"><table style="font-size:11px"><tr><th></th>'+''.join(f'<th>{s}</th>' for s in sh)+'</tr>')
    for i in range(15):
        w(f'<tr><th>{sh[i]}</th>')
        for j in range(15):
            v=M[i,j]
            if i==j: cell=f'<td class="muted">{v:+.2f}</td>'
            elif v<=-0.15: cell=f'<td class="bad">{v:+.2f}</td>'
            elif v>=0.15: cell=f'<td class="good">{v:+.2f}</td>'
            else: cell=f'<td>{v:+.2f}</td>'
            w(cell)
        w('</tr>')
    w('</table></div>')
w('<p class="muted">Δloss (NLL) matrices for both conditions are in <code>D_interlink.json</code> (verbatim). Heatmaps: <code>D_env_dacc.png, D_high_dacc.png, D_env_dloss.png, D_high_dloss.png</code>.</p>')
w('<p class="good"><strong>0.5B comprehensive run COMPLETE</strong> (A+B+C+D). Next: Qwen2.5-3B → §14.</p>')

block="\n".join(o)
raw=HTMLF.read_text(encoding="utf-8")
placeholder='<p class="muted">C (fragility ranking: which capability fails first under global removal) and D (15×15 interlink) pending GPU ablation — to be appended here.</p>'
if placeholder in raw:
    raw=raw.replace(placeholder,block,1); act="replaced placeholder"
elif 'Deliverable C — Global stress' not in raw:
    raw=raw.replace("</body></html>",block+"\n</body></html>",1); act="appended (no placeholder)"
else:
    act="already present, skipped"
HTMLF.write_text(raw,encoding="utf-8")
print("HTML",act,"len",len(raw))
print("fragility(NLL):",ranking[:4],"...")
print("diag self-damage worst:",diag_hi[:3])
print("off-diag collateral worst:",off[:3])
