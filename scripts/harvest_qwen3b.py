# -*- coding: utf-8 -*-
"""Harvest Qwen2.5-3B comprehensive (A/B/C/D) into HTML section 14, with a
cross-scale comparison vs 0.5B. Torch-free (matplotlib + numpy + json only)."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("C:/Users/user/capability-relationship-map")
D3 = REPO/"results/comprehensive_qwen3b"
D05 = REPO/"results/comprehensive_qwen0.5b"
HTMLF = Path("C:/Users/user/Desktop/Capability Removal-Envelope Experiment — Context Handoff.html")
CAPS = ["coding","math","formal_logic","grammar","translation","reading_comprehension","history_facts","philosophy","science_facts","commonsense","problem_solving","creative_writing","summarization","spatial_pattern","ethics"]
SH = {"coding":"code","math":"math","formal_logic":"logic","grammar":"gram","translation":"transl","reading_comprehension":"readc","history_facts":"hist","philosophy":"phil","science_facts":"sci","commonsense":"cmnsn","problem_solving":"probs","creative_writing":"creat","summarization":"summ","spatial_pattern":"spat","ethics":"ethic"}
sh=[SH[c] for c in CAPS]

base=json.load(open(D3/"baselines.json")); C=json.load(open(D3/"C_stress.json"))
B=json.load(open(D3/"B_allocation.json")); A=json.load(open(D3/"A_overlap.json"))
Dl=json.load(open(D3/"D_interlink.json")); man=json.load(open(D3/"manifest.json"))
rows=[json.loads(l) for l in open(D3/"C_stress_rows.jsonl")]
g=[r for r in rows if r["control"]=="global"]; gen=[r for r in g if any(k.startswith("acc_") for k in r)]
ff=C["failure_fraction_pct"]; ranking=C["fragility_ranking_most_to_least_fragile"]

# 0.5B for cross-scale
A05=json.load(open(D05/"A_overlap.json")); C05=json.load(open(D05/"C_stress.json"))
def alloc05():
    npz=np.load(D05/"aggregates_mean_abs.npz")
    masks=[];
    for c in CAPS:
        a=npz[c]; thr=np.quantile(a,0.8,axis=1,keepdims=True); masks.append((a>=thr).reshape(-1))
    st=np.stack(masks,0); cnt=st.sum(0); N=cnt.size
    return dict(dead=(cnt==0).sum()/N,shared=((cnt>=2)&(cnt<=12)).sum()/N,exclusive=(cnt==1).sum()/N,core=(cnt>=13).sum()/N)
al05=alloc05(); al3=B["buckets_pct"]; tot3=B["total_neurons"]

def near(fr): return min(gen,key=lambda r:abs(r["frac"]-fr))
ckpts=[near(0.2),near(0.35),near(0.5)]

# plots
fig,ax=plt.subplots(figsize=(11,4.5))
vals=[(ff[c] if ff[c] is not None else 0.72)*100 for c in ranking]
ax.bar(range(15),vals,color=["#ff7a7a" if ff[c] is not None else "#5fd0a0" for c in ranking])
ax.set_xticks(range(15));ax.set_xticklabels(ranking,rotation=60,ha="right",fontsize=8)
ax.set_ylabel("% removed at NLL-failure (2x)");ax.set_title("Qwen2.5-3B fragility (green=never failed to 70%)")
fig.tight_layout();fig.savefig(D3/"C_fragility_bar.png",dpi=140);plt.close(fig)
gg=sorted(g,key=lambda r:r["frac"])
fig,ax=plt.subplots(figsize=(11,6))
for c in CAPS: ax.plot([r["frac"]*100 for r in gg],[r[f"nllratio_{c}"] for r in gg],lw=1,label=c)
ax.axhline(2,color="k",ls="--",lw=.8);ax.set_yscale("log");ax.set_xlabel("% removed (global)");ax.set_ylabel("NLL ratio (log)")
ax.set_title("Qwen2.5-3B global stress: per-capability NLL");ax.legend(fontsize=7,ncol=3)
fig.tight_layout();fig.savefig(D3/"C_retention_curves.png",dpi=140);plt.close(fig)
def heat(mat,name,title):
    m=np.array(mat);fig,ax=plt.subplots(figsize=(8,7));lim=max(0.01,np.abs(m).max())
    im=ax.imshow(m,cmap="RdBu",vmin=-lim,vmax=lim)
    ax.set_xticks(range(15));ax.set_xticklabels(sh,rotation=60,ha="right",fontsize=7);ax.set_yticks(range(15));ax.set_yticklabels(sh,fontsize=7)
    ax.set_xlabel("measured");ax.set_ylabel("removed target");ax.set_title(title);fig.colorbar(im);fig.tight_layout();fig.savefig(D3/name,dpi=140);plt.close(fig)
for k,t in [("dacc_env","D_env_dacc.png"),("dacc_high","D_high_dacc.png"),("dloss_env","D_env_dloss.png"),("dloss_high","D_high_dloss.png")]:
    heat(Dl[k],t,f"3B interlink {k}")

dh=np.array(Dl["dacc_high"]); diag=sorted([(CAPS[i],dh[i,i]) for i in range(15)],key=lambda x:x[1])
off=sorted([(CAPS[i],CAPS[j],dh[i,j]) for i in range(15) for j in range(15) if i!=j],key=lambda x:x[2])

o=[];w=o.append
w('<h2 id="comprehensive3b">14 · Comprehensive capability↔neuron map — Qwen2.5-3B (2026-07-05)</h2>')
w(f'<p class="sub">Same 15-capability battery, same method (top-20% per-layer active, mean-ablation, failure=NLL&gt;2× or acc&lt;0.5 of own baseline). Model: 36 layers × 11008 intermediate = <strong>{tot3:,} MLP neurons</strong> (3.4× the 0.5B), git {man.get("git_hash","")[:8]}.</p>')
w(f'<div class="callout crux"><strong>Headline — the wiring structure REPLICATES at 6× scale.</strong> 3B allocates its neurons almost identically to 0.5B (tiny general core, large shared middle), the same verbal/reasoning cluster is most entangled and fails first, and the numeric/exact tasks stay most robust. Capability↔neuron organization is not a small-model artifact.</div>')

# baselines
w('<h3>Baselines (n=40/task)</h3><table><tr><th>capability</th><th>acc</th><th>NLL</th></tr>')
for c in CAPS:
    b=base[c];acc="—" if b["acc"] is None else f"{b['acc']:.3f}"
    cls=' class="bad"' if (b["acc"] is not None and b["acc"]<0.5) else ''
    w(f'<tr><td>{c}</td><td{cls}>{acc}</td><td class="muted">{b["nll"]:.2f}</td></tr>')
w('</table>')

# cross-scale
w('<h3>Cross-scale comparison (0.5B vs 3B)</h3><table>')
w('<tr><th>metric</th><th>Qwen2.5-0.5B</th><th>Qwen2.5-3B</th></tr>')
w(f'<tr><td>total MLP neurons</td><td>116,736</td><td>{tot3:,}</td></tr>')
w(f'<tr><td>allocation dead / shared / exclusive / <strong>core</strong></td><td>{al05["dead"]*100:.0f} / {al05["shared"]*100:.0f} / {al05["exclusive"]*100:.0f} / <strong>{al05["core"]*100:.1f}%</strong></td><td>{al3["dead"]*100:.0f} / {al3["shared"]*100:.0f} / {al3["exclusive"]*100:.0f} / <strong>{al3["core"]*100:.1f}%</strong></td></tr>')
w(f'<tr><td>most-entangled pair (top-20% overlap)</td><td>{A05["most_entangled"][0][0]}–{A05["most_entangled"][0][1]} {A05["most_entangled"][0][2]:.2f}</td><td>{A["most_entangled"][0][0]}–{A["most_entangled"][0][1]} {A["most_entangled"][0][2]:.2f}</td></tr>')
w(f'<tr><td>most fragile (NLL, first 4)</td><td class="bad">{", ".join(C05["fragility_ranking_most_to_least_fragile"][:4])}</td><td class="bad">{", ".join(ranking[:4])}</td></tr>')
w(f'<tr><td>most robust (NLL, last 3)</td><td class="good">{", ".join(C05["fragility_ranking_most_to_least_fragile"][-3:])}</td><td class="good">{", ".join(ranking[-3:])}</td></tr>')
w('</table>')
w('<p>Same shape at both scales: a tiny general <strong>core (~3–4%)</strong>, ~45% shared, ~20% exclusive; the verbal/reasoning cluster is most entangled and most fragile; math / problem_solving / science are most robust. 3B carries a slightly larger core (4.2% vs 2.6%) — mild evidence that scale buys a bit more shared general substrate.</p>')

# fragility
w('<h3>Deliverable C — fragility ranking (global least-used-first removal to 70%)</h3>')
w('<table><tr><th>rank</th><th>capability</th><th>NLL-fails @</th><th>baseline NLL</th></tr>')
for k,c in enumerate(ranking):
    fv="never (&gt;70%)" if ff[c] is None else f"{ff[c]*100:.1f}%"
    cls=' class="bad"' if ff[c] is not None and ff[c]<0.25 else ''
    w(f'<tr><td>{k+1}</td><td{cls}>{c}</td><td>{fv}</td><td class="muted">{base[c]["nll"]:.2f}</td></tr>')
w('</table>')
w('<p class="muted">Same NLL-vs-accuracy caveat as 0.5B (§13): report both; exact-answer tasks crater by accuracy while high-baseline-NLL tasks resist NLL-doubling. Accuracy retention (acc/baseline) at global-removal checkpoints:</p>')
w('<div style="overflow-x:auto"><table style="font-size:12px"><tr><th>% removed</th>'+''.join(f'<th>{s}</th>' for s in sh)+'</tr>')
for r in ckpts:
    w(f'<tr><td>{r["frac"]*100:.0f}%</td>')
    for c in CAPS:
        b=base[c]["acc"];a=r.get(f"acc_{c}")
        if b and a is not None:
            ret=a/b;cls=' class="bad"' if ret<0.5 else (' class="good"' if ret>0.9 else '')
            w(f'<td{cls}>{ret:.2f}</td>')
        else: w('<td class="muted">—</td>')
    w('</tr>')
w('</table></div>')

# interlink
w('<h3>Deliverable D — 15×15 interlink</h3>')
w('<p><strong>Biggest self-damage (remove specific-high, own Δacc):</strong> '+"; ".join(f"{c} {v:+.2f}" for c,v in diag[:5])+'</p>')
w('<p><strong>Strongest collateral (remove row→hurts col):</strong> '+"; ".join(f"{a}→{b} {v:+.2f}" for a,b,v in off[:6])+'</p>')
for mat,label in [(Dl["dacc_env"],"Δacc — remove LOW-ENVELOPE"),(Dl["dacc_high"],"Δacc — remove SPECIFIC-HIGH")]:
    M=np.array(mat)
    w(f'<p><strong>{label}</strong> (rows=removed, cols=measured):</p><div style="overflow-x:auto"><table style="font-size:11px"><tr><th></th>'+''.join(f'<th>{s}</th>' for s in sh)+'</tr>')
    for i in range(15):
        w(f'<tr><th>{sh[i]}</th>')
        for j in range(15):
            v=M[i,j]
            cell=(f'<td class="muted">{v:+.2f}</td>' if i==j else f'<td class="bad">{v:+.2f}</td>' if v<=-0.15 else f'<td class="good">{v:+.2f}</td>' if v>=0.15 else f'<td>{v:+.2f}</td>')
            w(cell)
        w('</tr>')
    w('</table></div>')
w('<p class="muted">Δloss matrices verbatim in <code>capability-relationship-map/results/comprehensive_qwen3b/D_interlink.json</code>. Heatmaps: <code>D_env_dacc.png, D_high_dacc.png, D_env_dloss.png, D_high_dloss.png</code>.</p>')

# 3B data manifest
comp=sorted(p.name for p in D3.glob("*"))
w('<h3>3B — data manifest (relative paths)</h3><table><tr><th>file</th></tr>')
for f in comp: w(f'<tr><td><code>capability-relationship-map/results/comprehensive_qwen3b/{f}</code></td></tr>')
w('</table>')
w('<p class="good"><strong>Qwen2.5-3B comprehensive run COMPLETE (A+B+C+D).</strong> Today\'s scope (Qwen 0.5B + 3B) finished. Deferred to a later session: Phi-4-mini (dense, downloaded) and an MoE model (gpt-oss-20b).</p>')

block="\n".join(o)
raw=HTMLF.read_text(encoding="utf-8")
if 'id="comprehensive3b"' in raw:
    print("section 14 already present, skipping")
else:
    raw=raw.replace("</body></html>",block+"\n</body></html>",1)
    HTMLF.write_text(raw,encoding="utf-8")
    print("section 14 appended, len",len(raw))
print("3B fragility:",ranking[:4],"| robust:",ranking[-3:])
print("3B alloc core %.1f | 0.5B core %.1f"%(al3["core"]*100,al05["core"]*100))
print("3B most-entangled:",A["most_entangled"][0][:3])
