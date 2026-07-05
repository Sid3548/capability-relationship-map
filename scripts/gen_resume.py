# -*- coding: utf-8 -*-
"""Insert/refresh a RESUME/RESTART handoff section (incl. a copy-paste startup
prompt) near the top of the Desktop context HTML. Idempotent via HTML markers."""
import html, time
from pathlib import Path

HTMLF = Path("C:/Users/user/Desktop/Capability Removal-Envelope Experiment — Context Handoff.html")
STAMP = time.strftime("%Y-%m-%d %H:%M")

STARTUP_PROMPT = """You are resuming the Capability Removal-Envelope COMPREHENSIVE experiment on this Windows PC (RTX 4060 Ti, 16GB). Read this whole HTML first: C:\\Users\\user\\Desktop\\Capability Removal-Envelope Experiment - Context Handoff.html  (esp. this RESUME section + sections 12-13).

REPO: capability-relationship-map/  (all data paths below are RELATIVE to this folder). On this machine it lives at C:\\Users\\user\\capability-relationship-map ; run shell commands from your home dir C:\\Users\\user.  VENV: .venv  (torch 2.6.0+cu124, transformers 5.13.0, Python 3.12).

HARD RULES:
- YOU (orchestrator) do ALL writing/HTML/analysis directly via Bash + python scripts. Do NOT spend sub-agent tokens on writing; if you use a sub-agent it computes only and dumps to disk.
- CRITICAL BUG: pyarrow parquet write SEGFAULTS (exit 139) whenever a torch-CUDA context is live on this box. In ANY torch process write JSON/JSONL/npz ONLY. Make PNGs/parquet AFTERWARD in a separate torch-free process. KMP_DUPLICATE_LIB_OK does NOT fix it.
- One GPU -> run models SEQUENTIALLY. A running job uses ~8GB; never stack jobs summing >~14GB (OOM).
- Log ALL results VERBATIM into the HTML: section 13=Qwen0.5B, 14=Qwen3B, 15=Phi-4-mini, 16=gpt-oss-20b. Completeness over polish. Use scripts/gen_resume.py to refresh THIS section after each milestone.

METHOD: neuron = one channel of MLP intermediate silu(gate)*up (= down_proj input); read dims from config, never hardcode. "active" for a capability = mean_abs rank in TOP 20% within its own layer (per-layer percentile; +10/30% sensitivity). Ablation = mean-ablate via reversible hooks (zero-ablate as bound). FAILURE per task = accuracy_retention<0.5 OR NLL>2x that task's OWN baseline. ALWAYS report NLL and accuracy together: they diverge (NLL-doubling is confounded by baseline confidence; exact-answer tasks die by accuracy while confident MCQ tasks die by NLL).

BATTERIES: data/batteries/comprehensive/*.jsonl  (15 caps x 40 prompts; built by scripts/build_comprehensive.py).
DELIVERABLES per model: A=15x15 active-set overlap; B=neuron allocation (dead/core/shared/exclusive-per-cap); C=global stress sweep to 70% least-used-first + random control -> fragility ranking (which cap fails first); D=15x15 interlink (remove each cap's low-envelope & specific-high, eval all 15).

MODEL LOADING GOTCHA: HF hub revalidation HANGS (unauthenticated HEAD requests) and offline-by-id can misresolve. ALWAYS load from the LOCAL snapshot dir with offline env vars. Get path: SNAP=$(ls -d ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/ | head -1)

RUN A DENSE MODEL (resume-safe; reuses baselines.json/aggregates if present):
  cd capability-relationship-map
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/Scripts/python.exe -u -m scripts.run_full "$SNAP" results/comprehensive_<tag> 0.70  > logs/<tag>.log 2>&1
Resume just C+D for 0.5B (aggregates already captured):
  .venv/Scripts/python.exe -u -m scripts.run_cd Qwen/Qwen2.5-0.5B results/comprehensive_qwen0.5b 0.70

PLAN (2026-07-05): TODAY = QWEN ONLY (0.5B done; 3B finishing = last run of the day, then STOP). DEFERRED to a later session: Phi-4-mini-instruct [phi4mini] (dense, already downloaded) and an MoE model gpt-oss-20b [gptoss20b] (needs HF token + free disk + a MoE hook adapter). To continue later, run Phi then gpt-oss with run_full / the MoE adapter.
gpt-oss is MoE -> needs an adapter first: neuron = per-expert silu(gate)*up channel (see modeling_gpt_oss.py GptOssExperts.forward line ~114 `gated_output`); capture+ablate by monkeypatching GptOssExperts.forward (experts are fused tensors, not modules; routing-aware: a neuron only fires when its expert is in the top-4). ~24x32x2880 ~= 2.2M neurons; run A+B first (fast, one capture pass), then C+D (slow).

AFTER each run: read the JSON outputs (A_overlap.json, B_allocation.json, C_stress.json, C_stress_rows.jsonl, D_interlink.json) and APPEND a new HTML section with the overlap matrix, allocation table, fragility ranking, and interlink matrices, verbatim. Then run scripts/gen_resume.py to update status."""

esc = html.escape(STARTUP_PROMPT)

BLOCK = f"""<!--RESUME_START-->
<div class="callout warn" style="border-color:#ff7a7a">
<h2 style="margin-top:0;border:none" id="resume">⚡ RESUME / RESTART HANDOFF <span class="muted" style="font-size:13px">(updated {STAMP})</span></h2>
<p><strong>If this process stopped, paste the startup prompt below into a fresh Claude Code session in the repo. Everything needed to continue is on disk + in this file.</strong> The experiment is long-running (multi-model, hours each); nothing is lost on a stop because every deliverable checkpoints to JSON on disk.</p>

<h3>Copy-paste startup prompt</h3>
<pre>{esc}</pre>

<h3>Live status snapshot</h3>
<table>
<tr><th>Model</th><th>tag / results dir</th><th>A overlap</th><th>B alloc</th><th>C fragility</th><th>D interlink</th><th>HTML sec</th></tr>
<tr><td>Qwen2.5-0.5B</td><td class="muted">qwen0.5b</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td>13 ✅</td></tr>
<tr><td>Qwen2.5-3B</td><td class="muted">qwen3b</td><td colspan="4" class="good">RUNNING — finishing today (last model for today)</td><td>14</td></tr>
<tr><td>Phi-4-mini-instruct</td><td class="muted">phi4mini</td><td colspan="4" class="muted">DEFERRED (downloaded, ready) — not part of today's scope</td><td>15</td></tr>
<tr><td>gpt-oss-20b (MoE)</td><td class="muted">gptoss20b</td><td colspan="4" class="muted">DEFERRED to a later session — the MoE model; needs HF token + disk + MoE adapter</td><td>16</td></tr>
</table>
<p class="muted">Verify against disk: <code>git -C capability-relationship-map log --oneline</code> and <code>ls results/comprehensive_*/</code> show the true latest state (this snapshot may lag).</p>

<h3>Estimated time (single GPU, sequential) — as of {STAMP}</h3>
<table>
<tr><th>Model</th><th>est. compute</th><th>ETA</th></tr>
<tr><td>Qwen2.5-0.5B</td><td class="good">done (~1.5h)</td><td class="good">✅ complete</td></tr>
<tr><td>Qwen2.5-3B (running)</td><td>C ~2.5h + D ~3h ≈ 5h</td><td>~20:30 (8:30pm) today</td></tr>
<tr><td>Phi-4-mini</td><td>~5h</td><td>~01:30 (after 3B)</td></tr>
<tr><td>gpt-oss-20b (MoE)</td><td>adapter ~2-4h + run ~10-16h</td><td>next day</td></tr>
</table>
<p class="muted">Dense set (through Phi) ~9-11h from ~16:00; full set incl. gpt-oss ~24-36h. Speed lever: drop max_fraction 0.70→0.50 + coarser gen checkpoints for Phi/gpt-oss (model broadly broken by ~35% anyway) ≈ halves their runtime.</p>

<h3>File map</h3>
<ul>
<li><strong>Repo:</strong> <code>capability-relationship-map</code> (git) · <strong>venv:</strong> <code>.venv</code></li>
<li><strong>Pipeline:</strong> <code>scripts/run_full.py</code> (full A-D, JSON-only, dense) · <code>scripts/run_cd.py</code> (C+D reuse aggregates) · <code>scripts/build_comprehensive.py</code> (batteries) · <code>scripts/gen_resume.py</code> (this section)</li>
<li><strong>Core src:</strong> <code>src/models.py, hooks.py, capture.py, scoring.py, ablation.py, eval/*</code></li>
<li><strong>Results:</strong> <code>results/comprehensive_&lt;tag&gt;/</code> → baselines.json, aggregates_mean_abs.npz, A_overlap.json, B_allocation.json, C_stress.json, C_stress_rows.jsonl, D_interlink.json, manifest.json</li>
<li><strong>Logs:</strong> <code>logs/*.log</code> · earlier 0.5B smoke/cliff/4-task results in <code>results/</code></li>
</ul>

<h3>Non-negotiable gotchas</h3>
<ul>
<li class="bad">pyarrow write segfaults under live torch-CUDA → JSON only in torch process; plots/parquet in a separate torch-free step.</li>
<li>One GPU (16GB): sequential runs; ~8GB per job; don't stack &gt;~14GB.</li>
<li>Log everything verbatim to this HTML; orchestrator does all writing (no sub-agent tokens on writing).</li>
<li>Report NLL <em>and</em> accuracy — the fragility metric depends on which you use (see §13).</li>
</ul>
</div>
<!--RESUME_END-->
"""

raw = HTMLF.read_text(encoding="utf-8")
import re
if "<!--RESUME_START-->" in raw:
    raw = re.sub(r"<!--RESUME_START-->.*?<!--RESUME_END-->", lambda m: BLOCK.strip(), raw, flags=re.DOTALL)
    action = "updated"
else:
    anchor = '<div class="panel toc">'
    raw = raw.replace(anchor, BLOCK + "\n" + anchor, 1)
    action = "inserted"
HTMLF.write_text(raw, encoding="utf-8")
print(f"RESUME section {action}; file len={len(raw)}")
