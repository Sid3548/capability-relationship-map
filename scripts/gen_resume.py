# -*- coding: utf-8 -*-
"""Insert/refresh the RESUME/RESTART handoff section (detailed conversation-autostart
prompt + findings + status + file map) near the top of the Desktop context HTML.
Idempotent via HTML markers."""
import html, time, re
from pathlib import Path

HTMLF = Path("C:/Users/user/Desktop/Capability Removal-Envelope Experiment — Context Handoff.html")
STAMP = time.strftime("%Y-%m-%d %H:%M")

AUTOSTART = """AUTOSTART PROMPT — paste this into a fresh Claude Code session to resume Siddharth's "Capability Removal-Envelope" experiment exactly where we left off. (Open/point it at this HTML file too.)

=== WHO / WORKING AGREEMENT ===
This is Sid's interpretability research. You are the driver: YOU do ALL writing / analysis / HTML directly via Bash + python scripts in the repo. If you use a sub-agent it COMPUTES ONLY and dumps to disk (minimal output tokens) — never spend sub-agent tokens on writing. Log ALL results VERBATIM into this HTML (sections; relative paths capability-relationship-map/...). Be honest about caveats. One GPU (RTX 4060 Ti 16GB) => models run SEQUENTIALLY. Read this whole HTML first, especially sections 12 (coding cliff), 13 (0.5B comprehensive), 14 (3B comprehensive).

=== THE EXPERIMENT ===
How far can MLP neurons be deleted from a small LLM while keeping ONE capability intact, and which skills share wiring / fail first. Removal-envelope direction (max removable set that PRESERVES a skill) — NOT fragility. Neuron = one channel of MLP intermediate silu(gate)*up (= down_proj input); read dims from config, never hardcode. "Active" for a capability = mean_abs rank in TOP 20% within its own layer (per-layer percentile; SiLU scale unbounded => raw cutoff not comparable; +10/30% sensitivity). Ablation = mean-ablate via reversible hooks (zero-ablate as bound). FAILURE per task = accuracy_retention<0.5 OR NLL>2x that task's OWN baseline. ALWAYS report NLL and accuracy together (they diverge — exact-answer tasks die by accuracy; confident MCQ tasks die by NLL; NLL-doubling is confounded by baseline confidence). Battery: data/batteries/comprehensive/*.jsonl (15 capabilities x 40 prompts). Deliverables per model: A=15x15 active-set overlap; B=allocation (dead/core/shared/exclusive per cap); C=global stress sweep to 70% least-used-first + random control -> fragility ranking; D=15x15 interlink.

=== STATUS (2026-07-05) ===
DONE: Qwen2.5-0.5B and Qwen2.5-3B — full A+B+C+D, logged sections 13 & 14, with a 0.5B->3B cross-scale comparison. Also earlier: 0.5B smoke envelope + coding cliff + 4-task interlink (sections 11-12).
DEFERRED (Sid's call): Phi-4-mini-instruct (dense, Western; already downloaded) and gpt-oss-20b (the MoE, Western; needs HF token + free disk + a MoE hook adapter).

=== FINDINGS SO FAR ===
- Removal envelope: coding retains competence far out — strict pass@1 cliff ~25% of MLP removed; smooth-loss competence barely bends to ~50%.
- Allocation: only ~3-4% of neurons are general "core"; ~45% shared across a few skills; ~20% exclusive to one skill; ~25-30% ~dead at this threshold.
- Overlap: a big shared VERBAL/REASONING cluster (grammar, reading-comp, commonsense, ethics, logic, philosophy, spatial, summarization) overlaps 0.6-0.85; a separate numeric pair math-problem_solving ~0.70; creative_writing most isolated.
- Removed-neuron identity: coding's low-activation "envelope" neurons are NOT dead — ~95% fire harder on OTHER tasks; you strip neurons the other skills own.
- Fragility (which fails first under global removal): the verbal/shared-wiring cluster fails first (by NLL); math/problem_solving/science most robust. Same at 0.5B and 3B.
- CROSS-SCALE: structure REPLICATES 0.5B->3B (6x): near-identical allocation, same most-entangled cluster, same fragility order. 3B core slightly larger (4.2% vs 2.6%).

=== INFRA GOTCHAS (do not re-hit) ===
1. pyarrow parquet write SEGFAULTS (exit 139) with a live torch-CUDA context on this Windows box. In ANY torch process write JSON/JSONL/npz ONLY; make PNGs/parquet in a separate torch-free process. KMP_DUPLICATE_LIB_OK does NOT fix it.
2. HF from_pretrained HANGS on "Fetching N files" (unauthenticated hub revalidation) even when cached; offline-by-id can misresolve. ALWAYS load from the LOCAL snapshot dir with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1. SNAP=$(ls -d ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/ | head -1)
3. Launch long GPU runs with a harness-tracked background job, NOT a shell & (detached => no completion notification).
4. DISK is ~95% full. Do NOT retry snapshot_download in a kill/relaunch loop — it restarts partial .incomplete blobs and filled the disk (~22GB garbage once). Clean stray *.incomplete from ~/.cache/huggingface/hub/*/blobs. gpt-oss needs ~13GB.

=== NEXT ACTIONS (in order, when resuming) ===
1) Phi-4-mini (dense, ready): SNAP=$(ls -d ~/.cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/*/ | head -1); HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/Scripts/python.exe -u -m scripts.run_full \"$SNAP\" results/comprehensive_phi4mini 0.70 > logs/phi4mini.log 2>&1 (run_in_background). Then harvest a new HTML section 15 (copy scripts/harvest_qwen3b.py pattern) + cross-family note.
2) gpt-oss-20b (MoE): first get Sid's HF_TOKEN and free disk. Build a MoE adapter: neuron = per-expert silu(gate)*up channel; capture+ablate by monkeypatching GptOssExperts.forward (experts are FUSED tensors, not modules; routing-aware — a neuron only fires when its expert is in the top-4). 24x32x2880 ~= 2.2M neurons. Run A+B first (one fast capture pass), then C+D. Harvest section 16.

=== PIPELINE / COMMANDS ===
cd (repo).  Dense model full run (resume-safe, JSON-only): HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/Scripts/python.exe -u -m scripts.run_full \"$SNAP\" results/comprehensive_<tag> 0.70 > logs/<tag>.log 2>&1
Harvest to HTML: adapt scripts/harvest_qwen3b.py (torch-free plots + tables + manifest). Then: python scripts/gen_resume.py ; python scripts/relativize_html.py ; git add -A && git commit.
Speed lever: max_fraction 0.70->0.50 halves C runtime (model broadly broken by ~35% anyway)."""

esc = html.escape(AUTOSTART)

BLOCK = f"""<!--RESUME_START-->
<div class="callout warn" style="border-color:#ff7a7a">
<h2 style="margin-top:0;border:none" id="resume">⚡ RESUME / RESTART HANDOFF <span class="muted" style="font-size:13px">(updated {STAMP})</span></h2>
<p><strong>To continue this project later, paste the AUTOSTART PROMPT below into a fresh Claude Code session</strong> (and open this HTML). It carries the full context — working agreement, method, findings, gotchas, and the exact next steps. Every deliverable is checkpointed to JSON on disk, so nothing is lost on a stop.</p>

<h3>AUTOSTART PROMPT (copy-paste to resume the whole project)</h3>
<pre>{esc}</pre>

<h3>Live status</h3>
<table>
<tr><th>Model</th><th>tag</th><th>A</th><th>B</th><th>C</th><th>D</th><th>HTML sec</th></tr>
<tr><td>Qwen2.5-0.5B</td><td class="muted">qwen0.5b</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td>13 ✅</td></tr>
<tr><td>Qwen2.5-3B</td><td class="muted">qwen3b</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td class="good">done</td><td>14 ✅</td></tr>
<tr><td>Phi-4-mini-instruct</td><td class="muted">phi4mini</td><td colspan="4" class="muted">DEFERRED (downloaded, ready) — dense Western</td><td>15</td></tr>
<tr><td>gpt-oss-20b (MoE)</td><td class="muted">gptoss20b</td><td colspan="4" class="muted">DEFERRED — needs HF token + disk + MoE adapter</td><td>16</td></tr>
</table>
<p class="muted">Truth is on disk: <code>git -C capability-relationship-map log --oneline</code> and <code>ls results/comprehensive_*/</code>. Repo root on this machine: C:\\Users\\user\\capability-relationship-map ; run shell commands from C:\\Users\\user.</p>

<h3>Findings so far (one-glance)</h3>
<ul>
<li><strong>Envelope:</strong> coding strict-cliff ~25% of MLP removed; smooth competence bends only ~50%.</li>
<li><strong>Allocation:</strong> ~3-4% general core, ~45% shared, ~20% exclusive, ~25-30% dead (top-20% threshold).</li>
<li><strong>Removed neurons aren't dead</strong> — ~95% of coding's envelope fire harder on other tasks (cross-task specialization).</li>
<li><strong>Fragility:</strong> verbal/shared-wiring cluster fails first; math/problem_solving/science most robust.</li>
<li><strong class="good">Cross-scale:</strong> structure REPLICATES 0.5B→3B (6×) — same allocation, same entangled cluster, same fragility order.</li>
</ul>

<h3>File map</h3>
<ul>
<li><strong>Repo:</strong> <code>capability-relationship-map/</code> (git; on this machine C:\\Users\\user\\capability-relationship-map) · <strong>venv:</strong> <code>.venv</code> (torch 2.6.0+cu124, transformers 5.13.0)</li>
<li><strong>Pipeline:</strong> <code>scripts/run_full.py</code> (full A-D, JSON-only, dense) · <code>run_cd.py</code> (C+D reuse) · <code>build_comprehensive.py</code> (15×40 batteries) · <code>harvest_qwen3b.py</code> / <code>harvest_05b_cd.py</code> (HTML harvest+plots) · <code>gen_resume.py</code> (this section) · <code>relativize_html.py</code></li>
<li><strong>Results:</strong> <code>results/comprehensive_&lt;tag&gt;/</code> → baselines.json, aggregates_mean_abs.npz, A_overlap.json, B_allocation.json, C_stress.json, C_stress_rows.jsonl, D_interlink.json, manifest.json, *.png</li>
</ul>

<h3>Non-negotiable gotchas</h3>
<ul>
<li class="bad">pyarrow write segfaults under live torch-CUDA → JSON only in torch process; plots/parquet in a separate torch-free step.</li>
<li class="bad">HF load hangs on hub revalidation → load from LOCAL snapshot path with HF_HUB_OFFLINE=1.</li>
<li class="bad">Disk ~95% full → never kill/relaunch snapshot_download in a loop (piles up .incomplete blobs); clean stray *.incomplete.</li>
<li>One GPU: sequential runs; log everything verbatim to this HTML; orchestrator does all writing.</li>
</ul>
</div>
<!--RESUME_END-->
"""

raw = HTMLF.read_text(encoding="utf-8")
if "<!--RESUME_START-->" in raw:
    raw = re.sub(r"<!--RESUME_START-->.*?<!--RESUME_END-->", lambda m: BLOCK.strip(), raw, flags=re.DOTALL)
    action = "updated"
else:
    anchor = '<div class="panel toc">'
    raw = raw.replace(anchor, BLOCK + "\n" + anchor, 1)
    action = "inserted"
HTMLF.write_text(raw, encoding="utf-8")
print(f"RESUME section {action}; file len={len(raw)}")
