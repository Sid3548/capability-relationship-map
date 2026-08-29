# Capability Removal-Envelope Experiment

How far can you remove MLP neurons from a small language model before a given
capability breaks — and when you remove one capability's neurons, what happens
to the others?

This started as a weekend exploration and turned into something bigger than I
expected. It's a work in progress and a learning project, not a finished result,
and I'm writing this down partly so I understand exactly what was run.

## What it does

- **Neuron** = one channel of the MLP intermediate `h = silu(gate_proj(x)) * up_proj(x)`
  (the input to `down_proj`), scored by **mean |h|** over answer-token positions.
- **Removal** = reversible **mean-ablation**: the neuron is pinned to its
  calibration-mean activation via a forward pre-hook on `down_proj`. No weights
  are changed — it's a measurement probe, not deployment pruning.
- **15 capabilities × 40 prompts** each (`data/batteries/comprehensive/`).
- Four analyses:
  - **A** — overlap of each capability's top-20% active neurons (*observational*).
  - **B** — neuron allocation: dead / exclusive / shared / core.
  - **C** — global stress sweep: mean-ablate the lowest-activity neurons in 0.1%
    steps up to 70%, measuring all 15 capabilities' teacher-forced NLL at each step.
  - **D** — 15×15 interlink: remove each capability's own lowest-5% envelope,
    measure the effect on all 15.

**Important detail about C:** the removal order comes from the *average* of the 15
capabilities' per-layer percentile ranks, so C is a **single global sweep**, not 15
per-capability sweeps. Per-capability ranking is used only in **D**, and only at a
single 5% depth. The clean per-capability cumulative sweep is a separate, not-yet-run
experiment.

## Status

- **Qwen2.5-0.5B** — A–D complete.
- **Qwen2.5-3B** — A–D complete.
- **Llama-3.1-8B (int8)** — baselines, A, B, C complete; smoke + int8/fp16 fidelity
  check passed. **D (interlink) is incomplete**: the run was interrupted after 2 of
  15 capabilities, so `results/comprehensive_llama31_8b/D_interlink.json` has real
  rows only for `coding` and `math` — the other 13 rows are zero placeholders, not
  measurements. Don't use the Llama D matrix.

## Layout

- `scripts/` — pipeline. `run_full.py` drives A–D; `smoke_*`, `fidelity_*`, `emit_*` are helpers.
- `src/` — `capture`, `scoring`, `ablation`, `eval`, `hooks`, `models`.
- `data/batteries/` — prompt sets.
- `results/` — per-model outputs (JSON/JSONL + `.npz` activation aggregates + PNGs).

## Reproduce

```
python -m scripts.run_full <model_name> <outdir> [max_fraction]
# e.g. python -m scripts.run_full Qwen/Qwen2.5-0.5B results/comprehensive_qwen0.5b 0.70
```
Stack: torch 2.6 + cu124, transformers, a single RTX 4060 Ti 16GB. Baselines and
activation aggregates are cached, so a rerun resumes rather than recomputing.

## Caveats

- Mean-ablation is one intervention type; zero and resampling controls aren't in yet.
- 40 prompts per capability is small — trust the NLL trends over single accuracy cells.
- A is correlational; C and D are causal.
- Neurons are polysemantic, so a "neuron" here is not a clean feature — this is
  neuron-level on purpose, because neurons are the unit you can actually prune.
