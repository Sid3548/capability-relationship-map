# Llama-3.1-8B int8 pilot: implementation and runtime audit

**Audit status:** complete static/read-only audit; no model was loaded, no GPU computation was started, no Qwen process was started, and no existing source or result file was modified.

**Audit date:** 2026-08-29 (Asia/Calcutta)

**Repository HEAD observed:** `23978ff3674acc25064886367af2fb656ce75a3d`. The worktree is dirty and contains user/agent changes and untracked files; a run manifest must therefore record a source-tree content hash in addition to HEAD. The new runner must never stage or commit the worktree.

## Executive conclusion

The repository contains a useful proof of concept for hooking the input of each Llama MLP `down_proj`, and the installed Transformers source confirms that this input is exactly `SiLU(gate_proj(x)) * up_proj(x)` in the current local environment. That narrow mechanism is reusable after validation and engineering fixes.

The old Llama runner, scoring code, persistence code, configs, and old result artifacts are **not valid implementations of the requested pilot**. They must not be run or used for headline results. Critical defects include separately tokenizing prompt and gold before concatenation; using a constant rounded step rather than the required exact cumulative counts; per-layer-percentile rather than raw global activation ordering; calibration means computed from the evaluation batteries and pooled across all capabilities; no ranking stability analysis; no per-token/per-item dense results; no reliable resume or atomic state tracking; no mask/ranking hashes; insufficient manifesting; unsafe generated-code execution; and no hard protection against Qwen or fractions above 10%.

The appropriate implementation approach is a new, pilot-specific runner and artifact schema in the versioned pilot directory. Reuse only small, reviewed concepts—not old experiment outputs or the old orchestration paths.

## Governing evidence inspected

### Authoritative for current static facts

- `.venv/Lib/site-packages/transformers/models/llama/modeling_llama.py`, installed Transformers implementation. In the installed version, `LlamaMLP.forward` calls:
  `self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`.
  A forward pre-hook on `down_proj` therefore receives the requested post-gate product. This still requires a live assertion after int8 conversion because bitsandbytes can replace linear module classes.
- `D:/hf_models/Meta-Llama-3.1-8B/config.json`: `LlamaForCausalLM`, `model_type=llama`, 32 layers, hidden size 4,096, intermediate size 14,336, `hidden_act=silu`, vocabulary 128,256, source `torch_dtype=bfloat16`.
- `D:/hf_models/Meta-Llama-3.1-8B/model.safetensors.index.json`: 291 indexed tensor keys and four referenced shards; all four referenced shard files exist. Index metadata reports 16,060,522,496 tensor bytes; shard files total 16,060,556,376 bytes including file/container overhead. This is only an index/presence check, not a cryptographic integrity check.
- The current pilot protocol/prompt is authoritative for schedule, interventions, controls, and safety boundaries.

### Provenance only, not authoritative for new scientific findings

- `logs/run_full_llama.log`
- `results/comprehensive_llama31_8b/*`
- `results/llama31_8b_smoke/*`
- `smoke_llama_run.log`
- `fidelity_llama_run.log`
- all old batteries when used in old results

### Reusable only after fixes and new tests

- `src/models.py`: the `ModelBundle.down_proj(layer_idx)` accessor and architecture-dimension container are reasonable seams. Loading logic is not pilot-ready.
- `src/hooks.py`: the forward-pre-hook concept and reversible handle removal pattern are useful. Hook state, assertions, devices, statistics, and replacement modes require redesign.
- `src/ablation.py`: deterministic stable sorting and the concept of layer-count-matched random masks are useful. Core exclusion, schedule construction, device transfer, and persistence are not suitable for the requested primary design.
- `src/eval/base.py`: item-level macro averaging intent, attention masks, and deterministic greedy generation are useful concepts. Token construction and returned schemas must be replaced.
- parser helpers in `src/eval/*`: may be retained only behind task-specific tests. The coding executor is unsafe, and translation/closed-choice metrics do not meet the new protocol.

### Obsolete for this pilot or provenance-only

- `scripts/run_full_llama.py`, `scripts/smoke_llama.py`, `scripts/fidelity_llama.py`: do not execute for the pilot.
- `src/sweep_runner.py`, `src/multitask.py`, `src/comprehensive.py`: orchestration and schemas do not implement the requested experiment.
- `configs/model.yaml`: explicitly points to `Qwen/Qwen2.5-0.5B`; it must never be reachable from the pilot entry point.
- `configs/eval.yaml` and `configs/sweep.yaml`: old two-task smoke configs, only five random seeds, wrong mask schedule implementation downstream, and insufficient metrics/controls.
- `src/storage.py`: read-modify-write Parquet append is neither scalable nor crash-safe; direct writes are not atomic.
- every old Llama result file as evidence for the new go/no-go verdict.

## Static environment and checkpoint inventory

The repository virtual environment contains:

| Component | Observed version |
|---|---:|
| Python environment | repository `.venv` (exact Python version must be captured by the run manifest) |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.13.0 |
| bitsandbytes | 0.50.2 |
| Accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| NumPy | 2.5.1 |
| PyArrow | 24.0.0 |

Read-only GPU inventory at audit time:

- NVIDIA GeForce RTX 4060 Ti, 16,380 MiB total VRAM, compute capability 8.9.
- Driver 591.86; driver-reported CUDA compatibility 13.1.
- The PyTorch wheel is built for CUDA 12.4 (`+cu124`), which is compatible in principle with the newer driver but must be tested by the pilot smoke run.
- About 986 MiB was occupied by graphical desktop processes; no model/compute process was shown by `nvidia-smi`.

Checkpoint facts:

- Path: `D:/hf_models/Meta-Llama-3.1-8B`.
- Four safetensors shards are present, totaling about 16.06 GB on disk.
- The checkpoint config is BF16 and contains no quantization config. Int8 is a runtime bitsandbytes transformation, not a property of the stored checkpoint.
- Architecture implies `32 * 14,336 = 458,752` intervention channels, matching the expected nominal count.
- The directory does not itself establish a remote Hugging Face revision/commit. The pilot must hash the local config, tokenizer files, index, and all four shards and label the identity as a local-file checkpoint unless an independently verifiable revision is present.

## Hook and intervention audit

### Correct static hook location

`src/models.py:43-49` returns `model.model.layers[layer_idx].mlp.down_proj`. The installed Transformers 5.13.0 Llama source confirms the module's input is the canonical gated MLP product. `src/hooks.py:99-133` uses a forward pre-hook and returns a replacement input tuple without modifying weights. This is the correct general intervention site.

### Required live validation

Static source inspection is not enough. After the one authorized int8 load, the pilot must assert and store:

1. Exactly 32 decoder layers and an intermediate width of 14,336 (or halt and document a live discrepancy).
2. Each layer exposes `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj` and calls the observed `down_proj` once per MLP forward.
3. The pre-hook input has shape `[batch, sequence, 14336]` and equals a separately captured `act_fn(gate_proj(x)) * up_proj(x)` on a tiny pass within dtype-appropriate tolerance.
4. Actual module classes after quantization, including the number and names of `Linear8bitLt` or equivalent quantized modules.
5. Every parameter/buffer/module execution device. Halt on unexpected CPU or disk offload.
6. Hook input dtype, replacement dtype, output dtype, quantization compute dtype, and autocast state.
7. The model remains resident and peak/reserved/allocated VRAM is recorded.

### Defects in the current hooks

- `AblationHook` uses a Python `assert` for its mode check (`src/hooks.py:115`); assertions can be disabled. Use explicit typed validation and failure messages.
- No validation checks index range, uniqueness, layer count, channel count, mask nesting, input rank/shape, finite replacement values, or intended dtype.
- Channel indices are copied to the input device on every hook invocation (`src/hooks.py:127`). The complete mean vector is also converted on every invocation before indexing (`src/hooks.py:131`). During generation this happens once per layer per decoding step and is avoidable overhead.
- Every affected MLP input is cloned (`src/hooks.py:126`) even when an empty mask is passed. A persistent hook should fast-path empty masks and keep device-resident, pre-indexed state. Any in-place implementation must first prove it is safe under `torch.no_grad()` and does not mutate aliased tensors.
- Old runners repeatedly attach and remove 32 hooks for every state. The requested design should attach one persistent hook per layer, update its mask/replacement state in place, and remove all hooks once at phase exit.
- If hook attachment fails partway through `attach_ablation_hooks`, already attached handles are not automatically cleaned up. Use a context manager with partial-failure cleanup.
- Only zero and fixed-mean replacement are implemented. There is no resampling/in-distribution intervention.
- The code keeps `mean_values` as float32 CPU tensors and casts repeatedly. Persist float32 calibration artifacts, but stage only selected values on the correct module device/dtype for execution.
- No mask hash or intervention-state audit record is produced.

### Calibration-mean defect

`compute_calibration_means` (`src/capture.py:71-94`) pools every supplied item. The old Llama runner supplies all 15 evaluation batteries (`scripts/run_full_llama.py:115`). This leaks evaluation data into the intervention and produces a capability-pooled mean rather than the requested target-calibration mean. The new pilot requires two separate mean arrays, one derived only from the frozen coding calibration set and one only from the frozen translation calibration set, using a preregistered token aggregation rule.

The old docstring says the mean uses all non-pad positions, but the implementation uses `answer_mask` only (`src/capture.py:85-90`). The new protocol must explicitly decide whether means/ranking statistics use localization answer-token hidden states, predictor positions for answer likelihood, all localization sequence positions, or a separately labeled span. That choice must be frozen before held-out evaluation. Current comments are internally inconsistent: `teacher_forced_nll` notes that hidden state at `t-1` predicts token `t` but returns the unshifted answer-position mask for capture (`src/eval/base.py:114-126`).

## Activation capture and ranking audit

### Current capture limitations

- `CaptureHook` collects only sum of absolute value and maximum absolute value. It does not collect signed mean, second moment/variance, a robust upper quantile, or any comparator score.
- `capture_task_aggregates` first calls `teacher_forced_nll`, which executes a model forward, then executes a second forward for aggregation (`src/capture.py:27-45`). Calibration means repeat the same double-forward pattern. Cached joint tokenization and one capture forward per batch should replace this.
- Items are processed singly, not length-batched.
- Capture state lives on `bundle.device`, assumed to be the string `cuda`; it is not derived from each hooked module. This will fail or silently misrepresent a partially offloaded device map.
- `max_abs` is an unstable extreme statistic and is not a substitute for an upper quantile. Use a documented deterministic streaming/reservoir quantile approximation or omit the quantile with an explicit limitation.
- There is no per-split capture, prompt bootstrap, disjoint-half ranking, layerwise Spearman, or top-k Jaccard.

### Current ordering is not the requested primary ordering

`percentile_rank_per_layer` transforms raw activations to an independent percentile distribution within each layer before global sorting. That forces each layer to contribute similar percentile structure and can change the ordering relative to raw mean absolute activation. The prompt requires a global ranking across every layer/channel using mean absolute post-gate activation. The primary implementation should flatten the raw `[layer, channel]` score array and perform one stable ascending sort, with a documented deterministic tie break based on flat index.

The old code also excludes a constructed “core” set from low/high/random controls. No such exclusion is part of the requested primary pilot. All `N` verified channels must be eligible unless a separately documented control requires otherwise.

Comparator requirements absent from the old code include activation times outgoing-weight magnitude, reliable gradient-times-activation/Taylor, global non-target-conditioned activity, and shuffled-target/label scoring. Outgoing norms must be carefully defined as the norm of the corresponding column of `down_proj`; the implementation must verify how to retrieve or dequantize bitsandbytes weights. If norms are computed from the original BF16 shards instead, state that explicitly and stream layers without retaining another full model.

### Exact schedule defect

Old code uses `round(0.001 * total)` as a constant step (`scripts/run_full_llama.py:181`, `scripts/smoke_llama.py:99`, `src/sweep_runner.py:47`). For `N=458,752`, this is 459 channels per step. The requested exact schedule is:

`k(s) = floor(458752 * s / 1000)`, for `s=1..100`.

Therefore `k(1)=458`, `k(100)=45,875`, and step increments alternate between 458 and 459. The old constant-step method reaches 45,900 at step 100, 25 channels too many. Implement the formula independently at every state; do not accumulate increments. Persist `step`, requested permille, exact `k`, actual count, per-layer counts, added indices, full-mask hash, and parent-mask hash.

## Teacher-forced scoring audit

### Critical boundary-tokenization error

Both `teacher_forced_nll` and `batched_teacher_forced_nll` tokenize the prompt independently, tokenize gold independently with `add_special_tokens=False`, and concatenate token IDs (`src/eval/base.py:81-93` and `139-153`). This is not necessarily the tokenization of `prompt + gold`; BPE tokens can merge differently across the character boundary. Every old NLL and activation span based on these functions is therefore provenance only.

The new scorer must tokenize the complete canonical prompt-plus-answer string jointly and map the frozen character boundary to answer labels using offsets or an equivalently tested method. Prefer canonical prompt formatting that guarantees a tokenizer boundary. Any token spanning the character boundary must trigger a validation failure or an explicitly preregistered policy; it must not be silently assigned.

### Other scoring/schema gaps

- Dense batched scoring returns only `per_item_nll`, a macro mean, and `exp(macro mean NLL)`. It omits item IDs, split labels, capability, answer-token counts, per-token losses/IDs/positions, boundary status, parse/validation status, and checkpoint/mask identity.
- Old result rows preserve only capability means. They cannot support paired prompt bootstrap, reanalysis, confidence intervals, or raw-result verification.
- `eval_loss_battery` reports mean item perplexity, while the batched function reports exponentiated mean item NLL. Those are different quantities. The pilot should make item-level NLL primary and define perplexity ratio only as `exp(NLL_state - NLL_baseline)` on the stated macro aggregation.
- No left-versus-right-padding equivalence test exists. Right padding and attention masks are used, but position IDs, pad behavior, and sequence-length grouping were never tested against single-item scoring.
- No explicit context-window check or truncation prohibition exists.
- Tokenized inputs and gold masks are not cached.
- No repeated deterministic-run test exists. `set_all_seeds` seeds libraries but does not configure/record deterministic algorithm settings.
- The old code calls `log_softmax` over the full vocabulary and materializes a large float32 tensor. Cross-entropy with unreduced labels/masks can reduce peak memory while preserving per-token losses; benchmark both for numerical agreement.
- No schema version exists.

### Required scorer invariants

Before GPU dense execution, tests must prove:

1. Joint token IDs equal tokenizer output for the exact concatenated string.
2. Every selected loss label corresponds to an answer token and no prompt/pad token.
3. The first answer label is predicted by the last prompt-side/prefix position, with a documented policy for special tokens and boundary delimiters.
4. Batched right-padded, batched left-padded (if supported), and single-item NLL agree within tolerance for hand-inspected Latin, Devanagari, Arabic, and Han-script examples.
5. Item macro averaging is invariant to answer length replication in other items; token-weighted summaries, if retained, are separately labeled.
6. Per-token values aggregate exactly to each stored per-item value; per-item values aggregate exactly to capability/split summaries.
7. The 600 held-out and 120 in-sample/calibration diagnostic rows are never combined in the primary mean.

## Functional generation and grading audit

- `greedy_generate` explicitly sets greedy decoding, but the complete resolved generation config and tokenizer special-token settings are not stored. The checkpoint's default generation config is sampling (`do_sample=true`, temperature 0.6, top-p 0.9), so every pilot generation call must override all relevant settings and persist the resolved config.
- The old unified dispatcher returns only aggregate accuracy and parse-failure values to `run_full_llama.py`; per-item generations and grading records are discarded.
- Translation uses thresholded normalized English-style token F1, not SacreBLEU/chrF, and has no language/direction reporting.
- Closed-choice tasks use unconstrained generation and letter extraction instead of restricted-choice probabilities/accuracy.
- Creative writing is NLL-only.
- Generation checkpoints are not represented in a preregistered pilot manifest.

### Critical code-execution safety defect

`src/eval/coding.py:38-70` calls the coding executor a “sandbox,” but it writes model-generated Python to a temporary file and runs it with the repository virtual environment's normal Python interpreter and the user's normal OS permissions. Timeout and output capture do **not** disable network, filesystem, subprocess creation, or access to user data. Generated code could alter the workspace or other accessible files. Do not execute model generations with this helper.

Coding pass@1 must be blocked until a real isolation boundary is available and smoke-tested: for example, a disposable container/VM with no network, read-only runtime, a small writable scratch directory, process/memory/CPU/time limits, and no mounted user workspace. If strong isolation is unavailable, report coding functional grading as blocked rather than mislabeling the existing helper as safe.

## Persistence, resume, and manifest audit

### Old runner behavior

- `scripts/run_full_llama.py` reuses `baselines.json` and `aggregates_mean_abs.npz` merely when files exist. It does not verify dataset hash, model hash, tokenizer hash, source hash, schema, config, seed, completeness, shape, or finite values.
- `C_stress_rows.jsonl` is opened with mode `w`; restarting destroys prior rows.
- `D_interlink.json` is overwritten after each target with matrices initialized to zero. A zero can mean “not run” or a genuine measured zero.
- The final manifest is written only after all 15 interlink rows finish. The old Llama run stopped before that point, so no Llama manifest exists.
- `src/storage.py` uses direct writes. Its Parquet append loads the entire existing file and rewrites it in place, which is neither scalable nor atomic.
- There is no phase/state completion marker, no row checksum, no fsync strategy, no duplicate-key protection, and no resume reconciliation.
- Exact ranking arrays, calibration means, mask indices, mask hashes, token caches, per-item dense results, telemetry, failures/restarts, and complete resolved config are absent.
- `gitcommit()` in `scripts/run_full_llama.py:48-54` runs `git add -A` and commits the entire worktree after phases. This can capture unrelated user files and is unacceptable. Remove this behavior entirely; source identity must be recorded without mutating Git.
- The output directory is accepted from an arbitrary positional argument and existing files are overwritten/reused. The pilot entry point must resolve and assert that outputs stay under its unique versioned directory and must refuse an incompatible/nonempty run unless it is a validated resume of the same run ID.

### Required resumable design

- Freeze a canonical machine-readable config with a schema version and SHA-256 before the first intervention outcome.
- Use a run ID and immutable run manifest. Store environment, local-file hashes, source-tree hash, dataset/split hashes, seeds, architecture, quantization, device map, and resolved decoding config.
- Store phase and state records with unique compound keys such as `(target, ranking_id, intervention, comparator, random_seed, step, item_id)`.
- Write new Parquet part files or atomic temporary files followed by `os.replace`; never rewrite a large monolith to append. Maintain checksummed completion markers only after a part validates.
- On resume, validate every dependency hash, read and deduplicate part keys, discard/quarantine only incomplete temporary parts, and continue at the first missing state. Never infer completeness from file existence.
- Flush logs after each state and fsync often enough that at most one state is lost.
- Store long-form per-token and per-item files separately from derived summaries. HTML must be derived, never authoritative.
- Store ranking scores/order as arrays plus a tabular `(rank, layer, channel, score)` representation. Store per-step exact additions and hashes so each full nested mask is reproducible without duplicating 101 dense boolean arrays.

## Old Llama artifacts: exact completeness findings

`results/comprehensive_llama31_8b` contains seven files and no `manifest.json`:

- `baselines.json`
- `aggregates_mean_abs.npz`
- `A_overlap.json`
- `B_allocation.json`
- `C_stress.json`
- `C_stress_rows.jsonl`
- `D_interlink.json`

`C_stress_rows.jsonl` has exactly 1,055 rows:

- 700 global-activity rows for steps 0 through 699.
- 355 random rows: five seeds at step 0, every 10 steps, and the final step.
- The maximum recorded fraction is 0.6993778773716518, not exactly 0.70, because of the fixed rounded step.
- Rows contain capability-level NLL and NLL/base-NLL ratios; sparse global rows also contain capability aggregate generation scores. They contain no item-level losses, answer token counts, exact ablated count, exact indices, hashes, calibration identity, runtime telemetry, or schema version.

`D_interlink.json` has nonzero environment and high-removal rows only for coding and mathematics. The other 13 target rows remain initialized zeros and are unmeasured. The log ends after “math done” and has no completion marker. This artifact is explicitly incomplete.

The old `C_stress.json` field `failure_fraction_pct` actually stores unit fractions (for example 0.147..., meaning about 14.7%), not percentage numbers. Old rows also use `nllratio = NLL_state / NLL_baseline`; that must not be confused with the requested perplexity ratio `exp(NLL_state - NLL_baseline)`.

The old Llama runner performed a global-activity stress sweep to roughly 70%, not the requested two target-conditioned dense pilot trajectories to exactly 10%. It is not a substitute for this pilot.

## Old fidelity check: scope and defects

The old smoke fidelity artifact reports:

- int8 coding NLL 0.93302817 versus CPU FP16 NLL 0.93499770 (0.211% relative difference), on the old small coding battery;
- mean top-20% activation Jaccard 0.9489 across five sampled layers `[0, 8, 16, 24, 31]`.

This is encouraging provenance but does not pass the new fidelity gate because it uses the boundary-incorrect scorer, an old/leaky small battery, only coding, only five layers, top-20% rather than top-5%/top-10%, no ranking Spearman, and no functional-accuracy comparison. It cannot justify claims about the new int8 rankings.

The old CPU FP16 run took approximately 6 minutes 27 seconds total: load about 21 seconds, baseline NLL about 2 minutes 22 seconds, and capture about 3 minutes 44 seconds. A full CPU FP16 validation is therefore impractical; a preregistered small, meaningful subset/layer comparison is reasonable if it satisfies the new definitions.

## Runtime evidence and provisional projection

Prior runtime evidence is useful only as an engineering prior:

- Old int8 model load: approximately 13 seconds.
- Old comprehensive baseline phase, including generation and NLL for 600 rows: about 14 minutes 27 seconds.
- Old capture plus pooled calibration: about 5 minutes 46 seconds, but the implementation used redundant forwards and invalid data separation.
- Old stress sweep: 23,198 seconds (6.44 hours) for 700 global conditions plus 355 random conditions, 600 old rows per NLL condition, and sparse generation.
- Away from generation checkpoints, 20 global states plus two sets of five random-seed states processed roughly 18,000 prompt-condition evaluations in about 335–337 seconds, approximately 53.4–53.7 prompt-condition evaluations/s. This matches the previously cited ~53.5 prompts/s, but it is not a benchmark of the new 720-item battery.
- The old 5% smoke shows generation dominates: ordinary NLL-only steps took about 7.5–8 seconds, while generation checkpoint steps added several minutes.

At 53.5 evaluations/s, the requested 144,000 dense non-baseline evaluations would take about 2,692 seconds, or 44.9 minutes. This estimate excludes the shared baseline, capture, stability resamples, comparators, ten-seed controls, fidelity, functional generation/grading, telemetry overhead, and storage of per-token data. It also assumes the new battery has similar sequence lengths and batching efficiency. It must not be used as the final measured projection.

Before dense execution, benchmark the real frozen 720 items with the actual cached tokenization and persistent hooks. Report separately:

1. no-hook and empty-hook batched NLL prompt/s and gold-token/s;
2. nonempty mean-ablation NLL at representative 0.1%, 5%, and 10% masks;
3. capture prompt/s and tokens/s;
4. zero/mean/resampling control throughput;
5. generation tokens/s for coding, translation, and sentinel tasks;
6. grading wall time, especially isolated code tests;
7. read/write volume and time for raw per-token and per-item persistence;
8. peak allocated/reserved VRAM and model residency.

Length-bucket batches, cached tokens/masks, inference mode, stable batch sizes, one resident model, persistent hooks, and minimal float32 materialization are the main likely gains. Any optimization must be checked for exact scorer agreement before use.

## Hard safety guards required in the new entry point

The pilot runner must fail closed unless all of the following are true:

- Resolved model directory is exactly the approved local Llama checkpoint (or a manifest-authorized alias resolving to it).
- Local config says `model_type=llama` and architecture is `LlamaForCausalLM`; reject any path/name/config containing or resolving to Qwen.
- Offline flags are set before importing/loading Transformers; no fallback download is permitted.
- `max_step == 100`, `max_permille == 100`, maximum fraction is exactly 0.10, and targets are exactly coding and translation. Reject arbitrary target lists and any fraction above 10%.
- Output resolves beneath the selected new versioned pilot directory and does not equal any old results directory.
- A nonempty output directory is accepted only as a hash-matching resume.
- Held-out split hashes are sealed and absent from ranking/capture inputs.
- Functional code execution is disabled until a genuine isolation self-test passes.
- The runner performs no Git mutation, no model download, no shell pipeline launch, and no subprocess capable of starting unrelated models.

## Prioritized implementation and test plan

### P0 — freeze schemas and pure CPU invariants before any Llama load

1. Create a pilot-only config and schema package under the versioned experiment directory. Include schema versions, target list, exact 101 states, sentinel fractions, seeds, intervention modes, metrics, generic-collapse bound, and artifact locations.
2. Implement the model/Qwen guard, 10% hard cap, path containment, offline enforcement, and immutable-resume validation first. Unit-test every rejection path.
3. Implement exact `k(s)=floor(N*s/1000)` schedule generation. For `N=458752`, assert `k(1)=458`, `k(100)=45875`, strict monotonicity, and increments in `{458,459}`.
4. Implement canonical stable global ordering of raw scores with flat-index tie break. Persist score/order arrays and deterministic SHA-256 hashes. Do not apply old per-layer percentile normalization or core exclusion to the primary ranking.
5. Implement nested-mask state records and tests: exact counts, uniqueness, bounds, prefix/nesting, per-layer counts, state hash reproducibility, and random masks matching the primary per-layer counts for every sentinel.
6. Implement joint prompt-plus-gold tokenization and boundary mapping. Add hand-inspected multilingual fixtures and exact aggregation tests. Freeze cached token IDs, masks, offsets/status, IDs, and hashes.
7. Define versioned long-form per-token, per-item, per-state, generation, grading, telemetry, and failure schemas. Test aggregation round trips and primary/diagnostic split separation.
8. Implement atomic part writes, checksummed completion markers, duplicate-key detection, and resume tests including simulated interruption during a write.
9. Replace generated-code execution with a genuine isolation adapter and self-test, or mark code functional grading blocked.

### P1 — synthetic hook/capture tests without loading Llama

1. Build a tiny local gated MLP with `gate_proj`, `up_proj`, and `down_proj`. Prove the pre-hook input equals the manual post-gate product.
2. Test no-op/empty mask equivalence at tensor/logit level and repeated-run determinism.
3. For known channels, compare zero, mean, and seeded-resampling hook outputs to a manual intervention.
4. Test dtype preservation, device placement, nonfinite mean rejection, invalid/duplicate/out-of-range indices, and cleanup after exceptions/partial attachment.
5. Implement persistent hooks with in-place state updates and prove that changing a state cannot mutate ranking/order or a previous mask.
6. Implement one-pass capture of signed sum, absolute sum, squared sum, count, and a preregistered upper-quantile estimator. Verify exact small-tensor statistics.
7. Implement outgoing-column norm extraction and validate against a synthetic dense layer before choosing a bitsandbytes/original-shard path.

### P2 — one authorized live int8 integrity session after data/protocol freeze

1. Load only `D:/hf_models/Meta-Llama-3.1-8B` once with the frozen int8 config. Record versions, hashes, quantization config, actual quantized modules, device map, dtypes, GPU/driver/CUDA, and VRAM.
2. Assert live architecture and exact hook tensor semantics.
3. Assert every model module stays on GPU; halt on offload.
4. Compare uninstrumented, empty-hook, and no-op-hook logits/NLL on fixed examples within a preregistered tolerance. Repeat the same seed/run and compare raw per-token losses.
5. Test single versus right-padded batch and, if implemented, left-padded batch across multilingual items.
6. Test one known nonempty intervention against a manual layer-local reference.
7. Run a two-item/two-state smoke that must produce exactly the expected unique keys and survive a forced resume.
8. Benchmark the frozen 720 battery and finalize the runtime/storage projection before authorizing dense execution.

### P3 — baseline/capture/ranking gates

1. Run and persist the uninstrumented/empty-hook baseline for all 720 items, keeping held-out 600 and in-sample 120 separate.
2. Run baseline functional grading using only safe task adapters and preserve every generation/grading record.
3. Capture coding and translation localization sets only, compute target-specific means, and collect all preregistered statistics in as few passes as possible.
4. Produce primary and comparator rankings without looking at held-out outcomes.
5. Run disjoint-half and bootstrap stability. Stop or expand localization data if the preregistered gates fail.
6. Freeze ranking IDs, array hashes, splits, means, masks, and analysis config before the first held-out intervention state.

### P4 — dense pilot and controls

1. Execute only the primary coding and translation rankings for states 1–100, all 720 fixed items at each state, with one shared verified baseline.
2. Persist/validate each state before advancing; update live ledger and telemetry at least per state.
3. Run preregistered sentinels for comparators/interventions and at least ten layer-count-matched random seeds as runtime permits.
4. Run predetermined sparse generation and only add breakpoint-bracketing states according to the frozen rule.
5. Never start a target beyond coding/translation or a state above 100.

### P5 — independent derivation and audit

1. Derive every macro, paired delta, perplexity ratio, curve summary, breakpoint, CI, collateral matrix, and runtime projection from raw versioned files.
2. Have an independent audit recompute row counts, mask counts/hashes, primary summaries, and report values without relying on generated HTML.
3. Treat any missing phase, unsafe coding grading, fidelity limitation, or intervention reversal as an explicit incomplete/conditional result rather than completion.

## Pre-GPU implementation acceptance checklist

GPU execution should remain blocked until all boxes below pass:

- [ ] Frozen pilot config and dataset/split hashes exist.
- [ ] Joint-token boundary fixtures pass for all relevant scripts/languages.
- [ ] Exact schedule and nested-mask/hash tests pass.
- [ ] Global ranking test confirms no implicit per-layer normalization/core exclusion.
- [ ] Random sentinel masks match primary per-layer counts for every seed/state.
- [ ] Synthetic post-gate hook, no-op, zero, mean, and resampling tests pass.
- [ ] Atomic persistence/resume and duplicate-key tests pass.
- [ ] Qwen, output-path, target-list, and >10% rejection tests pass.
- [ ] Generated coding evaluation is truly isolated or explicitly disabled.
- [ ] Raw schemas include token/item/capability/split/state/mask identities and schema versions.
- [ ] No code path mutates Git or old result directories.

## Final reuse decision

**Reuse after fixes:** down-projection accessor concept; forward-pre-hook intervention concept; stable sorting primitive; layer-count-matched random-control concept; deterministic greedy-generation concept; selected parser helpers after tests.

**Do not reuse as implementations:** old tokenization/NLL functions, capture/calibration pipeline, per-layer-percentile primary scoring, core-excluded mask builder, fixed-step sweep, old Llama runners/configs, storage/resume code, coding “sandbox,” aggregate-only result schema, or old fidelity procedure.

**Provenance only:** every old Llama numeric result, curve, breakpoint, activation array, and interlink matrix. They may be discussed as reasons for redesign but must not enter the new pilot verdict.
