# Independent prior-art and protocol audit

**Audit scope.** This note reviews the proposed coding/translation Llama-3.1-8B int8 go/no-go pilot against the eleven named primary sources and primary/official metric documentation. It is a protocol audit, not a results audit. No model or GPU process was started. Links were checked on 2026-08-29.

## Bottom line

The proposed study is scientifically defensible as a **causal, capability-conditioned MLP-channel activation-lesion pilot**, provided that the protocol is frozen before held-out evaluation and the corrections below are adopted. Its strongest contribution would be unusually dense, held-out, cross-capability dose-response mapping with stability, replacement, random-mask, quantization, and generation checks. It should **not** be presented as the first task-specific-neuron, language-neuron, activation-ranking, or pruning study. Xiao et al. already define the same gated-FFN intervention unit, use activation/output-magnitude indicators, and perturb functionality-ranked neurons while measuring cross-function perplexity; Gurgurov et al. study language neurons in the same Llama-3.1-8B base family; several other direct ancestors use task-conditioned neuron rankings and causal perturbations.

The central interpretation must remain: performance dependence under a specified intervention in the native MLP-channel basis. The experiment cannot establish that a channel “stores,” “contains,” or uniquely implements a capability; superposition, correlated prompts, layer-scale effects, and off-manifold replacement all remain alternatives.

## Direct prior-art verification and positioning

| # | Primary source and verified result | Positioning and methodological implication for this pilot |
|---|---|---|
| 1 | Xiao et al., **[Configurable Foundation Models: Building LLMs from a Modular Perspective](https://arxiv.org/abs/2409.02877)** (2024). This is partly a perspective/review but includes an empirical Llama-3-8B-Instruct/Mistral-7B-Instruct study. It explicitly defines a gated-FFN “neuron” as one row of the gate/input projections plus the corresponding output-projection column, with activation `sigma(W_G x) * (W_I x)`. It computes per-instance mean absolute activation over prompt tokens, uses label-discriminative average precision as a functionality score, compares activation with output magnitude, masks high-functionality neurons, and reports cross-function perplexity effects. It also dynamically masks low-indicator neurons per token/layer in its sparsity study. | This is the closest direct ancestor and must be discussed prominently, not relegated to generic modularity. The proposed fixed, globally cumulative, target-calibrated ranking; sealed 15-capability battery; mean replacement; dense 0.1% grid; ranking bootstrap; and generation validation are extensions/differences, not a license for a broad novelty claim. Its output-magnitude observation motivates a channel-output comparator, but its per-token dynamic masking is not the same intervention as this pilot’s frozen global masks. |
| 2 | Song et al., **[Does Large Language Model Contain Task-Specific Neurons?](https://aclanthology.org/2024.emnlp-main.403/)** (EMNLP 2024). CGVST localizes task-related neurons using causal gradient variation focused on task-significant/special tokens; it evaluates eight public tasks and validates selected neurons through inhibition and amplification. | Direct task-neuron localization/causal-perturbation ancestry. It supports including a technically validated gradient-based comparator and recording which token positions drive the score. The pilot’s mean-absolute activation score is methodologically different and should not be described as superior without direct evidence. |
| 3 | Liang et al., **[SEAP: Sparse Expert Activation Pruning Unlocks the Brainpower of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/40463)** (AAAI 2026, DOI 10.1609/aaai.v40i38.40463). SEAP uses a multi-task task-expert calibration set, hidden-state/neuron-activation clustering, task-aware masks, and a router to select computation paths; it evaluates cross-task transfer and targets deployable inference sparsity. | Direct task-conditioned activation-path/pruning ancestry. It motivates cross-task transfer/collateral evaluation and representative calibration. It differs because SEAP is dynamic task-aware structural computation/pruning with an efficiency objective, whereas this pilot keeps weights intact and performs fixed inference-time activation replacement. Do not claim speedup or model compression for the pilot. |
| 4 | Yang et al., **[Task-specific Compression for Multi-task Language Models using Attribution-based Pruning](https://aclanthology.org/2023.findings-eacl.43/)** (Findings of EACL 2023). The paper uses neuron attribution of the form activation times gradient of target-token probability, sums attribution over target tokens/examples, and structurally prunes low-attribution attention/FFN dimensions in T5; it also studies low-resource and unsupervised settings. | Direct task-conditioned channel-attribution/pruning ancestry and the correct citation for a first-order activation×gradient comparator. Because it physically slices weights and targets compression, its efficiency findings do not transfer to mean activation ablation. The comparator must specify the differentiated scalar (preferably gold-answer log likelihood/loss), sign/absolute-value handling, token aggregation, and numerical checks. |
| 5 | Wang et al., **[Finding Skill Neurons in Pre-trained Transformer-based Language Models](https://aclanthology.org/2022.emnlp-main.765/)** (EMNLP 2022). After prompt tuning, some FFN activations on soft-prompt positions predict task labels; selected neurons are perturbed, similar tasks show related neuron distributions, and multiple prompt-tuning trials are used to assess consistency. | Direct “skill neuron” and perturbation ancestry. It motivates calibration-resample stability and cross-task ranking overlap, while also showing that localization is conditional on prompting/tuning choices. Its classification/prompt-tuning setup does not establish localization in an untuned autoregressive base model. |
| 6 | Siam et al., **[Exploring the Limits of Pruning: Task-Specific Neurons, Model Collapse, and Recovery in Task-Specific Large Language Models](https://arxiv.org/abs/2604.27115)** (arXiv v1, 2026). It structurally prunes Qwen math/code models using target-versus-distractor activation selectivity, compares selective, random (two seeds), and reverse/high-selectivity pruning, tracks generation traps, and observes sharp damage at aggressive pruning. Exact ratios differ from nominal ratios because dimensions are kept divisible by 128. | Very direct recent preprint; cite with its unreviewed/preprint status and narrower task-specific-model setting. It motivates target-vs-distractor scores, high-importance positive controls, more than two random seeds, exact count reporting, and degeneration/EOS-loop monitoring. Its approximate 5–10% “collapse” and 15–20% robustness observations are study-specific results, **not standard thresholds** and not priors that justify tuning this pilot’s checkpoints. |
| 7 | Tang et al., **[Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models](https://aclanthology.org/2024.acl-long.309/)** (ACL 2024). LAPE ranks FFN neurons by entropy of per-language activation probabilities and reports language-associated neurons and language steering through activation/deactivation in LLaMA-2, BLOOM, and Mistral. | Direct multilingual-neuron/causal-intervention ancestry. Translation-conditioned activation is not equivalent to language specificity: each direction combines source comprehension, task recognition, and target production. Report language/direction effects and cross-language overlap instead of calling the ranking “translation neurons.” |
| 8 | Gurgurov et al., **[Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation](https://arxiv.org/abs/2507.22608)** (arXiv v3, 2025; listed as accepted to AACL main). It applies LAPE to 21 languages in Llama-3.1-8B base, Mistral-Nemo, and Aya models; reports deeper-layer concentration, greater specialization for non-Latin scripts, related-language overlap, resource-level differences, and activation-arithmetic steering across translation, QA, comprehension, NLI, and language forcing. | The closest same-model multilingual ancestor. It strongly supports direction-, script-, resource-, and layer-resolved translation analysis and cautions that differences can track training-resource coverage rather than an abstract translation module. The pilot differs in ranking a translation task across ten directions and lesioning by a frozen cumulative global order. |
| 9 | Sun et al., **[A Simple and Effective Pruning Approach for Large Language Models (Wanda)](https://arxiv.org/abs/2306.11695)** (ICLR 2024). Wanda scores each **weight** as `abs(W_ij) * ||X_j||_2` over calibration tokens and compares weights **per output row** before setting selected weights to zero. | Correction required: `mean_abs(post_gate_channel) * ||down_proj_column||` is **not Wanda**. It is a channel-level activation×outgoing-weight-norm proxy “inspired by Wanda/output-magnitude reasoning.” Name and formula must be explicit, and it must not inherit Wanda’s compression claims. |
| 10 | Elhage et al., **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)** (Transformer Circuits, 2022; [arXiv mirror](https://arxiv.org/abs/2209.10652)). Sparse toy features can occupy superposed directions, producing polysemantic neurons; the authors explicitly caution that generalization from toy models to real networks is uncertain. | Core claim-boundary source. A channel-basis lesion can affect several superposed features, and a feature can span many channels. “Functional dependence/sharing under this basis and intervention” is justified; “the neuron contains capability X” is not. |
| 11 | Pochinkov, Pasero & Shibayama, **[Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering](https://arxiv.org/abs/2408.17322)** (2024). The full title is longer than “Investigating Neuron Ablation.” It compares zero, mean, resampling, and peak ablation in language/vision settings and finds that the least-damaging method varies by regime/model, with resampling often most damaging. | Direct evidence that causal conclusions may depend on the replacement distribution. Mean, zero, and resampling controls are necessary. Note that its units/settings are not the same gated-MLP channels, so it motivates a robustness test rather than a predicted ordering. |

## Required protocol corrections before freezing

### 1. Define what “target-conditioned” means

Mean absolute activation on only coding prompts or only translation prompts is **target-corpus-conditioned**, but it is not a selectivity score: globally common channels can rank highly and globally dormant channels can rank low. The protocol should state this limitation and separate:

- the primary target-corpus mean-absolute ranking;
- an independent generic/global-activity ranking from a fixed generic corpus;
- an optional target-minus-reference or target/distractor selectivity comparator; and
- direct coding-vs-translation ranking overlap and cross-lesion effects.

A “shuffled calibration-label” control is undefined for a score that never uses labels. For the primary score, use a **shuffled target assignment/prompt-pool control** (e.g., deterministic coding/translation prompt reassignment while preserving lengths/formats), or reserve label shuffling for an explicitly label-aware selectivity/CGVST comparator.

### 2. Freeze activation aggregation and global scaling

Predeclare an item-balanced score, preferably: for each channel, average `abs(post_gate_product)` over non-padding localization tokens within each prompt, then average prompt means, so long prompts do not dominate. State whether prompt-only or prompt+gold tokens are captured; do not mix conventions across targets. Exclude padding and document BOS/EOS/special-token handling. Save token counts and both prompt-balanced and token-weighted diagnostics.

Raw global ranking can be dominated by layer-specific activation scale. If raw global mean absolute activation is the required primary, keep it unchanged but report the selected count by layer at every state, per-layer activation distributions, and layer-matched random controls. Any layer-normalized ranking is a labeled sensitivity analysis, not a silent replacement.

The activation×outgoing-weight comparator should be named **channel output-magnitude proxy** and given exactly, such as `mean_abs(h_lj) * ||W_down[:,j]||_2`. Do not call it Wanda.

### 3. Make localization, replacement, and evaluation sets unambiguous

There must be three disjoint roles in the manifest:

1. ranking/localization prompts (24–32+ per target);
2. replacement-statistic prompts used to estimate signed channel means/resampling distributions; and
3. the 600 sealed held-out plus 120 explicitly in-sample diagnostic items scored on the dense trace.

If (1) and (2) are the same prompts, say so. The 120 diagnostics must not be described merely as “calibration” because that can be confused with the separate ranking/replacement calibration sets. Publication conclusions use only the 600 sealed items. No battery edit, threshold change, comparator choice, or generation-checkpoint choice may be informed by their intervention outcomes.

Mean replacement must specify per-layer/channel signed mean, item/token weighting, token positions, dtype of accumulation/storage/application, and whether the same target-specific means are used for all evaluated capabilities under that target ranking. Resampling must specify the empirical unit (token/prompt), reservoir or stored distribution, deterministic RNG stream, and whether draws are shared across batch/order; otherwise it is not reproducible.

### 4. Tighten stability gates

Compute split-half and prompt-bootstrap stability using localization data only. Layerwise Spearman across all 14,336 channels can look high because of the stable bulk while the ablated tail is unstable; retain it, but make top-5% and top-10% overlap/Jaccard mandatory co-primary diagnostics. Predeclare whether the gate requires both overlap levels or one specific level—the phrase “top-5% or top-10% Jaccard ≥ 0.60” leaves researcher choice after seeing results. Report the distribution across layers (median, minimum, and failed-layer count), not only an average.

Bootstrap rankings by resampling whole prompts, recomputing the entire ranking, and preserving seed/split manifests. Adaptive reranking, if run, must use independent calibration-only activations at predeclared sentinel fractions; held-out bends cannot choose its masks.

### 5. Preserve the intervention semantics

Call all primary results cumulative structured **activation ablation/mean replacement**, not pruning. No smaller checkpoint or speedup is produced. The empty-mask hook should match no-hook logits and per-item NLL within a predeclared absolute/relative tolerance in the actual int8 execution dtype; “floating-point tolerance” alone is not reproducible. Test the hook against a direct reference computation of `SiLU(gate_proj(x)) * up_proj(x)` before `down_proj`, including hook input/output dtype.

For the exact schedule, save `k_s=floor(N*s/1000)` for `s=1..100`, selected `(layer, channel)` arrays, per-layer counts, and hashes. Assert strict set nesting and `|mask_s|=k_s`. With the expected `N=458,752`, the first state has 458 channels and the 10% state has 45,875; these remain provisional until live architecture verification.

### 6. Correct dense-sweep accounting

The two nonzero trajectories contain exactly `2 * 100 * 720 = 144,000` item-state scores. If the identical unablated baseline is computed once, the dense primary table should contain **144,720 rows**; if baseline rows are duplicated under each target for convenient joins, it has **145,440 rows** but only 144,720 unique model-item-state evaluations. Predeclare one schema and assert it in the smoke test and final audit.

### 7. Predefine the generic-collapse rule numerically

“Monitor generic quality” is insufficient. Before intervention outcomes, freeze a generic measure/corpus and a numeric broad-damage boundary, plus the logic for first sustained crossing. Continue recording to 10% as requested, but label post-bound states broad model damage. The all-capability macro alone is not independent of the claimed effect; a separately frozen generic LM sample is preferable, with held-out all-capability macro as a second monitor.

### 8. Analyze the correlated trajectory correctly

Use paired prompt bootstrap on **whole trajectories**, not independent resampling by fraction. For translation, stratify resampling by direction for overall summaries. For each bootstrap replicate, recompute first crossing, five-state sustained crossing, AUC/early damage, and specificity. Treat no crossing by 10% as right-censored (`>10%`), not 10%. Report percentile/basic/bootstrap intervals and the number of censored replicates.

`fail_frac_2x_gold_nll` means `NLL_s > 2*NLL_0`; it is not “2× perplexity.” Perplexity ratio is separately `exp(NLL_s-NLL_0)`. The endpoint is user-defined and baseline-scale-sensitive, not literature-standard collapse. Sparse generation must bracket any substantive bend/crossing before functional collapse is claimed.

Predeclare a small primary set, for example: target held-out AUC of paired ΔNLL through 10%; target-minus-nontarget specificity AUC; sustained 2×-NLL crossing; and target functional retention at 10%. Treat 15×full curves as descriptive. If inferential p-values are used across capability summaries, state the correction family and method (e.g., Holm familywise control); do not test 101 states independently.

### 9. Translation analysis must be direction-macro and exploratory at fine resolution

The overall translation score should macro-average the ten direction-level item means so English-heavy grouping does not accidentally dominate. Report source- and target-language groupings with exact denominators. Each direction has only four held-out prompts, so direction BLEU/chrF and breakpoints are exploratory with extremely wide prompt-bootstrap intervals. “Script-family effect” is an observation/hypothesis, not an identified causal mechanism; resource coverage and prompt difficulty are competing explanations.

### 10. Quantization fidelity needs matched estimands

Compare int8 and BF16/FP16 on identical tokenized examples, hooks, token aggregation, and layer/channel identities. Report NLL and functional differences, layerwise ranking Spearman, and the same top-k overlap definition used by the gate. A subset/layer check supports only scoped int8 conclusions. Do not average away layers that fail. Record whether BF16/FP16 involved CPU offload, because execution differences can contaminate timing but not necessarily ranking fidelity.

## Functional metric definitions to freeze

### Coding

HumanEval defines functional correctness by executing generated Python against tests; the original benchmark and `pass@k` estimator are documented by Chen et al., **[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)**. EvalPlus augments tests to expose false positives; see Liu et al., **[Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210)** and the official **[EvalPlus repository](https://github.com/evalplus/evalplus)**.

For deterministic one-completion generation, report **pass@1 as the fraction of items whose single completion passes all tests**; do not use the multi-sample pass@k estimator. Freeze and hash task/test versions, harness, sanitizer, imports, timeout/memory limits, completion extraction, and decoding. Report base-test and plus-test pass@1 separately. Execute untrusted generated code in an isolated, resource-limited environment.

### Translation

Use corpus SacreBLEU and archive the complete signature, not merely a scalar. Post, **[A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/)**, shows that tokenization/normalization choices materially change BLEU. The official **[SacreBLEU implementation](https://github.com/mjpost/sacrebleu)** emits version/configuration signatures and uses target-aware tokenizers (notably `zh` for Chinese when the language pair is supplied). Store detokenized hypotheses/references, library version, tokenizer, case, smoothing, effective-order, and reference count.

chrF is the character n-gram precision/recall F-score introduced by Popović, **[chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/)**. Freeze the SacreBLEU chrF signature (`beta`, character/word order, case, whitespace, smoothing). With only four prompts per direction, chrF is often more interpretable than direction-level BLEU, but neither yields a precise breakpoint. Bootstrap prompts and recompute corpus metrics; do not average sentence BLEU as though it were corpus BLEU.

### Question answering

Use the SQuAD convention from Rajpurkar et al., **[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)**: normalized exact match and bag-of-normalized-token F1 (precision/recall overlap). Freeze normalization (case, punctuation, articles, whitespace), multi-reference max rule, and answer extraction. Count parse failures as failures, not missing observations.

### Summarization

ROUGE originates with Lin, **[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)**. Freeze implementation/version and report ROUGE-1/2 and ROUGE-L or ROUGE-Lsum F1 with stemming and sentence-splitting settings. ROUGE is lexical overlap and does not establish factuality or overall writing quality; preserve generations and add a fixed, disclosed rubric if qualitative validity matters.

### Mathematics, closed choice, classification, and open-ended tasks

For grade-school numeric tasks, the GSM8K paper (Cobbe et al., **[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)**) reports final-answer accuracy. Freeze a deterministic final-answer parser and numeric normalization before intervention outcomes. Any symbolic-equivalence extension must specify the algebra system/version, domain assumptions, timeout, and failure policy; label it separately from normalized exact match.

For closed choice, prefer accuracy from restricted probabilities over canonical answer labels. Freeze whether choices are scored as single label tokens or full strings; raw summed log probability of variable-length answer strings is length-biased. Record chance level and require materially above-chance baseline. For classification, freeze label mapping, extraction, and whether accuracy or macro-F1 is primary. Creative/open-ended outputs have no universal reference metric: use a preregistered rubric, blind/randomized grading, and reliability checks, and label automated-grader results exploratory.

## Additional stop/go implications

- The base (not instruction-tuned) Llama checkpoint may have weak prompt-following. Baseline gates must be applied before any lesion outcome is examined; failing capabilities should be revised/excluded only under a documented pre-outcome rule.
- The minimum random-control commitment should be explicit. If ten layer-matched seeds cannot be completed at all five nonzero sentinels, define the minimum acceptable fallback before results; otherwise “runtime permits” creates discretionary evidence strength.
- Positive-control degradation is necessary but not sufficient: high-activity ablation may cause broad damage. Specificity requires target-vs-nontarget contrast and generic-collapse timing.
- If mean, zero, and resampling disagree in sign/order, the correct conclusion is intervention dependence. Do not average replacements into an apparently stable central effect.
- Teacher-forced NLL and autonomous functional scores answer different questions. Multiple valid outputs make gold NLL especially incomplete for code, translation, summarization, and creative work. Directional agreement is a gate, not expected numerical equivalence.
- A GO should be withheld if target rankings fail the fixed stability gate, primary least-active masks are indistinguishable from layer-matched random masks, or effects begin only after the independent generic-collapse boundary. A comparator (output-magnitude proxy or Taylor) may justify a **method-redesign recommendation**, not retroactive substitution into the primary experiment.

## Citation-ready claim language

Recommended wording:

> We study cumulative structured activation ablation in the native gated-MLP channel basis. Rankings are conditioned on independently frozen coding or translation localization prompts, and selected post-gate products are replaced at inference time. The intervention identifies behavioral dependence and collateral effects under this basis and replacement distribution; it neither creates a compressed model nor establishes that an individual channel stores a capability.

> The study extends direct task-neuron, language-neuron, and task-aware pruning precedents through a sealed multi-capability battery, dense cumulative dose-response curves, calibration-resample stability, layer-matched random controls, replacement-method sensitivity, and paired functional generation. Because neuron polysemanticity and superposition can couple multiple features to one channel, observed selectivity is evidence of functional specialization/sharing under intervention, not a complete algorithmic decomposition.

## Audit disposition

**Protocol status: conditionally ready to freeze.** The design is strong enough for a go/no-go pilot after the corrections above are made machine-checkable in the protocol/config and before any held-out intervention result is inspected. The most consequential required changes are: (1) correct Wanda terminology; (2) distinguish corpus-conditioned activity from task selectivity; (3) resolve the three data roles and label-shuffle control; (4) freeze token/item aggregation, stability logic, generic-collapse bound, and random-control minimum; and (5) make dense row accounting and functional metric configurations exact and assertable.
