# Calibration: debiasing, cross-fitted calibrators and Jev's confidence

Sep 25, 2026; current results Sep 27, 2026 (one call per question)

## In short

All numbers in this list are the current one-call runs
([Current results](#current-results-one-call-per-question-holdout)).

- **Accuracy.** On held-out questions the emulator is less accurate than Jev: 0.7655 against
  0.8253 over 9 benchmarks (Δ −0.0598 [−0.0726, −0.0473]).
- **Calibration.** Both systems are overconfident on some tasks. One fitted temperature per
  question signature fixes most of it. After that, the emulator's probabilities are as well
  calibrated as Jev's (ECE 0.035 against 0.048). The gap that remains in NLL and Brier comes
  from knowledge, mostly MMLU-Pro and GPQA.
- **Debiasing.** In the multi-call study PriDe cost 25% more backend calls and gave no
  measurable gain, so the recommended setup uses no debiaser. No current run has one.
- **Deployment.** Use the per-signature temperature registry in
  [`calibration/qwen3.6-27b-int4-quanttrio/registry.json`](../../calibration/qwen3.6-27b-int4-quanttrio/registry.json)
  ([Recommended deployment](#recommended-deployment)).
- **Jev's `confidence` field** is a peakedness score, not a probability. Its formula is now
  known (`mode_distance`) and is the emulator's default.
- **Other models.** Before calibration, GPT-6 Luna, Gemma 4 and DeepSeek V4.1 Flash are the
  most overconfident (ECE 0.168, 0.167 and 0.105 on the 6 benchmarks every system answers,
  against 0.072 for QuantTrio). After calibration, Luna and DeepSeek still trail the QuantTrio
  emulator (Brier 0.340 and 0.326 against 0.314 on those 6). The fast MoE emulator gives up
  0.035 accuracy for 4.4x the throughput (both on `holdout`). Its GPTQ build ties it on
  accuracy (−0.0039 [−0.0146, +0.0068]) with slightly worse raw probabilities (ECE 0.087
  against 0.077); after calibration the two are close (ECE 0.042 against 0.037).
  Gemma 4 26B-A4B, a MoE with 3.8B active parameters, ties the fast emulator on accuracy
  (+0.0045 [−0.0104, +0.0195]) and is strongly overconfident before calibration (ECE 0.165);
  one temperature per signature brings its ECE to 0.050 and its NLL level with the fast
  emulator's.

**Yelp is dropped** (Sep 25, 2026, after this study). Its star levels are shown to the models
as digits 0-4, which are read ambiguously. The headline grids below cover the other 9
benchmarks (6 for GPT-6 Luna, 7 for DeepSeek V4.1 Flash). The detailed tables after them
(every calibrator, best arm, per benchmark, debiasing, deployment) were computed with Yelp and
are kept as recorded. Their Yelp rows and 10-benchmark macro averages are superseded.

**Scoring changed** (Sep 27, 2026, after this study). Every emulator question is now one model
call (`auto_single`): letters up to 32 options, single-token two-capital codes above that
(banking77, CLINC150), instead of the trie
([selection.md](selection.md#scoring-now-one-call-per-question)). The QuantTrio 27B `holdout`
run is redone without PriDe, and the Gemma 4 runs after the prompt fix. The current results
are in [Current results](#current-results-one-call-per-question-holdout). The later sections
keep the studies as run: multi-call history (before one-call scoring), with `auto_noecho`, the
27B runs with online PriDe, and Gemma before the prompt fix.

### Terms

- **`select` / `holdout`.** The two halves of each benchmark. The emulator configuration is
  chosen on `select` ([selection.md](selection.md)). Calibration and the final numbers use
  `holdout`, which selection never saw.
- **Calibration.** A system is calibrated when its probabilities match how often it is right:
  answers given 0.8 are correct 80% of the time. A *calibrator* remaps probabilities to get
  there, and is fitted on items with known answers.
- **NLL** (negative log-likelihood): −log of the probability given to the correct answer,
  averaged. **Brier score:** squared error between the probability vector and the correct
  answer. **ECE** (expected calibration error): answers are grouped into bins by their top
  probability, and ECE averages the gap between that probability and the bin's accuracy. Lower
  is better for all three.
- **Signature.** A key for a question's answer space: its type, option count and a hash of the
  option keys (or score levels), for example `choice:10:4a0bcc85ad6db29b`. Option text is not
  part of it, so all A–E questions share one signature, across benchmarks.
- **Calibrators.** *Temperature scaling* divides the log-probabilities by one fitted number T
  (T > 1 softens an overconfident system); it never changes the top answer. *Vector scaling*
  adds a fitted bias per option position, so it can change answers. *Platt*, *isotonic* and
  *histogram* replace the top answer's probability with a fitted estimate of how often it is
  right: a logistic curve, a non-decreasing step function, or per-bin accuracy.
- **Scope** (`@global`, `@benchmark`, `@signature`): one calibrator overall, per benchmark, or
  per signature.
- **Cross-fitting (out of fold).** Items are split into five folds. Each fold is scored with a
  calibrator fitted on the other four, so no item is scored by a calibrator that saw it.
- **Debiasers** remove the model's preference for answer positions (for example, picking A too
  often). *PriDe* (prior debiasing) estimates that position prior by showing some questions
  with their options rotated, then divides it out. *Batch* divides by the mean prediction over
  a batch.
- **IDK:** an "I don't know" option. **q/s:** questions per second. **Δ:** a paired difference
  with its 95% confidence interval in brackets.

## What this page covers

- **Debiasers** (`jevemu.debias`): remove the emulator's preference for answer positions.
- **Cross-fitted calibration** (`jevemu.eval.crossfit`, `metrics_calib`, `report`;
  `scripts/calibrate.py crossfit`): every calibrator is fitted on four folds and scored on the
  fifth. Jev and the emulator get the same treatment ("calibrate both or neither").
- **Jev's confidence function (J3)** (`jevemu.confidence.fit`): Jev's `confidence` is
  `mode_distance`, now the emulator's default.

**Result as recorded** (multi-call history, before one-call scoring; holdout, 19,840 paired
items, 10 benchmarks including Yelp, macro average):

- **Headline accuracy.** The selected configuration (`qwen3.6-27b-int4-quanttrio`,
  `state_first`, `auto_noecho`, no debiaser) scores **0.7512 [0.7393, 0.7632]**. Jev scores
  **0.8097 [0.7992, 0.8200]**. Δ −0.0585 [−0.0703, −0.0468]. Selection never saw these items,
  so this is the unbiased number (on `select` the gap was −0.060).
- **Calibration.** Per-signature temperature scaling takes the emulator's ECE from 0.086 to
  0.043 (Jev: 0.078 to 0.047) and its NLL from 0.767 to 0.692. Calibrated, the emulator's ECE
  matches Jev's (Δ −0.004). Its NLL is 0.118 [0.102, 0.134] worse and its Brier 0.062
  [0.054, 0.070] worse. That gap is knowledge (accuracy on MMLU-Pro −0.197, GPQA −0.222).
- **Deployment.** No debiaser, and a `temperature@signature` registry fitted on the whole
  holdout run (the file now holds the one-call refit):
  [`calibration/qwen3.6-27b-int4-quanttrio/registry.json`](../../calibration/qwen3.6-27b-int4-quanttrio/registry.json).
  PriDe has no measurable effect (Δ NLL −0.003 [−0.008, +0.002]) and costs 25% more backend
  calls. Batch priors and `vector@signature` score better on these benchmarks. But they learn
  each benchmark's label mix, and the registry would apply it to every user question with the
  same signature ([Recommended deployment](#recommended-deployment)).
- **Tied finalist.** `qwen3.6-27b-int4-cyankiwi`, run the same way, also ties QuantTrio on
  holdout: accuracy −0.0006 [−0.0091, +0.0077], calibrated NLL +0.005 [−0.002, +0.013]
  ([The tied finalist](#the-tied-finalist-cyankiwi)).

Per-item outputs stay in `runs/`.

## Current results: one call per question (holdout)

Sep 27, 2026. Every emulator run here uses `auto_single`, `state_first` and no debiaser. The
grid is the macro average over **9 benchmarks** (Yelp dropped), 17,340 paired items, from the
`holdout_single` study ([Commands](#commands)). "Calibrated" means `temperature@signature`,
cross-fitted over 5 folds. Calibration does not change accuracy. The cyankiwi row and the
rows against QuantTrio come from a second study over the same runs plus cyankiwi
(`--reference quanttrio`, `runs/calibration/holdout_single_vs_quanttrio`), the rows against
the fast emulator from a third (`runs/calibration/holdout_single_vs_fast`).

| System | Accuracy | NLL raw | NLL calibrated | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 0.8253 [0.8138, 0.8365] | 0.703 [0.666, 0.741] | 0.555 [0.530, 0.581] | 0.263 | 0.255 | 0.072 | 0.048 |
| **Emulator (QuantTrio 27B)** | 0.7655 [0.7523, 0.7787] | 0.750 [0.721, 0.780] | 0.688 [0.663, 0.714] | 0.328 | 0.320 | 0.067 | 0.035 |
| cyankiwi 27B | 0.7641 [0.7507, 0.7773] | 0.748 [0.718, 0.780] | 0.691 [0.666, 0.718] | 0.328 | 0.321 | 0.065 | 0.032 |
| Fast emulator (AWQ 35B-A3B) | 0.7304 [0.7169, 0.7438] | 0.900 [0.868, 0.932] | 0.835 [0.808, 0.862] | 0.375 | 0.363 | 0.077 | 0.037 |
| GPTQ 35B-A3B | 0.7265 [0.7131, 0.7401] | 0.915 [0.882, 0.947] | 0.838 [0.811, 0.864] | 0.383 | 0.366 | 0.087 | 0.042 |
| Gemma 4 26B-A4B | 0.7349 [0.7212, 0.7484] | 1.270 [1.207, 1.337] | 0.833 [0.804, 0.862] | 0.426 | 0.372 | 0.165 | 0.050 |

Paired differences (a − b, the same items):

| a − b | Δ accuracy | Δ NLL raw | Δ NLL calibrated | Δ Brier calibrated | Δ ECE calibrated |
| --- | --- | --- | --- | --- | --- |
| **Emulator − Jev** | -0.0598 [-0.0726, -0.0473] | +0.047 [+0.021, +0.073] | +0.133 [+0.115, +0.151] | +0.065 [+0.056, +0.074] | -0.013 |
| Fast − Jev | -0.0950 [-0.1085, -0.0817] | +0.197 [+0.166, +0.228] | +0.280 [+0.258, +0.303] | +0.108 [+0.097, +0.118] | -0.011 |
| GPTQ − Jev | -0.0989 [-0.1117, -0.0864] | +0.212 [+0.181, +0.242] | +0.283 [+0.261, +0.305] | +0.111 [+0.101, +0.122] | -0.006 |
| Gemma − Jev | -0.0905 [-0.1052, -0.0759] | +0.567 [+0.510, +0.625] | +0.278 [+0.251, +0.305] | +0.117 [+0.105, +0.130] | +0.002 |
| cyankiwi − QuantTrio | -0.0014 [-0.0101, +0.0069] | -0.002 [-0.012, +0.009] | +0.003 [-0.005, +0.012] | +0.001 [-0.003, +0.005] | -0.003 |
| Fast − QuantTrio | -0.0352 [-0.0491, -0.0217] | +0.150 [+0.127, +0.173] | +0.147 [+0.127, +0.167] | +0.043 [+0.034, +0.053] | +0.002 |
| GPTQ − QuantTrio | -0.0391 [-0.0515, -0.0266] | +0.164 [+0.143, +0.187] | +0.150 [+0.131, +0.169] | +0.047 [+0.038, +0.055] | +0.007 |
| Gemma − QuantTrio | -0.0306 [-0.0449, -0.0165] | +0.520 [+0.468, +0.575] | +0.145 [+0.121, +0.169] | +0.052 [+0.041, +0.064] | +0.015 |
| GPTQ − fast | -0.0039 [-0.0146, +0.0068] | +0.015 [+0.006, +0.024] | +0.003 [-0.005, +0.010] | +0.004 [-0.000, +0.007] | +0.005 |
| Gemma − fast | +0.0045 [-0.0104, +0.0195] | +0.370 [+0.317, +0.426] | -0.002 [-0.027, +0.023] | +0.009 [-0.003, +0.021] | +0.013 |

The study renormalizes Jev's probabilities over its options, so its Jev accuracy differs
slightly from `run_split.py compare` (Emulator − Jev there: -0.0595 [-0.0720, -0.0472]).

**Δ NLL raw against Jev understates Jev's lead.** Jev rounds its probabilities to 0.01. On 409
of the 17,340 items it gives the correct option exactly 0, which NLL clips to 1e-6 (13.8 nats
each). So Jev's raw NLL is inflated. With every system's probability of the correct answer
floored at 0.005 (half Jev's rounding step), Emulator − Jev is +0.149 instead of +0.047, and
Fast − Jev +0.284 instead of +0.197 (`reports/jev_vs_qwen`). Calibrated NLL is barely affected.

**Findings.**

1. **The emulator is as well calibrated as Jev once calibrated, but less accurate.** Its ECE
   falls from 0.067 to 0.035 (Jev: 0.072 to 0.048). The NLL and Brier gaps come from
   knowledge: the emulator scores 0.629 against 0.827 on MMLU-Pro, 0.616 against 0.808 on
   GPQA and 0.651 against 0.702 on LEXam, and is within 0.03 of Jev on the other 6
   benchmarks.
2. **The two 27B builds are still tied.** No metric separates cyankiwi from QuantTrio.
   QuantTrio stays the recommendation.
3. **The MoE builds trail the dense emulator by 0.03-0.04**, and their calibrated NLL by about
   0.15. The GPTQ build and the fast emulator are tied on every calibrated metric. Gemma is
   tied with the fast emulator on accuracy and, calibrated, on NLL and Brier; before
   calibration its NLL is 0.37 worse.
4. **One-call scoring made the probabilities slightly worse than the multi-call runs.** Against
   the multi-call grid [below](#jev-and-the-emulator-raw-vs-calibrated-holdout): the emulator's
   calibrated NLL rose from 0.666 to 0.688 and its gap to Jev from +0.112 to +0.133; the fast
   emulator's from 0.761 to 0.835 (gap +0.206 → +0.280). Most of this is on banking77 and
   CLINC150, where the codes replaced the letters
   ([selection.md](selection.md#what-one-call-scoring-changed)). Calibrated ECE did not get
   worse (emulator 0.039 → 0.035, fast 0.037 → 0.037).
5. **Temperatures.** The fitted global temperature is 1.269 for the emulator (Jev 1.263), 1.292
   for the fast emulator, 1.325 for GPTQ and 1.681 for Gemma. Per signature, AG News needs the
   most for the Qwen builds (emulator 2.112, fast 1.948, GPTQ 1.990), SST-5 for Gemma (3.568).
6. **Best arm:** `vector@signature` for every system (accuracy / NLL): Jev 0.8301 / 0.509,
   emulator 0.7792 / 0.634, cyankiwi 0.7757 / 0.635, fast 0.7456 / 0.766, GPTQ 0.7425 /
   0.770, Gemma 0.7481 / 0.764. It learns each benchmark's label mix, so it is not deployed
   ([Recommended deployment](#recommended-deployment)).
7. **As returned** (Jev-style, not renormalized; NLL / Brier / ECE of the top probability /
   ECE of `confidence`): Jev 0.7031 / 0.2632 / 0.0719 / 0.0935, emulator 0.7503 / 0.3284 /
   0.0669 / 0.0802, fast 0.9001 / 0.3751 / 0.0766 / 0.0869, GPTQ 0.9148 / 0.3825 / 0.0874 /
   0.0910, Gemma 1.2701 / 0.4259 / 0.1646 / 0.1634.
8. **Hosted models against the one-call emulator** (`run_split.py compare`, holdout, the
   Structured Outputs runs; their calibration studies below are history on the earlier free
   read): GPT-6 Luna − QuantTrio -0.0117 [-0.0311, +0.0079] over its 6 benchmarks, raw NLL
   +0.516; DeepSeek V4.1 Flash − QuantTrio +0.0052 [-0.0118, +0.0223] over its 7 (the 13,547
   items it answered), raw NLL +0.060. Both are tied with the emulator on accuracy.

**Registries.** Each finalist's `temperature@signature` registry is refitted on its one-call
`holdout` run: 12 signatures each, fallback T 1.269 (QuantTrio), 1.275 (cyankiwi), 1.292
(fast), 1.325 (GPTQ), 1.681 (Gemma).

## Commands

```bash
# Jev on holdout (network only; the response cache makes a rerun free)
uv run python scripts/run_split.py run --system jev --benchmarks all --split holdout \
    --out runs/select --max-usd 1.0

# The emulator on holdout: one call per question, no debiaser
eval "$(scripts/serve_vllm.sh --preset qwen3.6-27b-int4-quanttrio | grep '^export ')"
uv run python scripts/run_split.py run --system emulator \
    --system-id qwen3.6-27b-int4-quanttrio.auto_single.state_first \
    --renderer state_first --strategy auto_single \
    --benchmarks all --split holdout --out runs/select --concurrency 16
docker compose -f docker/vllm/compose.yaml down

# Cross-fitted calibration: the whole study, one command, any number of run directories of one
# split (the finalists, as scripts/make_report.py reads it)
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first \
    gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_single.state_first \
    gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_single.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_single

# The same runs paired against QuantTrio (with the cyankiwi 27B) and against the fast emulator
uv run python scripts/calibrate.py crossfit \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first \
    cyankiwi=runs/select/qwen3.6-27b-int4-cyankiwi.auto_single.state_first \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first \
    gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_single.state_first \
    gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_single.state_first jev=runs/select/jev \
    --split holdout --reference quanttrio --debias none \
    --out runs/calibration/holdout_single_vs_quanttrio
uv run python scripts/calibrate.py crossfit \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_single.state_first \
    gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_single.state_first \
    gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_single.state_first \
    --split holdout --reference fast --debias none --out runs/calibration/holdout_single_vs_fast

# Multi-call history (before one-call scoring): the debiasing study ran the 27B builds with
# --strategy auto_noecho --debiaser pride:alpha=0.1 (run directories
# <preset>.auto_noecho.state_first.pride) and crossfit --debias none,batch,pride
# --out runs/calibration/holdout

# Jev's confidence function (J3): fit on select, check on holdout
uv run python scripts/calibrate.py confidence runs/select/jev --fit-split select \
    --eval-split holdout

# Deployment registry on the whole holdout run. --priors none fits it on the strategy's raw
# probabilities (no debiaser); --priors PRIORS.json on raw / priors (a batch or pride debiaser
# loading that file); without --priors, on the recorded probabilities.
uv run python scripts/calibrate.py registry \
    runs/select/qwen3.6-27b-int4-quanttrio.auto_single.state_first \
    --calibrator temperature --priors none \
    --out calibration/qwen3.6-27b-int4-quanttrio/registry.json
# Debiasing priors, if a debiaser is wanted (not recommended, see below)
uv run python scripts/calibrate.py priors runs/select/RUN --method pride --out priors.json
```

`crossfit` writes three files under `--out`:

- `report.md`: the tables below.
- `metrics.json`: every estimate and interval, the reliability bins, and every fitted
  calibrator per fold.
- `predictions.<system>.jsonl`: per item, the fold, gold answer, keys and every arm's
  probabilities.

It takes 88 s for Jev's 19,840 holdout items with 10,000 resamples, 341 s for Jev plus one
emulator run's four systems, and 604 s with both emulator runs.

## Method

An *arm* is one way of producing final probabilities: `raw`, or a calibrator at a scope, such
as `temperature@signature`.

- **Items.** Every item that all runs answered. Probabilities are read in the question's key
  order from the frozen split file and renormalized to sum to 1 (Jev rounds to 0.01). The
  predicted key is the first one with the highest probability. So on near-tie items, accuracy
  can differ from `run_split.py summarize`, which scores Jev's reported `choice` (LEXam:
  0.7023 here against 0.6990).
- **Folds.** Five folds per benchmark, split by question id and stratified by stratum. Within
  each stratum, question ids are sorted by `sha256(seed, benchmark, question_id)` and dealt
  round-robin, continuing across strata. All permutations of a question land in the same
  fold. The folds depend only on the split, so every system gets the same folds.
- **Arms.** `raw` is the probabilities as recorded. Every other arm is out of fold: the
  calibrator for fold *f* is fitted on the other four folds. For both systems the input is
  `log(clip(p, 1e-4, 1))`, so no probability is below the design's floor ε = 1e-4. Where one
  group mixes option counts, missing options are padded with `-inf`. Scopes:
  - `@global`: one calibrator.
  - `@benchmark`: one per benchmark.
  - `@signature`: one per question signature, pooled across benchmarks. This is the key the
    calibrator registry uses. For example, GPQA and LEXam share the A–E signature.

  A group with fewer than 30 training items falls back to the fold's global temperature. The
  calibrators are the registered ones: `identity` (the ε floor only), `temperature`,
  `vector`, `platt`, `isotonic` and `histogram`. The main arm is `temperature@signature`,
  the design's recommendation; it is the "calibrated" column of the 2×2 grid.
- **Metrics.** NLL is `-log max(p_gold, 1e-6)`. Brier is the multiclass version. ECE uses the
  top label and 10 bins with equal numbers of items. `metrics.json` holds the reliability
  bins.
- **Intervals.** 95% percentile intervals from 10,000 cluster bootstrap resamples (by question
  id), with a fixed seed. All systems and arms of a benchmark share the resamples, so every Δ
  is paired. The macro average weighs benchmarks equally, and its resamples are stratified by
  benchmark.

  ECE is reported as a point estimate only. Resampling adds binning noise, which pushes
  resampled ECEs above the estimate. The interval of an ECE, or of a Δ ECE, can then exclude
  the estimate itself, so `metrics.json` keeps those intervals as rough guides only. The
  design's ECE noise floor is about 0.36/√(items per bin): 0.11 for GPQA (99 items), 0.02 for
  MMLU-Pro.
- **Offline debiasing** (emulator runs with diagnostics):
  - `EMU` is the run as recorded (in the multi-call study, with online PriDe).
  - `EMU+none` uses the strategy's raw probabilities: the plain emulator. It is left out when
    it equals `EMU` (a run without a debiaser).
  - `EMU+batch` and `EMU+pride` divide by priors fitted out of fold, like the calibrators. If
    the run recorded an `EMU+pride` question under every cyclic shift of its options, the
    answer is the average over those shifts.

  The report's notes say which system is which. A "Debiasing" table gives each debiased
  system minus `EMU+none`, paired.
- **As returned.** A short section scores each run the way a Jev user sees it:
  - NLL, Brier and ECE of the returned probabilities, not renormalized;
  - the ECE of the answers' `confidence` field, read as the probability that the returned
    answer is correct (Choice: `choice`; Score: the most likely level). Noul answers carry
    no `confidence`, so this macro averages nine benchmarks.

## Jev on holdout

`jev-1.13.0` answered all 19,840 `holdout` items with 0 errors:

- **Time.** 07:26 to 07:43 UTC on Sep 25, 2026, with 16 requests in flight (20 items/s, Jev's
  rate limit), at jevemu commit `bb7241b`.
- **Calls.** 19,832 network calls; 8 answers came from the response cache.
- **Spend: $0.5106**, within the $1.00 cap.

The table comes from `run_split.py summarize`, which scores Jev's reported `choice`. IDK rate
is how often Jev picked "I don't know". MAE (mean absolute error) and Within-1 apply to the
rating benchmarks and are measured in score levels:

| Benchmark | n | Accuracy [95% CI] | IDK rate | NLL | Brier | ECE | MAE | Within-1 | Spend $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.8081 [0.7273, 0.8788] | 0.040 | 0.633 | 0.310 | 0.131 |  |  | 0.0023 |
| lexam_en_idk | 309 | 0.6990 [0.6472, 0.7508] | 0.016 | 0.797 | 0.393 | 0.091 |  |  | 0.0066 |
| mmlu_pro | 6016 | 0.8276 [0.8178, 0.8368] |  | 0.745 | 0.265 | 0.048 |  |  | 0.1402 |
| arc_challenge | 586 | 0.9795 [0.9625, 0.9881] |  | 0.097 | 0.032 | 0.008 |  |  | 0.0093 |
| ag_news | 3800 | 0.8884 [0.8779, 0.8976] |  | 0.672 | 0.189 | 0.067 |  |  | 0.0572 |
| banking77 | 1540 | 0.7864 [0.7643, 0.8058] |  | 1.280 | 0.326 | 0.095 |  |  | 0.0662 |
| clinc150 | 2250 | 0.9196 [0.9071, 0.9302] |  | 0.413 | 0.126 | 0.013 |  |  | 0.1338 |
| boolq | 1635 | 0.9229 [0.9083, 0.9346] |  | 0.227 | 0.127 | 0.030 |  |  | 0.0279 |
| sst5 | 1105 | 0.5937 [0.5629, 0.6208] |  | 1.462 | 0.600 | 0.168 | 0.483 | 0.957 | 0.0157 |
| yelp_stars | 2500 | 0.6692 [0.6508, 0.6872] |  | 0.945 | 0.470 | 0.134 | 0.382 | 0.978 | 0.0515 |
| **macro (10)** | 19840 | 0.8094 [0.7987, 0.8196] |  | 0.727 [0.693, 0.762] | 0.284 [0.273, 0.295] |  |  |  | 0.5106 |

Holdout is close to select: macro accuracy 0.8094 against 0.7955, NLL 0.727 against 0.751,
Brier 0.284 against 0.301. GPQA is the exception: its two 99-item halves differ by 0.16 in
accuracy (0.6465 on select), and its confidence interval is about ±0.08 wide.

## The emulator on holdout

*Multi-call history (before one-call scoring): this run used `auto_noecho` and online PriDe.
The current run uses `auto_single` and no debiaser.*

The selected configuration answered all 19,840 `holdout` items with 0 errors. It ran with
online PriDe, so one pass also gives the plain emulator and the offline debiasing arms:

- **System.** Preset `qwen3.6-27b-int4-quanttrio` (`QuantTrio/Qwen3.6-27B-AWQ` at `9b507bd`,
  vLLM 0.30.0), template `state_first-5d29289f8b87`, strategy `auto_noecho_tau0.001`,
  debiaser `pride(alpha=0.1, seed=0, max_options=32)`, confidence `mode_distance`. Run
  directory `runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride`.
- **Time.** 14:11:26 to 16:16:20 UTC on Sep 25, 2026 (2 h 5 min), 16 items in flight, at
  jevemu commit `89f3aae` plus this change set.
- **PriDe cost.** 31,903 backend calls, of which 6,460 are PriDe's permuted presentations of
  1,091 questions (predicted: about 6,500): +25% calls. MMLU-Pro took 2,057 s against 1,121 s
  without PriDe on `select`.
- **GPU time: 2 h 15 min** (14:09:30 to 16:24:28 UTC): 1.9 min startup, the run, then 8 min
  of an idle server while the study, the deployment registry and the smoke test ran. Before
  that, 1 min of Qwen3-0.6B re-recorded the emulator golden fixtures for the new defaults.

`run_split.py summarize` scores the recorded answers, which are online PriDe's: macro accuracy
0.7498 [0.7376, 0.7618], NLL 0.762. The tables below use these systems:

- `emulator`: the plain emulator (`quanttrio+none`, the strategy's probabilities). This is
  the configuration selection chose.
- `+PriDe online`: as recorded.
- `+PriDe offline`, `+batch`: priors fitted out of fold.

## Jev and the emulator, raw vs calibrated (holdout)

*Multi-call history (before one-call scoring): the emulator row is the `auto_noecho` run. The
current grid is in [Current results](#current-results-one-call-per-question-holdout).*

The 2×2 grid below is the macro average over **9 benchmarks (Yelp dropped)**, 17,340 paired
items. "Calibrated" means `temperature@signature`. Calibration does not change accuracy.
Command: `scripts/calibrate.py crossfit ... --debias none --out runs/calibration/holdout_noyelp`
(the loaders now skip Yelp).

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated | Δ ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 0.8253 [0.8138, 0.8365] | 0.703 [0.666, 0.741] | 0.555 [0.530, 0.581] | -0.148 [-0.166, -0.131] | 0.263 [0.252, 0.275] | 0.255 [0.244, 0.266] | -0.008 [-0.010, -0.006] | 0.072 | 0.048 | -0.024 |
| Emulator | 0.7673 [0.7542, 0.7805] | 0.723 [0.695, 0.752] | 0.666 [0.642, 0.691] | -0.057 [-0.064, -0.049] | 0.324 [0.311, 0.337] | 0.315 [0.303, 0.328] | -0.009 [-0.010, -0.007] | 0.071 | 0.039 | -0.032 |
| **Emulator − Jev** | -0.0580 [-0.0710, -0.0451] | +0.020 [-0.006, +0.046] | +0.112 [+0.094, +0.129] | | +0.061 [+0.052, +0.070] | +0.060 [+0.051, +0.069] | | -0.001 | -0.009 | |
| *Jev, with Yelp (10, superseded)* | 0.8097 [0.7992, 0.8200] | 0.727 [0.693, 0.762] | 0.575 [0.552, 0.598] | -0.152 [-0.169, -0.136] | 0.284 [0.273, 0.295] | 0.273 [0.263, 0.283] | -0.011 [-0.013, -0.009] | 0.078 | 0.047 | -0.031 |
| *Emulator, with Yelp (10, superseded)* | 0.7512 [0.7393, 0.7632] | 0.767 [0.741, 0.793] | 0.692 [0.670, 0.715] | -0.075 [-0.082, -0.068] | 0.352 [0.339, 0.363] | 0.335 [0.324, 0.346] | -0.016 [-0.018, -0.015] | 0.086 | 0.043 | -0.043 |
| *Emulator − Jev, with Yelp* | -0.0585 [-0.0703, -0.0468] | +0.040 [+0.016, +0.064] | +0.118 [+0.102, +0.134] | | +0.068 [+0.059, +0.076] | +0.062 [+0.054, +0.070] | | +0.008 | -0.004 | |

Dropping Yelp barely moves the accuracy gap (-0.0585 → -0.0580). It narrows the calibrated NLL
gap (+0.118 → +0.112), and the emulator's calibrated ECE goes from 0.043 to 0.039.

**From here until "GPT-6 Luna on holdout", the tables are the 10-benchmark study (with Yelp)
as recorded.**

Every calibrator, macro average. Jev:

| Arm | Accuracy | NLL | Brier | ECE | Δ NLL vs raw |
| --- | --- | --- | --- | --- | --- |
| `raw` | 0.8097 [0.7992, 0.8200] | 0.727 [0.693, 0.762] | 0.284 [0.273, 0.295] | 0.078 |  |
| `identity@global` (ε floor only) | 0.8097 [0.7992, 0.8200] | 0.642 [0.616, 0.670] | 0.284 [0.273, 0.294] | 0.076 | -0.085 [-0.095, -0.075] |
| `temperature@global` | 0.8097 [0.7992, 0.8200] | 0.617 [0.595, 0.639] | 0.285 [0.275, 0.294] | 0.084 | -0.110 [-0.125, -0.096] |
| `temperature@benchmark` | 0.8097 [0.7991, 0.8200] | 0.573 [0.548, 0.601] | 0.272 [0.261, 0.283] | 0.042 | -0.154 [-0.171, -0.136] |
| `temperature@signature` | 0.8097 [0.7991, 0.8200] | 0.575 [0.552, 0.598] | 0.273 [0.263, 0.283] | 0.047 | -0.152 [-0.169, -0.136] |
| `vector@signature` | 0.8163 [0.8061, 0.8265] | 0.530 [0.510, 0.551] | 0.258 [0.248, 0.268] | 0.040 | -0.197 [-0.216, -0.179] |
| `platt@signature` | 0.8082 [0.7976, 0.8184] | 0.592 [0.568, 0.618] | 0.272 [0.262, 0.283] | 0.036 | -0.135 [-0.149, -0.121] |
| `isotonic@signature` | 0.8075 [0.7969, 0.8178] | 0.599 [0.573, 0.626] | 0.274 [0.264, 0.285] | 0.035 | -0.128 [-0.142, -0.116] |
| `histogram@signature` | 0.8077 [0.7969, 0.8179] | 0.603 [0.577, 0.630] | 0.276 [0.265, 0.287] | 0.035 | -0.124 [-0.138, -0.111] |

The emulator:

| Arm | Accuracy | NLL | Brier | ECE | Δ NLL vs raw |
| --- | --- | --- | --- | --- | --- |
| `raw` | 0.7512 [0.7393, 0.7632] | 0.767 [0.741, 0.793] | 0.352 [0.339, 0.363] | 0.086 |  |
| `identity@global` (ε floor only) | 0.7512 [0.7393, 0.7632] | 0.764 [0.738, 0.790] | 0.351 [0.339, 0.363] | 0.086 | -0.003 [-0.005, -0.002] |
| `temperature@global` | 0.7512 [0.7393, 0.7632] | 0.731 [0.710, 0.751] | 0.346 [0.335, 0.356] | 0.084 | -0.036 [-0.043, -0.030] |
| `temperature@benchmark` | 0.7512 [0.7393, 0.7632] | 0.692 [0.669, 0.715] | 0.335 [0.324, 0.347] | 0.041 | -0.075 [-0.083, -0.068] |
| `temperature@signature` | 0.7512 [0.7393, 0.7632] | 0.692 [0.670, 0.715] | 0.335 [0.324, 0.346] | 0.043 | -0.075 [-0.082, -0.068] |
| `vector@signature` | 0.7714 [0.7597, 0.7830] | 0.628 [0.606, 0.649] | 0.312 [0.301, 0.324] | 0.040 | -0.140 [-0.151, -0.129] |
| `platt@signature` | 0.7492 [0.7374, 0.7611] | 0.697 [0.674, 0.720] | 0.334 [0.323, 0.345] | 0.038 | -0.070 [-0.077, -0.063] |
| `isotonic@signature` | 0.7497 [0.7378, 0.7617] | 0.706 [0.681, 0.731] | 0.336 [0.325, 0.347] | 0.032 | -0.061 [-0.070, -0.051] |
| `histogram@signature` | 0.7460 [0.7339, 0.7581] | 0.707 [0.683, 0.732] | 0.337 [0.326, 0.349] | 0.027 | -0.060 [-0.069, -0.049] |

Best arm per system (macro average; the offline-debiased systems agree):

| System | Accuracy | NLL | Brier | ECE |
| --- | --- | --- | --- | --- |
| Jev | `vector@signature` 0.8163 | `vector@signature` 0.530 | `vector@signature` 0.258 | `histogram@signature` 0.035 |
| Emulator | `vector@signature` 0.7714 | `vector@signature` 0.628 | `vector@signature` 0.312 | `histogram@signature` 0.027 |
| Emulator +PriDe online | `vector@signature` 0.7697 | `vector@signature` 0.630 | `vector@signature` 0.313 | `platt@signature` 0.034 |
| Emulator +batch | `vector@signature` 0.7721 | `vector@signature` 0.624 | `vector@signature` 0.312 | `isotonic@signature` 0.031 |
| Emulator +PriDe offline | `vector@signature` 0.7704 | `vector@signature` 0.630 | `vector@signature` 0.313 | `histogram@signature` 0.031 |

With `vector@signature` on both, the emulator − Jev gap is: accuracy −0.0450
[−0.0567, −0.0336], NLL +0.097 [+0.081, +0.113], Brier +0.055 [+0.046, +0.063], ECE +0.000.

Per benchmark, Jev against the emulator. Accuracy is raw; NLL and Brier are calibrated
(`temperature@signature`); Δ is emulator − Jev, paired:

| Benchmark | n | Acc. Jev | Acc. emulator | Δ accuracy | NLL cal. Jev | NLL cal. emulator | Δ NLL cal. | Brier cal. Jev | Brier cal. emulator | ECE raw Jev | ECE cal. Jev | ECE raw emulator | ECE cal. emulator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.8081 | 0.5859 | -0.2222 [-0.3232, -0.1313] | 0.624 | 0.915 | +0.291 [+0.188, +0.400] | 0.306 | 0.486 | 0.131 | 0.121 | 0.118 | 0.124 |
| lexam_en_idk | 309 | 0.7023 | 0.6570 | -0.0453 [-0.1003, +0.0097] | 0.770 | 0.858 | +0.089 [-0.003, +0.174] | 0.394 | 0.464 | 0.089 | 0.076 | 0.068 | 0.067 |
| mmlu_pro | 6016 | 0.8273 | 0.6300 | -0.1973 [-0.2098, -0.1855] | 0.651 | 1.127 | +0.475 [+0.445, +0.505] | 0.269 | 0.482 | 0.047 | 0.062 | 0.051 | 0.013 |
| arc_challenge | 586 | 0.9795 | 0.9778 | -0.0017 [-0.0154, +0.0119] | 0.085 | 0.075 | -0.010 [-0.041, +0.018] | 0.035 | 0.034 | 0.008 | 0.024 | 0.017 | 0.019 |
| ag_news | 3800 | 0.8884 | 0.8821 | -0.0063 [-0.0116, -0.0013] | 0.362 | 0.404 | +0.042 [+0.031, +0.052] | 0.179 | 0.189 | 0.067 | 0.019 | 0.083 | 0.017 |
| banking77 | 1540 | 0.7864 | 0.7857 | -0.0006 [-0.0149, +0.0136] | 0.907 | 0.959 | +0.052 [+0.005, +0.098] | 0.312 | 0.329 | 0.095 | 0.046 | 0.090 | 0.034 |
| clinc150 | 2250 | 0.9196 | 0.9071 | -0.0124 [-0.0222, -0.0027] | 0.355 | 0.372 | +0.018 [-0.015, +0.051] | 0.126 | 0.143 | 0.013 | 0.009 | 0.024 | 0.026 |
| boolq | 1635 | 0.9229 | 0.9028 | -0.0202 [-0.0318, -0.0086] | 0.223 | 0.268 | +0.045 [+0.028, +0.061] | 0.125 | 0.153 | 0.030 | 0.018 | 0.021 | 0.018 |
| sst5 | 1105 | 0.5937 | 0.5774 | -0.0163 [-0.0389, +0.0063] | 1.016 | 1.021 | +0.005 [-0.021, +0.030] | 0.551 | 0.558 | 0.168 | 0.053 | 0.167 | 0.034 |
| yelp_stars | 2500 | 0.6692 | 0.6064 | -0.0628 [-0.0800, -0.0456] | 0.754 | 0.926 | +0.172 [+0.151, +0.194] | 0.435 | 0.512 | 0.134 | 0.041 | 0.224 | 0.077 |
| **macro** | 19840 | 0.8097 | 0.7512 | -0.0585 [-0.0703, -0.0468] | 0.575 | 0.692 | +0.118 [+0.102, +0.134] | 0.273 | 0.335 | 0.078 | 0.047 | 0.086 | 0.043 |

Raw → calibrated per benchmark, Jev:

| Benchmark | n | NLL raw | NLL cal. | Δ NLL | Brier raw | Brier cal. | Δ Brier | ECE raw | ECE cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.633 | 0.624 | -0.009 [-0.014, -0.004] | 0.310 | 0.306 | -0.004 [-0.006, -0.002] | 0.131 | 0.121 |
| lexam_en_idk | 309 | 0.797 | 0.770 | -0.027 [-0.069, +0.001] | 0.393 | 0.394 | +0.000 [-0.001, +0.002] | 0.089 | 0.076 |
| mmlu_pro | 6016 | 0.745 | 0.651 | -0.094 [-0.115, -0.074] | 0.265 | 0.269 | +0.004 [+0.003, +0.005] | 0.047 | 0.062 |
| arc_challenge | 586 | 0.097 | 0.085 | -0.012 [-0.055, +0.021] | 0.032 | 0.035 | +0.003 [-0.000, +0.005] | 0.008 | 0.024 |
| ag_news | 3800 | 0.672 | 0.362 | -0.311 [-0.365, -0.257] | 0.189 | 0.179 | -0.010 [-0.014, -0.006] | 0.067 | 0.019 |
| banking77 | 1540 | 1.280 | 0.907 | -0.373 [-0.455, -0.293] | 0.326 | 0.312 | -0.014 [-0.018, -0.009] | 0.095 | 0.046 |
| clinc150 | 2250 | 0.413 | 0.355 | -0.059 [-0.083, -0.036] | 0.126 | 0.126 | -0.000 [-0.000, +0.000] | 0.013 | 0.009 |
| boolq | 1635 | 0.227 | 0.223 | -0.003 [-0.008, +0.001] | 0.127 | 0.125 | -0.002 [-0.003, -0.001] | 0.030 | 0.018 |
| sst5 | 1105 | 1.462 | 1.016 | -0.446 [-0.561, -0.337] | 0.600 | 0.551 | -0.049 [-0.066, -0.032] | 0.168 | 0.053 |
| yelp_stars | 2500 | 0.945 | 0.754 | -0.191 [-0.235, -0.148] | 0.470 | 0.435 | -0.035 [-0.043, -0.027] | 0.134 | 0.041 |

The emulator:

| Benchmark | n | NLL raw | NLL cal. | Δ NLL | Brier raw | Brier cal. | Δ Brier | ECE raw | ECE cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.913 | 0.915 | +0.002 [-0.001, +0.005] | 0.486 | 0.486 | +0.000 [-0.001, +0.002] | 0.118 | 0.124 |
| lexam_en_idk | 309 | 0.855 | 0.858 | +0.003 [+0.000, +0.005] | 0.463 | 0.464 | +0.001 [-0.000, +0.002] | 0.068 | 0.067 |
| mmlu_pro | 6016 | 1.145 | 1.127 | -0.018 [-0.024, -0.013] | 0.486 | 0.482 | -0.004 [-0.005, -0.002] | 0.051 | 0.013 |
| arc_challenge | 586 | 0.073 | 0.075 | +0.002 [+0.001, +0.003] | 0.034 | 0.034 | +0.000 [+0.000, +0.001] | 0.017 | 0.019 |
| ag_news | 3800 | 0.605 | 0.404 | -0.202 [-0.235, -0.170] | 0.206 | 0.189 | -0.017 [-0.021, -0.013] | 0.083 | 0.017 |
| banking77 | 1540 | 1.079 | 0.959 | -0.120 [-0.155, -0.086] | 0.342 | 0.329 | -0.013 [-0.017, -0.008] | 0.090 | 0.034 |
| clinc150 | 2250 | 0.365 | 0.372 | +0.007 [+0.001, +0.012] | 0.143 | 0.143 | -0.000 [-0.000, +0.000] | 0.024 | 0.026 |
| boolq | 1635 | 0.271 | 0.268 | -0.003 [-0.008, +0.002] | 0.153 | 0.153 | +0.000 [-0.001, +0.002] | 0.021 | 0.018 |
| sst5 | 1105 | 1.201 | 1.021 | -0.180 [-0.224, -0.138] | 0.604 | 0.558 | -0.047 [-0.061, -0.033] | 0.167 | 0.034 |
| yelp_stars | 2500 | 1.165 | 0.926 | -0.239 [-0.269, -0.209] | 0.599 | 0.512 | -0.086 [-0.096, -0.077] | 0.224 | 0.077 |

Fitted temperatures of `temperature@signature` (T > 1 softens an overconfident system). The
Jev and Emulator columns are cross-fitted: the mean over the five folds. The last column is the
deployment registry, fitted on the whole run (next section). Signatures are pooled across
benchmarks: the letter-keyed A–D and A–E signatures mix ARC, MMLU-Pro's short questions, GPQA
and LEXam.

| Signature | Benchmarks | Jev | Emulator | Emulator, deployed |
| --- | --- | --- | --- | --- |
| `choice:10:4a0bcc85ad6db29b` | MMLU-Pro (10 options) | 1.09 | 1.22 | 1.220 |
| `choice:9:926eb22cfa3771fa`, `8:…`, `7:…`, `6:…` | MMLU-Pro (9, 8, 7, 6 options) | 1.43, 1.01, 1.15, 1.12 | 1.26, 1.03, 1.22, 1.22 | 1.256, 1.029, 1.222, 1.220 |
| `choice:4:503f248273fbb6c6` | ARC, MMLU-Pro (A–D) | 1.61 | 1.01 | 1.015 |
| `choice:5:62106b9c5cb2462d` | GPQA, LEXam (A–E, E = IDK) | 0.96 | 1.01 | 1.015 |
| `choice:4:cf9e907ab30ae267` | AG News | 2.13 | 2.11 | 2.112 |
| `choice:77:0a3f622029033ef1` | banking77 | 1.34 | 1.31 | 1.309 |
| `choice:150:6ad22f70c3e44a2a` | CLINC150 | 1.01 | 1.00 | 0.997 |
| `noul:2:da39667d142647f1` | BoolQ (every Noul question) | 0.84 | 1.18 | 1.179 |
| `score:5:6276777a0d628fc5` | SST-5 | 2.38 | 1.96 | 1.959 |
| `score:5:914aeb8045b2218c` | Yelp | 1.89 | 2.01 | 2.010 |
| (fallback, any other signature) | all items | 1.30 (`@global`) | 1.33 (`@global`) | 1.326 |

**Findings.**

1. **Jev's miscalibration depends on the benchmark, and held-out calibration fixes most of
   it.** Per-signature temperature cuts macro NLL by 0.152 [0.136, 0.169] and Brier by
   0.011 [0.009, 0.013], with accuracy unchanged. ECE falls from 0.078 to 0.047.
2. **Both systems are overconfident on the classification and rating tasks.**
   - Fitted temperatures on AG News, SST-5, Yelp and banking77: Jev 2.1, 2.4, 1.9 and 1.3;
     the emulator 2.1, 2.0, 2.0 and 1.3.
   - Jev is close to calibrated on CLINC150 (1.01) and MMLU-Pro (1.09), and underconfident
     on BoolQ (0.84).
   - The emulator is calibrated on CLINC150 and the letter-keyed A–D/A–E questions
     (1.00-1.01), and overconfident on MMLU-Pro (1.22) and BoolQ (1.18).
3. **Once calibrated, the emulator is as well calibrated as Jev, but less accurate.** Its ECE
   falls from 0.086 to 0.043 (Jev: 0.047). Yelp is the largest remaining miscalibration
   (0.224 → 0.077; Jev 0.041). The NLL and Brier gaps come from accuracy. The emulator trails
   on MMLU-Pro (−0.197), GPQA (−0.222) and Yelp (−0.063). It is within 0.02 of Jev on ARC,
   AG News, banking77, CLINC150, BoolQ and SST-5.
4. **The raw NLL comparison flatters the emulator.** Jev reports exact 0.00 probabilities.
   The ε = 1e-4 floor alone (`identity@global`) removes 0.085 of Jev's NLL but only 0.003 of
   the emulator's. Examples for Jev: banking77 1.280 → 1.006, AG News 0.672 → 0.522, SST-5
   1.462 → 1.291, MMLU-Pro 0.745 → 0.661. So the raw gap is +0.040, and the fair comparison
   is the calibrated gap, +0.118 [0.102, 0.134]. On MMLU-Pro and CLINC150 the floor is almost
   all of Jev's gain (0.084 of 0.094, and all of 0.059). There, temperature scaling slightly
   raises Jev's Brier and ECE (MMLU-Pro: Brier +0.004, ECE 0.047 → 0.062).
5. **One global temperature is not enough for either system.** `temperature@global` leaves
   ECE at 0.084 for both (raw: Jev 0.078, emulator 0.086), because it also softens the
   benchmarks that were already calibrated. Per-benchmark and per-signature temperatures
   perform the same, since signatures nearly coincide with benchmarks.
6. **The arms rank the same for both systems.**
   - `vector@signature` has the best NLL, Brier and accuracy. It adds 0.0066 accuracy for
     Jev and 0.0201 for the emulator, whose position and label biases are larger. Its
     per-class bias can change the top answer, so it is more than a calibrator: it is a
     supervised correction of the label prior. On `select` it also added 0.026 accuracy to
     cyankiwi.
   - The top-label maps (`platt`, `isotonic`, `histogram`) have the best ECE (0.027-0.038),
     but worse NLL than temperature.

### Does debiasing earn its cost?

Each debiased emulator minus the plain emulator, macro average, paired. Rows cover the online
run's own debiasing and the two offline variants:

| System | Arm | Δ Accuracy | Δ NLL | Δ Brier | Δ ECE |
| --- | --- | --- | --- | --- | --- |
| +PriDe online | `raw` | -0.0014 [-0.0077, +0.0047] | -0.006 [-0.010, -0.001] | -0.001 [-0.003, +0.002] | -0.003 |
| +PriDe online | `temperature@signature` | -0.0014 [-0.0077, +0.0047] | -0.003 [-0.008, +0.002] | -0.001 [-0.003, +0.002] | +0.001 |
| +PriDe online | `vector@signature` | -0.0017 [-0.0068, +0.0032] | +0.003 [-0.001, +0.007] | +0.001 [-0.001, +0.003] | -0.001 |
| +PriDe offline | `raw` | -0.0011 [-0.0074, +0.0051] | -0.005 [-0.010, -0.001] | -0.001 [-0.003, +0.002] | -0.003 |
| +PriDe offline | `temperature@signature` | -0.0011 [-0.0074, +0.0051] | -0.003 [-0.008, +0.002] | -0.000 [-0.003, +0.002] | -0.002 |
| +PriDe offline | `vector@signature` | -0.0009 [-0.0047, +0.0026] | +0.002 [-0.001, +0.007] | +0.001 [-0.001, +0.003] | -0.003 |
| +batch | `raw` | +0.0022 [-0.0032, +0.0074] | -0.049 [-0.054, -0.045] | -0.012 [-0.014, -0.010] | -0.012 |
| +batch | `temperature@signature` | +0.0022 [-0.0032, +0.0074] | -0.026 [-0.029, -0.024] | -0.007 [-0.009, -0.006] | -0.005 |
| +batch | `vector@signature` | +0.0007 [-0.0010, +0.0025] | -0.003 [-0.004, -0.002] | -0.000 [-0.001, -0.000] | -0.001 |

- **PriDe: no.** Online or offline, it changes no macro metric measurably after calibration.
  Online it costs 25% more backend calls; with fitted priors it costs 0 extra calls, still with
  no gain. It helps where the position prior is strong:
  - MMLU-Pro: calibrated NLL −0.035 [−0.045, −0.025]. The fitted prior puts 0.17-0.18 on A and
    E and 0.03 on H and I.
  - AG News: calibrated NLL −0.022, plus 0.0032 [0.0008, 0.0055] accuracy.

  GPQA moves the other way (+0.035 [−0.010, +0.081]), and the macro average nets to zero.
- **Batch helps only by learning each benchmark's label mix.** It divides every answer by the
  batch's mean prediction. That is right only when deployment traffic has the benchmark's
  label distribution. Its calibrated gain comes from CLINC150 (−0.075), Yelp (−0.071),
  banking77 (−0.069), MMLU-Pro (−0.023) and SST-5 (−0.020); it costs BoolQ +0.015. After
  `vector@signature`, which makes the same correction, it adds only −0.003.

### The tied finalist: cyankiwi

**One-call result** (current; [Current results](#current-results-one-call-per-question-holdout)):
cyankiwi − QuantTrio on the 9 benchmarks is -0.0014 [-0.0101, +0.0069] in accuracy and +0.003
[-0.005, +0.012] in calibrated NLL; calibrated ECE 0.032 against 0.035. The tie holds. The
registry [`calibration/qwen3.6-27b-int4-cyankiwi/registry.json`](../../calibration/qwen3.6-27b-int4-cyankiwi/registry.json)
is refitted on the one-call run: 12 signatures, fallback T = 1.275.

The rest of this section is multi-call history (before one-call scoring), 10 benchmarks with
Yelp.

`qwen3.6-27b-int4-cyankiwi` (`cyankiwi/Qwen3.6-27B-AWQ-INT4`) answered the
same 19,840 items the same way: online PriDe α = 0.1, 16:32:43 to 18:35:51 UTC, 0 errors,
31,594 backend calls of which 6,460 were PriDe's. **GPU time: 2 h 5 min** (16:30:38 to
18:35:52 UTC, 2.1 min startup). Plain emulator, macro average:

| System | Accuracy | NLL raw | NLL calibrated | Brier raw | Brier calibrated | ECE raw | ECE calibrated | `vector@signature` accuracy / NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QuantTrio | 0.7512 [0.7393, 0.7632] | 0.767 [0.741, 0.793] | 0.692 [0.670, 0.715] | 0.352 [0.339, 0.363] | 0.335 [0.324, 0.346] | 0.086 | 0.043 | 0.7714 / 0.628 |
| cyankiwi | 0.7506 [0.7383, 0.7626] | 0.760 [0.733, 0.788] | 0.697 [0.674, 0.721] | 0.351 [0.338, 0.363] | 0.337 [0.325, 0.349] | 0.076 | 0.035 | 0.7681 / 0.627 |
| **cyankiwi − QuantTrio** (paired) | -0.0006 [-0.0091, +0.0077] | -0.007 [-0.015, +0.002] | +0.005 [-0.002, +0.013] | -0.001 [-0.005, +0.003] | +0.002 [-0.002, +0.006] | -0.010 | -0.008 | -0.0033 [-0.0108, +0.0039] / -0.000 |

- **The tie holds.** No metric separates the two builds. Cyankiwi's lower calibrated ECE (0.035
  against 0.043) is a point estimate within ECE noise. QuantTrio stays the recommendation:
  selection chose it on `select`, and holdout gives no reason to change that. Cyankiwi is the
  equal alternative on the standard 27B launch (0.95 of the GPU).
- **Against Jev**, calibrated: accuracy −0.0591 [−0.0711, −0.0471], NLL +0.123
  [+0.106, +0.139], Brier +0.064 [+0.055, +0.073], ECE −0.012.
- **Debiasing: same verdict.** Online PriDe changes calibrated NLL by −0.008 [−0.014, −0.002]
  and accuracy by +0.0022 [−0.0038, +0.0084]. The NLL change is significant but negligible,
  and it costs 25% more calls. Batch: −0.029 [−0.032, −0.026] after temperature, −0.003 after
  `vector`.
- **As returned:** NLL 0.7511, Brier 0.3465, ECE 0.0767; `confidence` ECE 0.0867 (as
  recorded, with online PriDe).
- **Registry** (as first fitted): 13 signatures, fallback T = 1.324. It loaded and resolved
  offline. It was not smoke-tested against a live cyankiwi server.

### As returned (Jev-style)

Point estimates for the returned probabilities, not renormalized, and for `confidence` read as
the probability that the returned answer is correct:

| Run | NLL | Brier | ECE (top probability) | ECE (`confidence`) | Max Δ vs renormalized |
| --- | --- | --- | --- | --- | --- |
| Jev | 0.7272 | 0.2838 | 0.0781 | 0.1000 | 1.6e-04 |
| Emulator (+PriDe online, as recorded) | 0.7616 | 0.3507 | 0.0834 | 0.0925 | 2.8e-17 |

- Renormalizing Jev's two-decimal probabilities changes no macro metric by more than 0.0002.
- The `confidence` ECE averages the nine benchmarks with Choice or Score answers (BoolQ's Noul
  answers carry no `confidence`). Over the same nine, the top probability's ECE is 0.0835 for
  Jev and 0.0903 for the emulator.
- As a probability of being right, Jev's `confidence` is worse than its top probability
  (ECE 0.100). It is worst on GPQA (0.212 against 0.131) and SST-5/Yelp. `confidence` is a
  peakedness score (J3), not a probability, and so is the emulator's `mode_distance`. Set
  thresholds on calibrated probabilities instead.

## Recommended deployment

```python
from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.calibrate import CalibratorRegistry
from jevemu.emulator import Emulator
from jevemu.scoring import SingleCallStrategy

registry = CalibratorRegistry.load("calibration/qwen3.6-27b-int4-quanttrio/registry.json")
async with VLLMHTTPBackend(url, "QuantTrio/Qwen3.6-27B-AWQ") as backend:
    emulator = Emulator(backend, strategy=SingleCallStrategy(), debiaser=None,
                        calibrators=registry, round_to=0.01)
```

- **Configuration.** Preset `qwen3.6-27b-int4-quanttrio`, the default `state_first` layout,
  `mode_distance` confidence, `SingleCallStrategy` (one model call per question), no debiaser,
  and `round_to=0.01` for Jev-like output. The server needs `--max-logprobs 576`.
- **Calibration: `temperature@signature`**, fitted on the whole one-call holdout run's raw
  probabilities (`scripts/calibrate.py registry ... --calibrator temperature --priors none`).
  The registry holds one temperature for each signature with at least 30 items: 12
  signatures, from 0.989 (CLINC150) to 2.112 (AG News); SST-5 1.961, BoolQ 1.182, MMLU-Pro
  (10 options) 1.219. Every other signature of this model and template gets T = 1.269.
  Held-out estimate on the 9 benchmarks (cross-fitted): ECE 0.035, NLL 0.688, Brier 0.320,
  accuracy unchanged.
- **Why not `vector@signature`?** It scores better here (NLL 0.634, Brier 0.303, accuracy
  0.7792). But its biases encode the benchmarks' label mix, and the registry would apply them
  to every question with the same signature (the logit values below are from the multi-call
  study):
  - Every Noul question has signature `noul:2:da39667d142647f1`. The BoolQ-fitted vector would
    add +1.04 logits toward "yes" to all of them (62% of BoolQ's holdout questions are
    answered yes).
  - Every A–E letter question shares GPQA and LEXam's signature. Their vector subtracts 1.7
    logits from E, because their E ("I don't know") is never the gold answer.

  Temperature only rescales. It never changes an answer, and it transfers to new questions.
  Fit `vector` only on labelled traffic of your own question types (`--calibrator vector`).
- **No debiaser**: see above. The artifacts hold parameters only: 12 temperatures plus the
  fallback, keyed by backend, model and template id. They contain no per-item data.
- **Smoke test** (with the multi-call registry, which had the same signatures plus Yelp's). The
  registry was loaded into an `Emulator` against the live server. It
  answered all 19 Jev doc-example requests (`tests/golden/fixtures/jev_docs`, 41 questions).
  Every response validates as a `SystemOneResponse` with `model`
  `jevemu/QuantTrio/Qwen3.6-27B-AWQ`. The diagnostics record which calibrator was used:
  `temperature:exact` for the 14 Noul questions (the BoolQ signature), and
  `temperature:temperature_fallback` for the 27 Choice and Score questions, whose option keys
  are the docs' own.

## Jev's confidence function (J3)

**Result.** Jev's `confidence` is

$$\text{confidence} = \operatorname{clip}\left(1 - \frac{\mathbb{E}[d(X, \text{mode})]}{\min_c \mathbb{E}_{\text{uniform}}[d(X, c)]},\ 0,\ 1\right)$$

In words: it measures how far the probability mass lies from the most likely answer (the
mode), relative to how spread out a uniform distribution would be:

- **Choice:** `d` is the 0/1 distance and the normalizer is (K − 1)/K, so the formula is
  exactly the documented `peak_linear`, (K·p_max − 1)/(K − 1).
- **Score:** `d` is the level distance |i − j| and the normalizer is K/4 (even K) or
  (K² − 1)/(4K) (odd K): 1.2 for 5 levels. Mass one level from the mode costs less than
  mass four levels away.

This matches both ScoreExplorer answers in Jev's docs that `peak_linear` misses:
`[0, 0, 0.48, 0.52]` gives 0.52 exactly, and `[0, 0.14, 0.86, 0, 0]` gives 0.883 against a
reported 0.89. It is registered as `jevemu.confidence.mode_distance`
(`CONFIDENCE_FUNCTIONS["mode_distance"]`).

**Protocol change.** A `ConfidenceFn` is now called as `fn(probs, ordinal=...)`, with
`ordinal=True` for Score answers. The emulator passes it, and every built-in function accepts
it.

**Fit.** The fit used Jev's 18,206 select-half Choice and Score answers (the 1,635 Noul
answers carry no confidence). It was checked on the 18,205 holdout answers. Probabilities are
renormalized. Predictions are rounded to 0.01, as Jev rounds, and then compared with the
reported value. The candidates:

- the design's four families;
- `mode_distance`;
- a small fitted monotone family, `power(mode_distance)`: `mode_distance^γ`, with one γ per
  question kind, fitted by least squares on select.

Mean absolute error:

| Kind | K | n (select) | `peak_linear` | `max_prob` | `margin` | `1 − norm. entropy` | `mode_distance` | `power(mode_distance)` | holdout: `mode_distance` | holdout: `power` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| choice | 3 | 12 | 0.0033 | 0.0825 | 0.0708 | 0.0667 | 0.0033 | 0.0033 | 0.0031 | 0.0031 |
| choice | 4 | 4678 | 0.0019 | 0.0162 | 0.0265 | 0.0249 | 0.0019 | 0.0018 | 0.0019 | 0.0019 |
| choice | 5 | 438 | 0.0061 | 0.0760 | 0.0958 | 0.1081 | 0.0061 | 0.0063 | 0.0059 | 0.0066 |
| choice | 6 | 46 | 0.0043 | 0.0252 | 0.0550 | 0.0576 | 0.0043 | 0.0041 | 0.0055 | 0.0055 |
| choice | 7 | 86 | 0.0050 | 0.0188 | 0.0377 | 0.0359 | 0.0050 | 0.0047 | 0.0044 | 0.0046 |
| choice | 8 | 146 | 0.0052 | 0.0175 | 0.0436 | 0.0398 | 0.0052 | 0.0045 | 0.0048 | 0.0047 |
| choice | 9 | 397 | 0.0059 | 0.0186 | 0.0558 | 0.0401 | 0.0059 | 0.0053 | 0.0058 | 0.0054 |
| choice | 10 | 5008 | 0.0059 | 0.0256 | 0.0653 | 0.0495 | 0.0059 | 0.0056 | 0.0060 | 0.0057 |
| choice | 77 | 1540 | 0.0050 | 0.0061 | 0.0715 | 0.0470 | 0.0050 | 0.0040 | 0.0053 | 0.0040 |
| choice | 150 | 2250 | 0.0042 | 0.0044 | 0.0462 | 0.0334 | 0.0042 | 0.0034 | 0.0041 | 0.0032 |
| score | 5 | 3605 | 0.0732 | 0.0342 | 0.1940 | 0.1108 | **0.0052** | 0.0046 | 0.0053 | 0.0048 |
| **all** | | 18206 | 0.0179 | 0.0216 | 0.0792 | 0.0542 | **0.0044** | 0.0040 | 0.0045 | 0.0041 |

With `mode_distance`, 59.8% of select predictions equal the reported value and 95.9% are
within 0.01 (holdout: 59.6% and 95.7%). With `peak_linear`, 52.1% and 82.5%.

**Choice of function.** We register `mode_distance`, not the fitted power version
(γ = 1.014 for Choice, 1.013 for Score):

- **Where the power model's gain comes from.** Its 0.0004 gain fits the rounding of Jev's
  *displayed* probabilities, not Jev's function. Jev computes confidence before rounding, as
  shown by the docs' examples and the residuals (95.9% within 0.01). Every option displayed as
  0.00 hides up to 0.005 of mass, so renormalized displayed vectors overstate p_max.
- **The evidence.** For Choice answers, the residual (reported − predicted) drifts from +0.001
  with no 0.00 options (1,189 answers), to −0.001 with 1-3 of them (5,548), to −0.005 with 4
  or more (7,864).
- **Consequence.** The emulator's probabilities are not rounded, so the γ correction would
  bias its confidence.

**Default.** The emulator's default `confidence_fn` is now `mode_distance`
(`jevemu.confidence.DEFAULT_CONFIDENCE`). It changes only Score confidences. The holdout run
above is the first to record it in its manifest; the `select` runs recorded `peak_linear`.

## Debiasers

`Emulator(debiaser=...)` runs a debiaser after scoring and before calibration. Build one with
`jevemu.debias.debiaser_from_spec("pride:alpha=0.1")`, or with the classes directly.
Debiasers apply to the emulator only; Jev is a black box.

A *free* position is an option that may move. Options described exactly "I don't know" are
fixed: they stay last and are never divided by a prior. *F* is the number of free options;
MCQ means multiple-choice question.

| Debiaser | Method | Applies to | Extra backend passes |
| --- | --- | --- | --- |
| `permutation` | mean option probability over presentations. Default: all *F* cyclic shifts of the *F* free options; or `n` evenly spaced shifts, or `kind=random` | Choice with 2 ≤ *F* ≤ `max_options` (32) | *n* − 1 per question (*F* − 1 by default) |
| `pride` | divide by a position prior: the softmax of the mean log-probability over all cyclic shifts, averaged over the estimation questions. An estimation question is answered with its permutation average | Choice, as above | *F* − 1 for an α share of questions plus the first of each prior key, ≈ ×(1 + α(*F* − 1)); 0 with fitted `priors` |
| `contextual` | divide by the distribution for a content-free state (`"N/A"`), cached per distinct question | every type | `len(content_free)` once per distinct question: +1 per item for MCQ banks and BoolQ, +1 per benchmark for AG News, banking77, CLINC150, SST-5 and Yelp |
| `batch` | divide by the mean distribution of a recorded batch (priors fitted offline) | every type | 0 |

- **Diagnostics.** Each answer's `x_jevemu` records:
  - `debiaser` (its id);
  - `debias_calls` (included in `n_backend_calls`);
  - `debias_prior`;
  - `permutations`;
  - `permutation_scores`: every presentation, with its option order and probabilities;
  - `raw_probabilities`: the identity order, before debiasing.

  A debiased run therefore also holds the undebiased emulator.
- **Manifests.** Run manifests record the debiaser id.
- **Priors.** A prior applies per *prior key*, the question signature plus the fixed
  positions. `LabelPriors` JSON (`scripts/calibrate.py priors`) turns PriDe and batch into
  zero-cost debiasers.
- **Offline variants** (in `crossfit --debias`):
  - `none`: the raw probabilities.
  - `batch`: needs any run with diagnostics.
  - `pride`: needs a run that recorded permuted presentations, such as an online `pride`
    run. Its priors are fitted out of fold.

  `contextual` and `permutation` need their own backend passes.
- **Tests.** The FakeBackend's model multiplies each option's content weight by a position
  bias (×12 on A):
  - PriDe and contextual recover the true distribution exactly.
  - Permutation restores the content ranking and is invariant to rotating the input order.
  - Batch priors fix every biased answer of a balanced batch.
  - The offline prior division that `registry --priors` fits on (`apply_label_priors`)
    reproduces what `BatchDebiaser` and `PriDeDebiaser(priors=...)` answer, "I don't know"
    fixed.

## Further emulator runs

Done (multi-call history, before one-call scoring): the required `--debiaser pride:alpha=0.1`
holdout runs of the Stage 2 winner and of the tied cyankiwi build (above).

Optional, for completeness only, since PriDe had no measurable effect:

1. **`--debiaser contextual`** adds one scoring pass per MCQ and BoolQ item, one per
   benchmark elsewhere, and gives the `contextual` arm.
2. **`--debiaser permutation`**, on the five MCQ benchmarks, costs ×*F* (×10 on MMLU-Pro). It
   is the full permutation arm, and PriDe's upper bound.

Add each run to the same `crossfit` command as another `NAME=RUN_DIR`.

## GPT-6 Luna on holdout

*History: this study used the multi-call QuantTrio run and Luna's earlier free first-token read
(`gpt-6-luna.first_token.state_first`, now in `runs/archive_noprefill/select/`). Luna now
answers through Structured Outputs; against the one-call QuantTrio run, Luna − QuantTrio is
-0.0117 [-0.0311, +0.0079] in accuracy over the 6 benchmarks
([Current results](#current-results-one-call-per-question-holdout)). Its current holdout
numbers are in the [summary report](../../reports/summary/index.html).*

Sep 25, 2026. `gpt-6-luna` over the OpenAI API, scored from logprobs like the emulator
([openai_probe_report.md](openai_probe_report.md), [selection.md](selection.md#gpt-6-luna-openai-api-logprob-scored)),
with no debiaser.

OpenAI returns at most 5 top logprobs, so **MMLU-Pro, banking77 and CLINC150 cannot be run on
OpenAI models**. **Yelp is dropped** for every system. This study therefore covers the other
**6 benchmarks, 7,534 holdout items** answered by all three systems. That is why its Jev and
emulator numbers differ from the 9-benchmark grid above.

```bash
uv run python scripts/run_split.py run --system openai --strategy first_token \
    --prompt-cache-mode explicit --system-id gpt-6-luna.first_token.state_first \
    --benchmarks all --split holdout --out runs/select --concurrency 16 --max-usd 2
# As run (multi-call history, before one-call scoring); current runs are <preset>.auto_single.state_first
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride \
    luna=runs/select/gpt-6-luna.first_token.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_luna_noyelp
```

The holdout run cost $0.160: 10,037 requests including Yelp's 2,500, with 3 empty replies
resent.

### 2x2 grid (macro over 6 benchmarks, calibrated = `temperature@signature`)

`Emulator` is QuantTrio's undebiased probabilities (`quanttrio+none`). Accuracy is unchanged by
calibration.

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 0.8158 [0.7995, 0.8321] | 0.648 [0.604, 0.695] | 0.512 [0.479, 0.547] | -0.136 [-0.159, -0.115] | 0.275 [0.259, 0.291] | 0.264 [0.248, 0.280] | -0.011 [-0.015, -0.008] | 0.082 | 0.048 |
| Emulator (QuantTrio) | 0.7638 [0.7448, 0.7829] | 0.653 [0.617, 0.689] | 0.589 [0.555, 0.622] | -0.064 [-0.074, -0.055] | 0.324 [0.306, 0.343] | 0.314 [0.296, 0.332] | -0.010 [-0.013, -0.008] | 0.079 | 0.044 |
| **GPT-6 Luna** | 0.7546 [0.7358, 0.7741] | 1.008 [0.939, 1.081] | 0.681 [0.637, 0.728] | -0.327 [-0.361, -0.294] | 0.383 [0.358, 0.409] | 0.344 [0.324, 0.365] | -0.039 [-0.046, -0.032] | 0.139 | 0.054 |
| **Luna − Jev** | -0.0612 [-0.0810, -0.0420] | +0.361 [+0.294, +0.427] | +0.170 [+0.124, +0.215] | | +0.108 [+0.086, +0.131] | +0.081 [+0.062, +0.101] | | +0.057 | +0.006 |
| *Luna − Jev, with Yelp (7, superseded)* | -0.0640 [-0.0812, -0.0474] | +0.384 [+0.326, +0.443] | +0.193 [+0.154, +0.232] | | +0.119 [+0.100, +0.139] | +0.086 [+0.069, +0.103] | | +0.068 | +0.003 |

Luna − emulator, point differences only (the study pairs each system with Jev, not with each
other): accuracy -0.009, NLL +0.355 raw and +0.092 calibrated, Brier +0.059 raw and +0.030
calibrated, ECE +0.060 raw and +0.010 calibrated.

Every calibrator, Luna (macro over the 6):

| Arm | Accuracy | NLL | Brier | ECE | Δ NLL vs raw |
| --- | --- | --- | --- | --- | --- |
| `raw` | 0.7546 [0.7358, 0.7741] | 1.008 [0.939, 1.081] | 0.383 [0.358, 0.409] | 0.139 |  |
| `identity@global` (ε floor only) | 0.7546 [0.7358, 0.7741] | 0.949 [0.883, 1.020] | 0.383 [0.358, 0.409] | 0.139 | -0.059 [-0.068, -0.051] |
| `temperature@global` | 0.7546 [0.7358, 0.7741] | 0.699 [0.669, 0.728] | 0.350 [0.335, 0.366] | 0.083 | -0.310 [-0.356, -0.266] |
| `temperature@benchmark` | 0.7546 [0.7358, 0.7741] | 0.666 [0.631, 0.700] | 0.339 [0.322, 0.356] | 0.048 | -0.343 [-0.387, -0.301] |
| `temperature@signature` | 0.7546 [0.7358, 0.7741] | 0.681 [0.637, 0.728] | 0.344 [0.324, 0.365] | 0.054 | -0.327 [-0.361, -0.294] |
| `vector@signature` | 0.7674 [0.7480, 0.7866] | 0.630 [0.589, 0.674] | 0.326 [0.305, 0.346] | 0.053 | -0.378 [-0.419, -0.339] |
| `platt@signature` | 0.7605 [0.7414, 0.7796] | 0.717 [0.666, 0.771] | 0.346 [0.325, 0.367] | 0.054 | -0.292 [-0.321, -0.262] |
| `isotonic@signature` | 0.7577 [0.7384, 0.7771] | 0.745 [0.682, 0.818] | 0.349 [0.328, 0.371] | 0.062 | -0.263 [-0.300, -0.220] |
| `histogram@signature` | 0.7517 [0.7328, 0.7709] | 0.733 [0.678, 0.791] | 0.350 [0.329, 0.372] | 0.045 | -0.276 [-0.306, -0.246] |

Luna per benchmark, raw → `temperature@signature`:

| Benchmark | n | Accuracy | NLL raw | NLL cal. | Δ NLL | Brier raw | Brier cal. | ECE raw | ECE cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.5354 [0.4343, 0.6364] | 1.554 | 1.323 | -0.231 [-0.345, -0.124] | 0.696 | 0.649 | 0.214 | 0.149 |
| lexam_en_idk | 309 | 0.7120 [0.6602, 0.7605] | 0.755 | 0.763 | +0.008 [-0.026, +0.040] | 0.392 | 0.393 | 0.084 | 0.051 |
| arc_challenge | 586 | 0.9608 [0.9454, 0.9761] | 0.158 | 0.118 | -0.041 [-0.084, -0.007] | 0.059 | 0.054 | 0.028 | 0.023 |
| ag_news | 3800 | 0.8774 [0.8666, 0.8876] | 0.952 | 0.435 | -0.517 [-0.585, -0.451] | 0.226 | 0.201 | 0.105 | 0.023 |
| boolq | 1635 | 0.8875 [0.8722, 0.9021] | 0.671 | 0.295 | -0.376 [-0.465, -0.295] | 0.197 | 0.170 | 0.091 | 0.019 |
| sst5 | 1105 | 0.5548 [0.5249, 0.5837] | 1.960 | 1.154 | -0.807 [-0.929, -0.685] | 0.731 | 0.598 | 0.310 | 0.059 |
| **macro** | 7534 | 0.7546 [0.7358, 0.7741] | 1.008 | 0.681 | -0.327 [-0.361, -0.294] | 0.383 | 0.344 | 0.139 | 0.054 |

### Findings

- **Raw Luna is the worst calibrated of the three:** ECE 0.139 against 0.082 for Jev and
  0.079 for the emulator; NLL 1.008. All three are overconfident, Luna most: its fitted global
  temperature is 2.57, against 1.87 for Jev and 1.81 for the emulator.
- **Calibration recovers most of it.** `temperature@signature` cuts Luna's NLL by 0.327
  (Jev 0.136, emulator 0.064). Its ECE drops from 0.139 to 0.054, close to Jev (0.048) and the
  emulator (0.044). The NLL gap to Jev halves (+0.361 → +0.170) but remains, because Luna is
  0.061 less accurate. Calibrated, Luna is still 0.092 NLL behind the emulator.
- **Best arm:** `vector@signature` (accuracy 0.7674, NLL 0.630). Same caveat as above: it
  learns each benchmark's answer mix.
- **Batch debiasing does not pay**, as for the emulator. Measured with Yelp (7 benchmarks),
  `luna+batch` − `luna` is accuracy +0.0010 [-0.0051, +0.0069] and NLL -0.005 after
  calibration.
- **Luna does not beat the local emulator here.** It is 0.009 less accurate on holdout
  (-0.0009 [-0.0231, +0.0210] on `select`) and worse on every probability metric, raw and
  calibrated. It costs about 4x as much per item as the emulator at the 410 W electricity
  estimate (6x at the old 280 W estimate;
  [selection.md](selection.md#gpt-6-luna-openai-api-logprob-scored)).

## The fast MoE emulator on holdout

Sep 26, 2026. The fast emulator is preset `qwen3.6-35b-a3b-fast`
(`QuantTrio/Qwen3.6-35B-A3B-AWQ`, a mixture-of-experts (MoE) model, tuned in
[selection.md](selection.md#fast-moe-tuning-qwen36-35b-a3b)). With no debiaser and 16 items in
flight, it answered all 19,840 holdout items in one runner invocation (Yelp included: the run
predates its removal). Throughput was 11.02 q/s (questions per second) over the 9 benchmarks
(11.88 with Yelp); latency p50 552 ms, p95 7,095 ms. The study compares it with Jev and the
dense emulator (QuantTrio 27B, its undebiased probabilities `quanttrio+none`, as above). This
study is multi-call history (before one-call scoring); the runs are redone with `auto_single`.

```bash
# As run (multi-call history, before one-call scoring); current runs are <preset>.auto_single.state_first
uv run python scripts/sweep_presets.py run --split holdout --out runs/select --strategy auto_noecho \
    --renderer state_first --concurrency 16 --presets qwen3.6-35b-a3b-fast
# 9 benchmarks (the loaders skip Yelp); make_report.py reads this study
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_fast
# with Yelp, named explicitly: --benchmarks <the 9>,yelp_stars ... --out runs/calibration/holdout_fast_10
uv run python scripts/calibrate.py registry runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    --split holdout --calibrator temperature --priors none \
    --out calibration/qwen3.6-35b-a3b-fast/registry.json
```

Each cross-fit took about 380 s. The "Fast − dense" rows pair the two emulators on the study's
per-item predictions (stratified bootstrap, 10,000 resamples, seed 0; ECE differences are point
differences). Every other row comes straight from the study.

### 2x2 grid (macro over 9 benchmarks, 17,340 paired items; calibrated = `temperature@signature`)

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated | Δ ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 0.8253 [0.8138, 0.8365] | 0.703 [0.666, 0.741] | 0.555 [0.530, 0.581] | -0.148 [-0.166, -0.131] | 0.263 [0.252, 0.275] | 0.255 [0.244, 0.266] | -0.008 [-0.010, -0.006] | 0.072 | 0.048 | -0.024 |
| Dense emulator (QuantTrio 27B) | 0.7673 [0.7542, 0.7805] | 0.723 [0.695, 0.752] | 0.666 [0.642, 0.691] | -0.057 [-0.064, -0.049] | 0.324 [0.311, 0.337] | 0.315 [0.303, 0.328] | -0.009 [-0.010, -0.007] | 0.071 | 0.039 | -0.032 |
| **Fast emulator (QuantTrio 35B-A3B)** | 0.7319 [0.7184, 0.7457] | 0.809 [0.780, 0.838] | 0.761 [0.736, 0.786] | -0.049 [-0.055, -0.043] | 0.371 [0.357, 0.384] | 0.358 [0.346, 0.371] | -0.012 [-0.014, -0.010] | 0.079 | 0.037 | -0.043 |
| **Fast − Jev** | -0.0934 [-0.1066, -0.0803] | +0.106 [+0.077, +0.135] | +0.206 [+0.185, +0.227] | | +0.108 [+0.097, +0.118] | +0.103 [+0.093, +0.114] | | +0.007 | -0.011 | |
| **Fast − dense** | -0.0353 [-0.0488, -0.0219] | +0.086 [+0.066, +0.107] | +0.094 [+0.076, +0.112] | | +0.047 [+0.037, +0.057] | +0.043 [+0.034, +0.052] | | +0.008 | -0.003 | |
| *Jev, with Yelp (10)* | 0.8097 [0.7992, 0.8200] | 0.727 [0.693, 0.762] | 0.575 [0.552, 0.598] | -0.152 [-0.169, -0.136] | 0.284 [0.273, 0.295] | 0.273 [0.263, 0.283] | -0.011 [-0.013, -0.009] | 0.078 | 0.047 | -0.031 |
| *Dense, with Yelp (10)* | 0.7512 [0.7393, 0.7632] | 0.767 [0.741, 0.793] | 0.692 [0.670, 0.715] | -0.075 [-0.082, -0.068] | 0.352 [0.339, 0.363] | 0.335 [0.324, 0.346] | -0.016 [-0.018, -0.015] | 0.086 | 0.043 | -0.043 |
| *Fast, with Yelp (10)* | 0.7180 [0.7056, 0.7304] | 0.865 [0.838, 0.892] | 0.787 [0.764, 0.810] | -0.079 [-0.086, -0.072] | 0.396 [0.383, 0.408] | 0.377 [0.366, 0.388] | -0.019 [-0.021, -0.017] | 0.092 | 0.040 | -0.052 |
| *Fast − Jev, with Yelp* | -0.0918 [-0.1036, -0.0798] | +0.138 [+0.111, +0.165] | +0.212 [+0.193, +0.231] | | +0.112 [+0.102, +0.122] | +0.104 [+0.095, +0.114] | | +0.014 | -0.007 | |
| *Fast − dense, with Yelp* | -0.0332 [-0.0453, -0.0211] | +0.098 [+0.079, +0.117] | +0.094 [+0.078, +0.111] | | +0.044 [+0.035, +0.054] | +0.042 [+0.034, +0.051] | | +0.006 | -0.003 | |

Italic rows come from `runs/calibration/holdout_fast_10` (19,840 items), for comparison with the
10-benchmark tables above. Like them, they are superseded.

Every calibrator, the fast emulator (macro, 9 benchmarks):

| Arm | Accuracy | NLL | Brier | ECE | Δ NLL vs raw |
| --- | --- | --- | --- | --- | --- |
| `raw` | 0.7319 [0.7184, 0.7457] | 0.809 [0.780, 0.838] | 0.371 [0.357, 0.384] | 0.079 |  |
| `identity@global` (ε floor only) | 0.7319 [0.7184, 0.7457] | 0.809 [0.780, 0.837] | 0.371 [0.357, 0.384] | 0.079 | -0.001 [-0.002, +0.001] |
| `temperature@global` | 0.7319 [0.7184, 0.7457] | 0.785 [0.762, 0.809] | 0.365 [0.353, 0.376] | 0.064 | -0.024 [-0.030, -0.018] |
| `temperature@benchmark` | 0.7319 [0.7184, 0.7457] | 0.760 [0.735, 0.786] | 0.359 [0.346, 0.371] | 0.036 | -0.049 [-0.055, -0.043] |
| `temperature@signature` | 0.7319 [0.7184, 0.7457] | 0.761 [0.736, 0.786] | 0.358 [0.346, 0.371] | 0.037 | -0.049 [-0.055, -0.043] |
| `vector@signature` | 0.7523 [0.7389, 0.7659] | 0.690 [0.666, 0.713] | 0.334 [0.321, 0.346] | 0.038 | -0.119 [-0.129, -0.110] |
| `platt@signature` | 0.7314 [0.7180, 0.7450] | 0.758 [0.733, 0.783] | 0.355 [0.343, 0.366] | 0.037 | -0.051 [-0.058, -0.044] |
| `isotonic@signature` | 0.7327 [0.7189, 0.7463] | 0.779 [0.750, 0.808] | 0.360 [0.348, 0.372] | 0.043 | -0.031 [-0.041, -0.018] |
| `histogram@signature` | 0.7312 [0.7175, 0.7449] | 0.784 [0.754, 0.814] | 0.362 [0.349, 0.374] | 0.040 | -0.026 [-0.037, -0.011] |

The fast emulator per benchmark, raw → `temperature@signature`:

| Benchmark | n | Accuracy | NLL raw | NLL cal. | Δ NLL | Brier raw | Brier cal. | ECE raw | ECE cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.5354 [0.4343, 0.6364] | 1.109 | 1.108 | -0.000 [-0.012, +0.010] | 0.575 | 0.574 | 0.100 | 0.106 |
| lexam_en_idk | 309 | 0.5987 [0.5437, 0.6537] | 1.006 | 1.008 | +0.002 [-0.005, +0.009] | 0.519 | 0.519 | 0.065 | 0.046 |
| mmlu_pro | 6016 | 0.5946 [0.5819, 0.6069] | 1.267 | 1.221 | -0.046 [-0.055, -0.038] | 0.530 | 0.518 | 0.090 | 0.016 |
| arc_challenge | 586 | 0.9556 [0.9386, 0.9727] | 0.127 | 0.134 | +0.007 [+0.001, +0.012] | 0.066 | 0.066 | 0.015 | 0.016 |
| ag_news | 3800 | 0.8732 [0.8626, 0.8834] | 0.578 | 0.419 | -0.159 [-0.185, -0.133] | 0.217 | 0.198 | 0.088 | 0.021 |
| banking77 | 1540 | 0.7188 [0.6961, 0.7416] | 1.180 | 1.100 | -0.080 [-0.106, -0.055] | 0.420 | 0.404 | 0.107 | 0.048 |
| clinc150 | 2250 | 0.8658 [0.8516, 0.8796] | 0.497 | 0.505 | +0.007 [+0.004, +0.010] | 0.199 | 0.198 | 0.019 | 0.014 |
| boolq | 1635 | 0.8832 [0.8673, 0.8985] | 0.314 | 0.302 | -0.012 [-0.021, -0.002] | 0.178 | 0.179 | 0.023 | 0.020 |
| sst5 | 1105 | 0.5620 [0.5330, 0.5910] | 1.206 | 1.048 | -0.158 [-0.196, -0.121] | 0.632 | 0.571 | 0.205 | 0.041 |
| **macro** | 17340 | 0.7319 [0.7184, 0.7457] | 0.809 | 0.761 | -0.049 [-0.055, -0.043] | 0.371 | 0.358 | 0.079 | 0.037 |

**Deployment registry:** `calibration/qwen3.6-35b-a3b-fast/registry.json`, now refitted on the
one-call `holdout` run's raw probabilities over the 9 benchmarks (17,340 items):

- 12 signatures with at least 30 items, with T from 1.024 (CLINC150's 150 options) to 1.948
  (AG News);
- T = 1.292 for every other signature of this model and template.

Held-out estimate on the benchmarks (one-call run): ECE 0.037, NLL 0.835, Brier 0.363
([Current results](#current-results-one-call-per-question-holdout)). As first fitted on the
multi-call run: T 0.96-1.95, fallback 1.277, the `temperature@signature` row above.

Smoke test against the live fast server (throwaway script, the multi-call registry): the
registry was loaded into an `Emulator` (`NoEchoAutoStrategy`, no debiaser). It answered all
21 request fixtures of
`tests/golden/fixtures/jev_docs` (41 questions: 14 choice, 14 noul, 13 score). Every response
validated as a `SystemOneResponse` and round-tripped through JSON. Every answer carried its
calibrator: noul by its own signature, choice and score by the model-and-template fallback.

### Findings

- **The fast emulator trades 0.035 accuracy for 3.8x the dense emulator's throughput.**
  - Throughput: 11.07 against 2.94 q/s on `select`, 9 benchmarks (in this multi-call study the
    dense holdout run's online PriDe added calls).
  - Accuracy: fast − dense is -0.0353 [-0.0488, -0.0219] on holdout. That is larger than on
    `select` (-0.0200 [-0.0333, -0.0063]), where the dense run scored 0.015 lower and the fast
    one the same. The fast emulator is 0.093 behind Jev.
- **Its raw probabilities are about as well calibrated as the dense emulator's:** ECE 0.079
  against 0.071; fitted global temperature 1.28 against 1.27 (Jev: 1.26).
  `temperature@signature` takes its ECE to 0.037, level with the dense emulator (0.039) and
  Jev (0.048). The remaining probability gap comes from accuracy: calibrated NLL +0.094 and
  Brier +0.043 against the dense emulator, +0.206 and +0.103 against Jev.
- **Best arm:** `vector@signature` (accuracy 0.7523, NLL 0.690). Same caveat as above: it
  learns each benchmark's answer mix.

### The GPTQ build on holdout

Sep 26, 2026. `qwen3.6-35b-a3b-int4-palmfuture` (`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, the
same MoE model, GPTQ 4-bit) answered all 17,340 holdout items of the 9 benchmarks with 0 errors
and no debiaser. Why it was run and its speed:
[selection.md](selection.md#the-gptq-build-as-a-finalist-qwen36-35b-a3b-int4-palmfuture). This
study is multi-call history (before one-call scoring); the runs are redone with `auto_single`.

```bash
# As run (multi-call history, before one-call scoring); current runs are <preset>.auto_single.state_first
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_noecho.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_gptq
uv run python scripts/calibrate.py registry \
    runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_noecho.state_first \
    --split holdout --calibrator temperature --priors none \
    --out calibration/qwen3.6-35b-a3b-int4-palmfuture/registry.json
```

The study took 327 s. Its Jev, dense and fast rows are identical to `holdout_fast`. The
"GPTQ − fast" and "GPTQ − dense" rows pair the emulators on the study's per-item predictions,
like "Fast − dense" above (a throwaway script with `make_report.py`'s bootstrap; the same script
reproduced the "Fast − dense" accuracy within ±0.0001). The other rows come straight from the
study.

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated | Δ ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast emulator (AWQ) | 0.7319 [0.7184, 0.7457] | 0.809 [0.780, 0.838] | 0.761 [0.736, 0.786] | -0.049 [-0.055, -0.043] | 0.371 [0.357, 0.384] | 0.358 [0.346, 0.371] | -0.012 [-0.014, -0.010] | 0.079 | 0.037 | -0.043 |
| **GPTQ build** | 0.7302 [0.7169, 0.7438] | 0.826 [0.796, 0.856] | 0.767 [0.743, 0.792] | -0.059 [-0.066, -0.052] | 0.378 [0.364, 0.392] | 0.363 [0.350, 0.375] | -0.016 [-0.018, -0.013] | 0.091 | 0.043 | -0.048 |
| **GPTQ − fast** | -0.0017 [-0.0107, +0.0074] | +0.017 [+0.008, +0.025] | +0.007 [-0.001, +0.014] | | +0.007 [+0.003, +0.012] | +0.004 [+0.000, +0.008] | | +0.012 | +0.006 | |
| **GPTQ − Jev** | -0.0951 [-0.1080, -0.0822] | +0.123 [+0.094, +0.152] | +0.213 [+0.192, +0.233] | | +0.115 [+0.104, +0.126] | +0.107 [+0.097, +0.118] | | +0.019 | -0.005 | |
| **GPTQ − dense** | -0.0370 [-0.0494, -0.0245] | +0.103 [+0.083, +0.124] | +0.101 [+0.084, +0.118] | | +0.054 [+0.045, +0.064] | +0.047 [+0.039, +0.056] | | +0.020 | +0.003 | |

Other arms, GPTQ build: `temperature@global` NLL 0.795, ECE 0.070 (T 1.302);
`temperature@benchmark` 0.767 / 0.041; `vector@signature` is again the best arm (accuracy
0.7508, NLL 0.696, Brier 0.335, ECE 0.038). Per benchmark, calibration helps most where the raw
ECE is highest: SST-5 0.225 → 0.061, banking77 0.111 → 0.042, MMLU-Pro 0.095 → 0.017, AG News
0.090 → 0.019.

**Deployment registry:**
[`calibration/qwen3.6-35b-a3b-int4-palmfuture/registry.json`](../../calibration/qwen3.6-35b-a3b-int4-palmfuture/registry.json).
It holds `temperature` per signature, refitted on the one-call `holdout` run's raw
probabilities (9 benchmarks, 17,340 items):

- 12 signatures with at least 30 items, with T from 1.056 (CLINC150) to 1.991 (AG News);
- T = 1.325 for every other signature of this model and template.

As first fitted on the multi-call run: T 0.99-1.99, fallback 1.302.

`CalibratorRegistry.load` reads it. It was not smoke-tested against a live server.

**Findings.**

- **Same accuracy as the AWQ fast emulator** on holdout (−0.0017, tied). On `select` it was
  0.011 lower, and that interval excludes 0
  ([selection.md](selection.md#the-gptq-build-as-a-finalist-qwen36-35b-a3b-int4-palmfuture)).
- **Slightly worse raw probabilities:** ECE 0.091 against 0.079, NLL +0.017. Calibration mostly
  closes the gap: calibrated NLL +0.007 [-0.001, +0.014] (tied), Brier +0.004 [+0.000, +0.008].
- **Against Jev** it is 0.095 behind, like the AWQ build (0.093).

### The Gemma 4 26B-A4B build on holdout

Sep 26, 2026. `gemma-4-26b-a4b-int4-cyankiwi` (`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`, an
INT4 build of Google's QAT Gemma 4 26B-A4B, a MoE with 3.8B active parameters) answered all
17,340 holdout items of the 9 benchmarks with 0 errors and no debiaser. Why it was run and its
speed: [selection.md](selection.md#gemma-4-26b-a4b-as-a-finalist). This study is multi-call
history (before one-call scoring) and predates the Gemma 4 prompt fix; the runs were redone.

```bash
# As run (multi-call history, before one-call scoring); current runs are <preset>.auto_single.state_first
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    gptq=runs/select/qwen3.6-35b-a3b-int4-palmfuture.auto_noecho.state_first \
    gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_noecho.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_gemma
uv run python scripts/calibrate.py crossfit \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    gemma=runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_noecho.state_first \
    --split holdout --reference fast --debias none --out runs/calibration/holdout_gemma_vs_fast
mkdir -p calibration/gemma-4-26b-a4b-int4-cyankiwi   # calibrate.py does not create it
uv run python scripts/calibrate.py registry \
    runs/select/gemma-4-26b-a4b-int4-cyankiwi.auto_noecho.state_first \
    --split holdout --calibrator temperature --priors none \
    --out calibration/gemma-4-26b-a4b-int4-cyankiwi/registry.json
```

The first study took 362 s. Its other rows are identical to `holdout_gptq`. The second study
pairs Gemma with the fast emulator directly (same folds and seed; Gemma's own row is the same in
both). Every row below comes straight from one of the two studies.

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated | Δ ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast emulator (AWQ) | 0.7319 [0.7184, 0.7457] | 0.809 [0.780, 0.838] | 0.761 [0.736, 0.786] | -0.049 [-0.055, -0.043] | 0.371 [0.357, 0.384] | 0.358 [0.346, 0.371] | -0.012 [-0.014, -0.010] | 0.079 | 0.037 | -0.043 |
| **Gemma 4 26B-A4B** | 0.7141 [0.7001, 0.7280] | 1.322 [1.254, 1.393] | 0.854 [0.829, 0.880] | -0.468 [-0.515, -0.424] | 0.459 [0.437, 0.481] | 0.383 [0.370, 0.396] | -0.076 [-0.086, -0.065] | 0.195 | 0.061 | -0.134 |
| **Gemma − fast** | -0.0178 [-0.0336, -0.0022] | +0.513 [+0.454, +0.575] | +0.094 [+0.070, +0.118] | | +0.088 [+0.069, +0.108] | +0.025 [+0.013, +0.037] | | +0.116 | +0.024 | |
| **Gemma − Jev** | -0.1113 [-0.1275, -0.0956] | +0.619 [+0.559, +0.683] | +0.300 [+0.274, +0.325] | | +0.196 [+0.177, +0.216] | +0.128 [+0.116, +0.141] | | +0.123 | +0.013 | |

Other arms, Gemma: `temperature@global` NLL 0.970, ECE 0.156 (T 1.718); `temperature@benchmark`
0.853 / 0.061; `vector@signature` is the best arm on every metric (accuracy 0.7402, NLL 0.775,
Brier 0.353, ECE 0.046). Vector scaling changes which answer is on top, so its accuracy gain
partly learns each benchmark's answer mix (same caveat as above).

Fitted temperatures (`temperature@benchmark`, mean over folds) are higher than the Qwen builds'
(about 1.0-2.0) on every benchmark: GPQA 3.29, SST-5 3.17, LEXam 3.16, AG News 2.77, MMLU-Pro
2.24, BoolQ 2.04, banking77 1.54, ARC 1.49, CLINC150 1.17. Per benchmark, calibration helps
most where the raw ECE is highest: GPQA 0.367 → 0.112, LEXam 0.337 → 0.104, SST-5 0.308 → 0.061,
MMLU-Pro 0.241 → 0.039, banking77 0.188 → 0.068.

**Deployment registry:**
[`calibration/gemma-4-26b-a4b-int4-cyankiwi/registry.json`](../../calibration/gemma-4-26b-a4b-int4-cyankiwi/registry.json).
It holds `temperature` per signature, refitted on the one-call `holdout` run after the prompt
fix (raw probabilities, 9 benchmarks, 17,340 items):

- 12 signatures with at least 30 items, with T from 1.197 (CLINC150) to 3.568 (SST-5);
- T = 1.681 for every other signature of this model and template (`state_first-5d29289f8b87`).

As first fitted on the multi-call run before the prompt fix: T 1.17-3.17, fallback 1.718.

It was not smoke-tested against a live server.

**Findings.**

- **Less accurate than the Qwen MoE builds:** -0.018 against the fast emulator, and the interval
  excludes 0. Against Jev it is 0.111 behind (the Qwen MoEs: 0.093-0.095).
- **Strongly overconfident before calibration.** Raw ECE is 0.195, against 0.079-0.091 for the
  Qwen MoEs and 0.072 for Jev, about as high as DeepSeek V4.1 Flash's free read (0.191 on its 7
  benchmarks, below). Gemma 4 12B showed the same on the screen (NLL 1.18). The cause is not
  known. These runs predate the Gemma 4 prompt fix: before it, the prompt lacked the empty
  thought channel, and the top-k could keep the wrong one of two same-text tokens. In a
  50-item-per-benchmark A/B the fix moved NLL both ways (26B-A4B 1.548 → 1.447, 12B 1.289 →
  1.439), so it does not explain the overconfidence.
- **Calibration removes most of it but not all.** Temperature per signature cuts NLL from 1.322
  to 0.854 and ECE to 0.061. Calibrated, Gemma is still behind the fast emulator (NLL +0.094,
  Brier +0.025, ECE +0.024).

## DeepSeek V4.1 Flash on holdout

*History: this study used the multi-call emulator runs and DeepSeek's earlier free first-token
read on Wafer (`deepseek-v4.1-flash@wafer.first_token.state_first`, now in
`runs/archive_noprefill/select/`). DeepSeek now answers through Structured Outputs on Makora;
against the one-call QuantTrio run, DeepSeek − QuantTrio is +0.0052 [-0.0118, +0.0223] in
accuracy over the 7 benchmarks
([Current results](#current-results-one-call-per-question-holdout)). Its current holdout
numbers are in the [summary report](../../reports/summary/index.html).*

Sep 26, 2026. `deepseek/deepseek-v4.1-flash` over OpenRouter, pinned to the `wafer` provider,
scored from logprobs like Luna ([openrouter_probe_report.md](openrouter_probe_report.md),
[selection.md](selection.md#deepseek-v41-flash-openrouter-logprob-scored)), with no debiaser.

OpenRouter returns up to 20 top logprobs, so MMLU-Pro runs, but **banking77 and CLINC150
cannot**. Yelp is dropped. Two studies:

- **Main study:** the other **7 benchmarks, 13,550 holdout items**, answered by Jev, both
  emulators and DeepSeek.
- **Second study:** adds Luna, and so drops MMLU-Pro (6 benchmarks, 7,534 items, the same items
  as the Luna study above).

```bash
uv run python scripts/run_split.py run --system openrouter --provider wafer --strategy first_token \
    --system-id deepseek-v4.1-flash@wafer.first_token.state_first \
    --benchmarks all --split holdout --out runs/select --concurrency 32 --max-usd 3
# As run (multi-call history, before one-call scoring); current runs are <preset>.auto_single.state_first
uv run python scripts/calibrate.py crossfit jev=runs/select/jev \
    quanttrio=runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first.pride \
    fast=runs/select/qwen3.6-35b-a3b-fast.auto_noecho.state_first \
    deepseek=runs/select/deepseek-v4.1-flash@wafer.first_token.state_first \
    --split holdout --reference jev --debias none --out runs/calibration/holdout_deepseek
# + luna=runs/select/gpt-6-luna.first_token.state_first --out runs/calibration/holdout_deepseek_luna
```

The holdout run cost $0.279 billed (16,050 requests including Yelp's 2,500; $0.221 without
them). Every response came from Wafer, with no empty replies. Each cross-fit took about 300 s.

### 2x2 grid (macro over 7 benchmarks, calibrated = `temperature@signature`)

`Emulator` is QuantTrio's undebiased probabilities (`quanttrio+none`). Accuracy is unchanged by
calibration.

| System | Accuracy | NLL raw | NLL calibrated | Δ NLL | Brier raw | Brier calibrated | Δ Brier | ECE raw | ECE calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 0.8175 [0.8030, 0.8314] | 0.662 [0.621, 0.703] | 0.533 [0.505, 0.562] | -0.129 [-0.149, -0.109] | 0.274 [0.260, 0.288] | 0.265 [0.252, 0.279] | -0.008 [-0.011, -0.006] | 0.077 | 0.054 |
| Emulator (QuantTrio) | 0.7447 [0.7280, 0.7613] | 0.723 [0.691, 0.755] | 0.667 [0.638, 0.695] | -0.057 [-0.064, -0.049] | 0.347 [0.331, 0.363] | 0.338 [0.323, 0.353] | -0.009 [-0.011, -0.007] | 0.075 | 0.042 |
| Fast MoE emulator | 0.7147 [0.6976, 0.7318] | 0.801 [0.768, 0.834] | 0.749 [0.720, 0.778] | -0.052 [-0.059, -0.045] | 0.388 [0.372, 0.405] | 0.375 [0.360, 0.390] | -0.013 [-0.016, -0.011] | 0.084 | 0.038 |
| **DeepSeek V4.1 Flash** | 0.7499 [0.7329, 0.7668] | 1.301 [1.216, 1.384] | 0.701 [0.671, 0.730] | -0.599 [-0.655, -0.543] | 0.428 [0.400, 0.456] | 0.345 [0.328, 0.361] | -0.083 [-0.096, -0.070] | 0.191 | 0.070 |
| **DeepSeek − Jev** | -0.0675 [-0.0860, -0.0497] | +0.639 [+0.561, +0.716] | +0.168 [+0.136, +0.200] | | +0.154 [+0.128, +0.180] | +0.080 [+0.064, +0.096] | | +0.113 | +0.016 |

DeepSeek − emulator, point differences only (the study pairs each system with Jev, not with
each other): accuracy +0.005, NLL +0.578 raw and +0.034 calibrated, Brier +0.081 raw and +0.007
calibrated, ECE +0.116 raw and +0.028 calibrated. Against the fast MoE: accuracy +0.035,
calibrated NLL -0.048.

On Luna's 6 benchmarks (second study):

- Accuracy: DeepSeek 0.7583 [0.7389, 0.7775], Luna 0.7546, Jev 0.8158, emulator 0.7638.
- DeepSeek − Jev: accuracy -0.0575 [-0.0789, -0.0368], calibrated NLL +0.135 [+0.097, +0.172]
  (Luna − Jev: -0.0612, +0.170).
- DeepSeek − Luna, point differences: accuracy +0.004, NLL +0.226 raw and -0.035 calibrated,
  Brier +0.034 raw and -0.011 calibrated, ECE +0.048 raw and +0.011 calibrated.

DeepSeek per benchmark, raw → `temperature@signature`:

| Benchmark | n | Accuracy | NLL raw | NLL cal. | Δ NLL | Brier raw | Brier cal. | ECE raw | ECE cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.5657 [0.4646, 0.6667] | 1.671 | 1.055 | -0.615 [-0.885, -0.353] | 0.724 | 0.559 | 0.314 | 0.148 |
| lexam_en_idk | 309 | 0.6440 [0.5890, 0.6958] | 1.797 | 0.972 | -0.825 [-1.048, -0.613] | 0.621 | 0.494 | 0.275 | 0.100 |
| mmlu_pro | 6016 | 0.6998 [0.6880, 0.7113] | 1.700 | 1.016 | -0.684 [-0.734, -0.635] | 0.491 | 0.416 | 0.212 | 0.079 |
| arc_challenge | 586 | 0.9676 [0.9522, 0.9812] | 0.163 | 0.131 | -0.032 [-0.088, +0.013] | 0.053 | 0.049 | 0.027 | 0.048 |
| ag_news | 3800 | 0.8947 [0.8847, 0.9045] | 0.982 | 0.396 | -0.586 [-0.660, -0.515] | 0.199 | 0.180 | 0.095 | 0.030 |
| boolq | 1635 | 0.9174 [0.9040, 0.9303] | 0.624 | 0.244 | -0.380 [-0.471, -0.292] | 0.152 | 0.133 | 0.072 | 0.031 |
| sst5 | 1105 | 0.5602 [0.5303, 0.5900] | 2.166 | 1.094 | -1.073 [-1.207, -0.940] | 0.756 | 0.584 | 0.340 | 0.052 |
| **macro** | 13550 | 0.7499 [0.7329, 0.7668] | 1.301 | 0.701 | -0.599 [-0.655, -0.543] | 0.428 | 0.345 | 0.191 | 0.070 |

### Findings (multi-call runs)

These findings compare DeepSeek's free-read run with the earlier multi-call emulator runs
(commands above). With the current runs (one-call QuantTrio, DeepSeek through Structured
Outputs), QuantTrio − DeepSeek on the 6 shared benchmarks is +0.0105 [−0.0093, +0.0299] in
accuracy: still a tie.

- **Raw DeepSeek (free read) was the worst calibrated system measured** (ECE 0.191, NLL 1.301).
  Its first-token logprobs are far too sharp. Its fitted global temperature is 2.49, against 1.45
  for Jev, 1.44 for the emulator and 1.51 for the fast MoE in this study. It is sharpest on the
  knowledge sets (GPQA, LEXam, MMLU-Pro: raw ECE 0.21–0.31) and SST-5 (0.34).
- **Calibration recovers most of it.** `temperature@signature` cuts NLL by 0.599 and ECE to
  0.070, still above Jev (0.054) and the emulators (0.042, 0.038). Calibrated, DeepSeek is
  0.034 NLL behind the dense emulator at slightly higher accuracy. It is ahead of Luna on
  their 6 benchmarks (-0.035). Its best arm, `vector@signature`, reaches accuracy 0.7557 and
  NLL 0.671 (same caveat as above: it learns each benchmark's answer mix).
- **Holdout confirms `select`.** DeepSeek ties the dense emulator on accuracy (+0.005 here,
  +0.002 on `select`) and is 0.068 behind Jev (0.068 on `select`). Because it can run
  MMLU-Pro, it is compared on 7 benchmarks where Luna manages 6, at a lower per-item price
  than Jev.
