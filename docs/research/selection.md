# Emulator configuration selection

Sep 24, 2026

This log records how the emulator configuration was chosen: model, quantization, scoring strategy
and prompt layout (renderer). Each candidate is a *preset*: one quantized model build with its vLLM
server settings. Jev answers the same items and is the reference for every comparison. Two hosted
models (GPT-6 Luna and DeepSeek V4.1 Flash) and CLM-8B, a dual encoder that answers Jev's wire
format, are compared at the end.

## Summary

- The chosen configuration is `qwen3.6-27b-int4-quanttrio` with the `state_first` prompt layout and
  one-call scoring (`auto_single`), on one RTX 3090. On the `select` half without Yelp it scores
  0.7520 macro accuracy against Jev's 0.8098 (Δ -0.0578 [-0.0734, -0.0422]). On `holdout` it is
  statistically tied with `qwen3.6-27b-int4-cyankiwi` (-0.0014 [-0.0101, +0.0069]); on `select`
  cyankiwi is 0.010 lower, and that interval only just excludes 0.
- The prompt layout moves accuracy more than the choice of quantized build. `state_first` added
  0.046 to 0.060 macro accuracy on every preset tested (Stage 2a), while the two Qwen3.6-27B builds
  differ by 0.010 on `select` and 0.001 on `holdout`.
- Most of the gap to Jev is on the knowledge benchmarks MMLU-Pro, GPQA and LEXam. On the other 6
  benchmarks the gap is at most 0.03 (`holdout`).
- The fast option, `qwen3.6-35b-a3b-fast`, gives up about 0.03 accuracy against the dense QuantTrio
  27B (-0.026 on `select`, -0.035 on `holdout`) for 4.5x its throughput on `select` (20.07 against
  4.44 q/s; 4.4x on `holdout`).
- The GPTQ build of the same MoE (`qwen3.6-35b-a3b-int4-palmfuture`) ties the AWQ fast preset on
  both halves (`holdout` -0.0039 [-0.0146, +0.0068], `select` -0.0041 [-0.0129, +0.0045]). Its raw
  probabilities are slightly worse (ECE 0.087 against 0.077); calibrated, the two are tied. It is
  about 1.8x faster on the 6 shared benchmarks, and 1.25x over all 9 (25.11 against 20.07 q/s on
  `select`).
- Gemma 4 26B-A4B (`gemma-4-26b-a4b-int4-cyankiwi`, a MoE with 3.8B active parameters) ties both
  Qwen MoE builds on `holdout` (+0.0045 [-0.0104, +0.0195] against AWQ fast). It is lower than AWQ
  fast on `select` (-0.0170 [-0.0319, -0.0023]) and on the screen, and about 0.09 below Jev. Over
  all 9 benchmarks it is about as fast as AWQ fast (20.47 against 20.07 q/s on `select`). Its raw
  probabilities are strongly overconfident (ECE 0.165 against 0.077); calibrated, its NLL ties AWQ
  fast's.
- Differences of about 1 point cannot be ranked. vLLM is not deterministic on this setup: sending
  byte-identical requests again flipped the top answer on 4% to 7% of hard multiple-choice questions
  (GPQA, LEXam, MMLU-Pro; about 0.3% on ARC), and the GPTQ build's GPQA accuracy ranged from 0.545
  to 0.576 across runs. The bootstrap intervals cover question sampling only and leave this noise
  out. So the local presets and hosted models in the 0.730 to 0.768 cluster (AWQ 0.734, GPTQ 0.730,
  Gemma 0.734 on the 6 shared `holdout` benchmarks) cannot be ranked against each other.
- GPQA's `holdout` half is easier than its `select` half for every system (+8 to +18 points; a
  permutation test gives p = 0.014). The hash split is the same for every system, so this is luck of
  the draw. It is why every system scores about 2 points higher on `holdout` than on `select` over
  the 6 shared benchmarks, and why Gemma's `select` to `holdout` jump is bigger.
- Raw NLL against Jev understates Jev's lead. Jev rounds its probabilities to 0.01, and its exact
  zeros are clipped to 1e-6 (13.8 nats each; 409 of 17,340 `holdout` items). So its raw NLL is
  inflated, and the "Δ NLL vs Jev" columns below understate its lead: on `holdout`, QuantTrio − Jev
  raw NLL is +0.047 as scored but about +0.15 with every system floored at 0.005
  ([jev_vs_qwen](../../reports/jev_vs_qwen/README.md)). Calibrated NLL is barely affected.
- The hosted models score like the local emulator and below Jev. Both answer through Structured
  Outputs. On `select`, GPT-6 Luna is +0.0015 [-0.0197, +0.0234] against QuantTrio and -0.0435
  against Jev (6 benchmarks). DeepSeek V4.1 Flash is +0.0104 [-0.0071, +0.0277] and -0.0562 (7
  benchmarks). Luna costs 28% more than Jev per item, DeepSeek about 3x.
- At an electricity-only estimate ($0.1107 per GPU-hour), QuantTrio costs 26% of Jev's price per
  item (`select`). Hardware is not included.
- CLM-8B, a dual encoder that answers Jev's wire format, scores 24.9% macro accuracy on `holdout`
  over the 9 benchmarks, 57.6 points below Jev. It is not a finalist
  ([CLM-8B](#clm-8b-dual-encoder)).

Yelp is dropped (2026-09-25). Its levels are shown to the models as `0) 1 star` ... `4) 5 stars`,
which reads ambiguously: the models write `5`, which is not a label, and `4` is ambiguous. The
protocol now has 9 benchmarks (`jevemu.eval.splits.DATASETS`). Summaries, comparisons and statistics
skip Yelp's records. Stage 1 and Stage 2 were run and decided with Yelp and are kept as recorded.
The current Stage 2b headline is in that section, in the paragraph that drops Yelp. The GPT-6 Luna
section already leaves Yelp out. The chosen configuration does not change.

Selection picks by accuracy on the `select` half of the ten built-in benchmarks. Calibrators are
then fitted, and final numbers reported, on the `holdout` half, which selection never sees
(`jevemu.eval.splits`).

## Scoring now: one call per question

Sep 27, 2026. Stages 1 and 2, the cost check and the fast-MoE tuning below ran with the
multi-call `auto_noecho` policy and are kept as multi-call history (before one-call scoring).
They chose the preset, the layout and the fast configuration. The finalists (QuantTrio 27B, the
AWQ and GPTQ Qwen3.6-35B-A3B builds, Gemma 4 26B-A4B) and every candidate's `state_first` screen
are rerun with the current method. The old multi-call results are archived; the reports no
longer use them. What changed:

- One-call scoring (`auto_single`, `jevemu.scoring.SingleCallStrategy`) answers every question in
  one model call, and every answer label is one token.
  - Up to 32 options: letters, read in one constrained call. This is exactly what `auto_noecho`
    did.
  - More than 32 options (banking77: 77, CLINC150: 150): two-capital codes `AA` .. `ZZ`. A code
    is kept only if it is a single token both as `XY` and as ` XY` (Qwen: 522 usable codes,
    Gemma: 588). One constrained call reads every code from one top-k, so the server now runs
    with `--max-logprobs 576` (was 64). vLLM's `logprob_token_ids` can read chosen token ids
    directly, but it takes at most 128 ids, too few for this.
  - The prompt text and `template_id` (`state_first-5d29289f8b87`) are unchanged.
  - Diagnostics record `n_backend_calls`. The reports refuse any run where it is not 1.
- A live check compared the codes with the trie on the Qwen3.6-35B-A3B GPTQ build, 300 screen items
  each: banking77 228/300 with codes against 226 with the trie, CLINC150 257/300 against 263. The
  codes hold about 0.98 of the probability mass. The trie had needed 4.46 (banking77) and 3.03
  (CLINC150) calls per item on the Qwen MoEs, and 2.03 and 1.62 on Gemma.
- The Gemma 4 prompt was fixed. vLLM rendered the `Answer:` prefill as a continued assistant
  message. The Gemma 4 12B and 26B-A4B templates (thinking off) open the model turn with an empty
  thought channel only on the generation prompt, so those models got `<|turn>model\nAnswer:` instead
  of `<|turn>model\n<|channel>thought\n<channel|>Answer:`. Now every model's messages are rendered
  with the generation prompt, `Answer:` is appended as token ids, and the ids go to
  `/v1/completions`. Every other preset gets identical token ids before and after. A second bug was
  fixed at the same time: Gemma's byte tokens decode to the same text as normal tokens (`<0x41>` is
  `"A"`), so logprobs are now keyed by token id, not text. In a live A/B (50 items of each of the 9
  benchmarks), 26B-A4B went from 0.693 to 0.689 macro accuracy and its NLL from 1.548 to 1.447; 12B
  from 0.669 to 0.682 and its NLL from 1.289 to 1.439, and the 12B's probability outside the labels
  disappears. Both models' runs were redone with the fix
  ([candidates.md](candidates.md#small-models-for-the-speedquality-study), finding 5).
- QuantTrio 27B no longer has a debiaser. Its `holdout` run used online PriDe before; it now runs
  without one, like every other system.
- The electricity estimate is higher. The RTX 3090 draws about 410 W while running (observed, not
  metered per run). At $0.27/kWh (New Jersey average) that is $0.1107 per GPU-hour. The earlier
  placeholder was 280 W ($0.0756 per GPU-hour). Costs are linear in watts, so a cost at the old
  estimate × 1.464 gives the current one.

### What one-call scoring changed

Old numbers here are the multi-call `auto_noecho` runs of the same presets, paired on the same
items. The Gemma 4 rows mix two changes (see below).

- The finalists barely moved. Their macro accuracy over 9 benchmarks changed by less than 0.006: on
  `select`, QuantTrio 27B went from 0.7521 to 0.7520, AWQ fast from 0.7321 to 0.7264 and GPTQ from
  0.7208 to 0.7224. The two 27B builds are still tied on `holdout` (cyankiwi − QuantTrio -0.0014
  [-0.0101, +0.0069]). For the Qwen builds only banking77 and CLINC150 get different requests; on
  the other 7 benchmarks the requests are byte-identical. On those two sets the codes cost the
  finalists up to about 0.03: CLINC150 on the MoEs (AWQ fast -0.032 on `select`, GPTQ -0.021, Gemma
  -0.017 to -0.023) and banking77 on QuantTrio (-0.029 on `holdout`).
- Small models lose a lot of accuracy on the codes. Paired on the screen, Qwen3.5-2B went from 0.65
  to 0.26 on CLINC150 and moved -0.14 on banking77; SmolLM3-3B moved -0.24 on banking77 and -0.15 on
  CLINC150; Llama 3.2 3B -0.10 and -0.20; Phi-4-mini -0.05 and -0.11; Qwen3-4B -0.07 and -0.09.
  Their 9-benchmark macro falls with it: Qwen3.5-2B from 0.551 to 0.492, SmolLM3-3B from 0.547 to
  0.503, Llama 3.2 3B from 0.530 to 0.494, Phi-4-mini from 0.555 to 0.538. These models put a lot of
  probability outside the codes: on Qwen3.5-2B, 45 banking77 and 46 CLINC150 items (of 300 each)
  have less than half their mass on the codes. The 27B to 35B models move by about ±0.03 on these
  two sets.
- Probabilities on the intent sets got worse for the Qwen builds. The codes make their raw
  probabilities more overconfident while accuracy barely moves. Raw NLL on `holdout`: AWQ fast
  banking77 +0.50 and CLINC150 +0.32, GPTQ +0.46 and +0.33, the dense 27B builds +0.19 to +0.22 and
  +0.04 to +0.05. Calibration closes only part of this. Calibrated macro NLL on `holdout` went from
  0.666 to 0.688 for QuantTrio, from 0.761 to 0.835 for AWQ fast and from 0.767 to 0.838 for GPTQ.
  So the calibrated NLL gap to Jev grew, from +0.112 to +0.133 for QuantTrio and from +0.206 to
  +0.280 for AWQ fast
  ([calibration.md](calibration.md#current-results-one-call-per-question-holdout)).
- One call is faster only on the Qwen3.5, Qwen3.6 and Qwen3.8 models. On the Qwen3.6 finalists the
  trie had needed about 2 to 4.7 calls per item on banking77 and CLINC150. Pooled q/s over 9
  benchmarks on `select` went from 2.94 to 4.44 for QuantTrio, from 11.07 to 20.07 for AWQ fast and
  from 12.40 to 25.11 for GPTQ (the [screen table](#speedquality-trade-off) has every preset). For
  Gemma 4, Qwen3 and Qwen2.5 the intent-set speed stayed the same or dropped. On `holdout`, Gemma
  26B-A4B went from 10.00 to 9.07 q/s on banking77 and from 7.90 to 6.96 on CLINC150 (from 22.03 to
  20.44 over 9). On the screen, Qwen3-32B went from 2.13 to 1.93 on banking77.
- The Gemma 4 prompt fix raised Gemma's accuracy. Gemma 4 26B-A4B went from 0.714 to 0.735 on
  `holdout`, and is now tied with the AWQ fast MoE (+0.0045 [-0.0104, +0.0195]); calibrated, their
  NLL is tied too (-0.002 [-0.027, +0.023]). Gemma 4 12B went from 0.672 to 0.708 on the screen.
  Every older Gemma 12B and 26B-A4B run had the render bug, so these changes mix the fix and the
  codes; neither effect is measured alone.
- The Qwen screen and `select` runs, and the other non-Gemma screens, ran at commit `79fa71f` (chat
  path). The Gemma 12B and 26B-A4B runs and every `holdout` run ran at `5237052` (`/tokenize` plus
  `/v1/completions`). The Qwen prompt token ids are identical on the two paths, and q/s agrees
  across halves (AWQ fast 20.07 on `select` against 19.70 on `holdout`, GPTQ 25.11 against 25.10,
  QuantTrio 4.44 against 4.44). Gemma 4 E4B was not rerun: its template has no empty thought
  channel, so its prompt is unchanged.
- The work took 8.9 h of GPU time (7.5 h of runs, 1.4 h of server startups): screen 2.6 h, `select`
  3.3 h, `holdout` 3.1 h. Of that, 0.45 h went to Gemma runs made before the prompt fix and redone.
  Every preset started on its first attempt with its own flags at `--max-logprobs 576`.

## Protocol

- The items are the `select` half of every benchmark in `jevemu.eval.splits.DATASETS` (seed 0),
  19,841 in all. Each run freezes the split to `runs/select/splits/<benchmark>.select.jsonl` (with
  its `items_sha256`), and every later system reads that file. So all systems answer byte-identical
  requests: `SystemOneRequest(state, questions={"answer": question})`. The IDK ("I don't know")
  benchmarks use evaluate-idk's option order, with "I don't know" as option E.
- The runner, `scripts/run_split.py run` (`jevemu.bench.runner.run_split`), writes one record per
  item to `runs/select/<system_id>/<benchmark>.select.jsonl`: answer or error, latency, usage, cost
  and emulator diagnostics. It also writes a `manifest.json`: system identity, split metadata, git
  commit, start and end, counts and spend. Reruns skip answered items and retry failed ones. The Jev
  budget guard stops a run cleanly. Per-item records stay in `runs/` (gitignored); this page reports
  only aggregates.
- Scoring (`jevemu.eval.item_scores`) renormalizes probabilities to sum to 1 (Jev rounds each to
  0.01). For NLL, Brier and ECE, lower is better.
  - choice: correct iff `choice == gold`. Choosing "I don't know" counts as wrong; the share of
    such answers is the IDK rate.
  - noul (BoolQ): predicts yes iff P(yes) ≥ 0.5.
  - score (SST-5, Yelp): predicts the most likely level. MAE (mean absolute error) is
    |reported `score` − gold level|. Within-1 is the share of predicted levels at most one level
    from gold.
  - NLL (negative log-likelihood) = −log max(p(gold), 1e-6).
  - Brier score: multiclass, Σₖ (pₖ − [k = gold])², range [0, 2] (noul: 2 (P(yes) − y)²).
  - ECE (expected calibration error): top-label, 10 equal-mass bins, with the predicted
    outcome's probability as the confidence.
- Statistics come from `jevemu.bench.compare`. Each benchmark's accuracy has a 95% bootstrap
  confidence interval (CI): 10,000 resamples, seed 0, BCa from 500 items and percentile below. The
  macro average weighs every benchmark equally. Its CI comes from a stratified bootstrap (items
  resampled within each benchmark). Candidates are paired with Jev on shared item ids:
  - accuracy: Δ with a paired bootstrap CI, plus an exact McNemar test (on the items only one
    of the two systems gets right);
  - NLL and Brier: Δ with a paired bootstrap CI;
  - macro Δ: a stratified bootstrap CI.

  A CI that crosses zero means "no evidence of a difference".

```bash
uv run python scripts/run_split.py run --system jev --benchmarks all --split select \
    --out runs/select --concurrency 16 --max-usd 1.0
uv run python scripts/run_split.py run --system emulator --system-id NAME --out runs/select \
    --strategy auto_single --renderer state_first   # the current method
uv run python scripts/run_split.py summarize runs/select/NAME
uv run python scripts/run_split.py compare runs/select/NAME runs/select/jev
```

## Jev reference (select half)

`jev-1.13.0` answered all 19,841 `select` items. The run went from 02:54 to 03:11 UTC on
Sep 25, 2026, at 16 requests in flight, held at Jev's 1,200 requests/min limit (20 items/s),
at jevemu commit `651fe0a`. The table below is `scripts/run_split.py summarize runs/select/jev`.

Latency is Jev's recorded HTTP round trip, measured from this client machine. Spend is
input tokens × $0.042 / 1M. The IDK rate is the share of "I don't know" answers; those answers
count as wrong in accuracy. MAE and within-1 apply to score questions only.

| Benchmark | Type | n | Errors | Accuracy [95% CI] | IDK rate | NLL | Brier | ECE | MAE | Within-1 | p50 ms | p95 ms | Spend $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | choice | 99/99 | 0 | 0.6465 [0.5556, 0.7374] | 0.030 | 0.978 | 0.479 | 0.085 |  |  | 149 | 857 | 0.0024 |
| lexam_en_idk | choice | 310/310 | 0 | 0.7452 [0.6968, 0.7935] | 0.016 | 0.803 | 0.388 | 0.064 |  |  | 133 | 181 | 0.0064 |
| mmlu_pro | choice | 6016/6016 | 0 | 0.8263 [0.8163, 0.8356] |  | 0.743 | 0.268 | 0.049 |  |  | 131 | 175 | 0.1401 |
| arc_challenge | choice | 586/586 | 0 | 0.9744 [0.9573, 0.9846] |  | 0.153 | 0.043 | 0.011 |  |  | 134 | 177 | 0.0093 |
| ag_news | choice | 3800/3800 | 0 | 0.8845 [0.8737, 0.8939] |  | 0.665 | 0.189 | 0.069 |  |  | 139 | 198 | 0.0572 |
| banking77 | choice | 1540/1540 | 0 | 0.8039 [0.7831, 0.8234] |  | 1.158 | 0.302 | 0.088 |  |  | 138 | 198 | 0.0661 |
| clinc150 | choice | 2250/2250 | 0 | 0.9240 [0.9120, 0.9338] |  | 0.430 | 0.122 | 0.020 |  |  | 139 | 208 | 0.1336 |
| boolq | noul | 1635/1635 | 0 | 0.9083 [0.8936, 0.9211] |  | 0.257 | 0.148 | 0.019 |  |  | 132 | 197 | 0.0277 |
| sst5 | score | 1105/1105 | 0 | 0.5756 [0.5457, 0.6036] |  | 1.378 | 0.595 | 0.184 | 0.470 | 0.966 | 134 | 183 | 0.0157 |
| yelp_stars | score | 2500/2500 | 0 | 0.6664 [0.6476, 0.6844] |  | 0.942 | 0.472 | 0.134 | 0.387 | 0.973 | 135 | 202 | 0.0510 |
| **macro (10)** |  | 19841/19841 | 0 | 0.7955 [0.7837, 0.8069] |  | 0.751 [0.715, 0.788] | 0.301 [0.288, 0.314] |  |  |  |  |  | 0.5094 |

- The spend was $0.5094 for 19,802 billed calls, under the $1.00 cap. The other 39 items came free
  from the response cache. 30 came from an earlier smoke run of this runner (3 items per benchmark,
  $0.0007; 3 of its items were already cached by an earlier 20-item LEXam-en smoke run). The other 9
  came from that 20-item run. Reruns read the cache and are free.
- No errors were left at the end. One banking77 request hit the client's 60 s read timeout
  (`JevConnectionError`, transient). It was recorded and the run continued. A resume re-asked only
  that item (1 call, 1,539 skipped). Jev returned no validation errors.
- Every choice and score answer gives a probability for every option or level. The rounded
  probabilities of each answer sum to between 0.99 and 1.00, including the 77- and 150-option
  benchmarks. In 13 of 19,841 answers (12 MMLU-Pro, 1 AG News), `choice` is 0.01 below the largest
  rounded probability. These look like near-ties broken before rounding. Accuracy always uses
  `choice`.
- Paired comparisons with Jev carry the most power on the large, high-accuracy benchmarks (MMLU-Pro,
  AG News, CLINC150). On the two ordinal benchmarks Jev is weak on exact level (0.58 and 0.67) but
  within one level on about 97% of items. GPQA-Diamond has only 99 items and a CI about ±0.09 wide,
  so it separates candidates only by large margins.

## Stage 1 screen

*Multi-call history (before one-call scoring): `auto_noecho`, `question_first`, the old prefill
rendering.*

All 16 presets in [candidates.md](candidates.md) answered the `screen` split. It is a stratified
subset of `select` with at most 300 items per benchmark: 2,799 items (GPQA-Diamond 99, 300 for
every other benchmark). Every preset used the same scoring policy (below), the then-default
renderer (`question_first`), 16 items in flight and diagnostics on. Each ran on a new container
with an empty prefix cache. The sweep ran from 03:56 to 07:17 UTC on Sep 25, 2026, at jevemu
commit `f418e53` plus the uncommitted sweep scripts. All 16 finished on the first attempt with 0
item errors.

Jev's answers to the same items were replayed from its `select` run through the response cache.
`run_split.py run --system jev --split screen --max-usd 0` made 0 network calls (2,799 cache
hits, $0) and wrote `runs/select/jev/*.screen.jsonl`. So `compare` pairs every candidate with Jev
on identical items. Jev's screen macro accuracy is 0.8043 [0.7890, 0.8200] (select: 0.7955).

```bash
uv run python scripts/sweep_presets.py run --presets P1,P2,... --split screen --out runs/select \
    --strategy auto_noecho --renderer question_first --concurrency 16   # serve, run, stop; resumable
uv run python scripts/sweep_presets.py report --presets P1,P2,... --split screen \
    --out runs/select --strategy auto_noecho --renderer question_first --reference runs/select/jev
```

For each preset, `sweep_presets.py run` stops any running server, starts the preset
(`scripts/serve_vllm.sh`), runs `run_split.py` into `runs/select/<preset>.auto_noecho/`, and stops
the server even if the run fails. A preset whose run directory already holds a complete run is
skipped, so rerunning the command resumes the sweep. A preset that fails to start or to finish is
recorded in `runs/select/sweep/sweep.screen.jsonl`, and the sweep moves on to the next one.
`report` prints the tables below; they are also saved in `runs/select/sweep/report.screen.md`.

### Multi-call history: the `auto_noecho` scoring policy

Plain `auto` falls back to echo scoring (S4) for every question whose letter labels do not fit
in the top 64 (more than 32 options). Here that means banking77 (77 intents) and CLINC150 (150).
Echo sends one request per option, and on vLLM 0.30.0 each one is a full prefill that never
reads the prefix cache.

A cost check timed each strategy on the first `select` items, on a dense 27B preset
(`qwen3.8-27b-int4-redhat`, one server, 03:31 to 03:54 UTC). Two facts set the cost:

- Prefill runs at about 1,300 tokens/s on this GPU (vLLM's log, `--max-num-batched-tokens=1024`).
- The Qwen3.8 hybrid model caches prefixes only in 784-token blocks. Banking77's prompts are
  about 520 tokens and CLINC150's about 690, so neither ever hits the cache.

So time is proportional to prompt tokens.

| Strategy | Benchmark | Items | Calls / item | Prompt tokens / item | Measured | 300 items at 1,300 tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| `auto` (S2 constrained letters) | the other 8 | 32 each | 1 | 107 (SST-5) to 287 (GPQA) | 4.8-10.7 items/s at 16 in flight | 5.1 min for all 2,499 |
| `auto` = S4 echo | banking77 | 3 | 77 | 40,311 | 31.6 s / item | 2.6 h |
| `auto` = S4 echo | clinc150 | 2 | 150 | 104,125 | 80.9 s / item | 6.7 h |
| S3 trie, τ = 1e-4 (echo fills) | banking77 | 5 | 16.8 | 8,728 | 6.9 s / item | 34 min |
| S3 trie, τ = 1e-4 (echo fills) | clinc150 | 2 | 78 | 54,053 | 42.3 s / item | 3.5 h |
| `auto_noecho`, τ = 1e-4 | banking77 | 16 | 5.5 | 2,868 | 0.40 items/s | 11 min |
| `auto_noecho`, τ = 1e-4 | clinc150 | 16 | 4.1 | 2,823 | 0.41 items/s | 11 min |
| **`auto_noecho`, τ = 1e-3** | banking77 | 16 | 3.1 | 1,628 | 0.62 items/s | **6.3 min** |
| **`auto_noecho`, τ = 1e-3** | clinc150 | 16 | 2.1 | 1,477 | 0.68 items/s | **5.7 min** |
| `auto_noecho`, τ = 1e-2 | banking77 | 16 | 2.4 | 1,237 | 0.78 items/s | 4.8 min |
| `auto_noecho`, τ = 1e-2 | clinc150 | 16 | 1.2 | 825 | 1.00 items/s | 3.2 min |

The plain trie (S3) is expensive because of how it fills missing labels. CLINC150's 150 intents
start with 114 distinct first tokens, but the root query returns only the top 64. Each of the
other 50 is echo-scored, one full prefill each. Deeper nodes add more: where one intent ends and
a longer one continues, the node is read unconstrained. Items/s measured on only 16 items
understates steady-state throughput, so the projections use tokens.

The policy is `jevemu.scoring.NoEchoAutoStrategy`, recorded in every manifest as
`auto_noecho_tau0.001` (`Emulator(backend, strategy=NoEchoAutoStrategy())`; in the scripts,
`--strategy auto_noecho`). It is `auto` running on the backend with echo switched off:

- ≤ 32 options (every benchmark except banking77 and CLINC150): what `auto` picks, S2
  constrained letters (digits after the gap for SST-5 and Yelp, `Yes`/`No` for BoolQ). This is
  one call and identical to `auto`.
- Banking77 and CLINC150: S3 trie over the option keys (the intent names), pruning decision
  points below τ = 1e-3.
- A label missing from a top-64 gets the upper bound (the smallest returned logprob, marked
  truncated) instead of an echo fill. This happens at CLINC150's root, and at banking77's root on
  the Qwen3 tokenizer, whose choice grammar also admits sub-word first tokens. Those labels sat
  near e⁻²⁴.

On the same 16 items of each benchmark, τ = 1e-3 and τ = 1e-4 gave the same argmax on all 32
items, with |Δ NLL| ≤ 0.004. τ = 1e-2 also gave the same argmax, with |Δ NLL| ≤ 0.042. Every
label under a pruned subtree gets the subtree's path probability, an upper bound below 1e-3.
Renormalizing over that added mass rescales the other labels by the same small factor.

The dense-27B screens took 16.3 to 18.4 min, against a projected 5.1 + 6.3 + 5.7 ≈ 17 min. Startup
added 2.1 min. For `qwen3.6-27b-int4-cyankiwi` the run took 974 s: 324 s for the eight constrained
benchmarks, 346 s for banking77 and 304 s for CLINC150.

### Ranking

The ranking is by macro accuracy over the 10 benchmarks, with ties broken by macro NLL. Every Δ
is paired on identical items (`compare`). Δ vs #1 is the preset minus
`qwen3.6-27b-int4-cyankiwi`; Δ vs Jev is the preset minus Jev. Items/s is answered items over the
runner's wall time; startup (from `compose up` to a healthy server) is listed separately.

| Rank | Preset | Macro acc. [95% CI] | Macro NLL [95% CI] | Δ acc. vs #1 [95% CI] | Δ acc. vs Jev [95% CI] | Δ NLL vs Jev [95% CI] | Items/s | Startup + run, min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `qwen3.6-27b-int4-cyankiwi` | 0.6877 [0.6713, 0.7047] | 0.874 [0.826, 0.923] | — | -0.1166 [-0.1340, -0.0992] | +0.143 [+0.088, +0.195] | 2.87 | 2.1 + 16.3 |
| 2 | `qwen3.6-27b-int4-quanttrio` | 0.6797 [0.6633, 0.6964] | 0.864 [0.820, 0.910] | -0.0080 [-0.0201, +0.0038] | -0.1246 [-0.1431, -0.1062] | +0.134 [+0.079, +0.185] | 2.81 | 2.1 + 16.7 |
| 3 | `qwen3.6-27b-int4-groxaxo` | 0.6720 [0.6559, 0.6887] | 0.901 [0.855, 0.949] | -0.0157 [-0.0288, -0.0033] | -0.1323 [-0.1500, -0.1149] | +0.170 [+0.115, +0.224] | 2.65 | 2.1 + 17.7 |
| 4 | `qwen3.8-27b-int4-palmfuture` | 0.6613 [0.6449, 0.6777] | 0.942 [0.893, 0.993] | -0.0264 [-0.0422, -0.0113] | -0.1430 [-0.1607, -0.1255] | +0.211 [+0.157, +0.263] | 2.86 | 2.1 + 16.4 |
| 5 | `qwen3.8-27b-int4-groxaxo` | 0.6610 [0.6440, 0.6777] | 0.913 [0.869, 0.956] | -0.0267 [-0.0435, -0.0106] | -0.1433 [-0.1614, -0.1253] | +0.182 [+0.127, +0.235] | 2.54 | 2.1 + 18.4 |
| 6 | `qwen3.8-27b-int4-btbtyler09` | 0.6586 [0.6422, 0.6754] | 0.937 [0.891, 0.984] | -0.0291 [-0.0449, -0.0137] | -0.1457 [-0.1634, -0.1283] | +0.207 [+0.152, +0.259] | 2.67 | 2.1 + 17.5 |
| 7 | `qwen3.8-27b-awq` | 0.6583 [0.6416, 0.6750] | 0.937 [0.890, 0.985] | -0.0294 [-0.0461, -0.0132] | -0.1460 [-0.1641, -0.1286] | +0.207 [+0.152, +0.259] | 2.62 | 2.1 + 17.8 |
| 8 | `qwen3.6-35b-a3b-int4-palmfuture` | 0.6543 [0.6377, 0.6713] | 0.996 [0.944, 1.048] | -0.0334 [-0.0488, -0.0181] | -0.1500 [-0.1681, -0.1317] | +0.266 [+0.206, +0.322] | 13.34 | 2.2 + 3.5 |
| 9 | `qwen3.8-27b-int4-redhat` | 0.6513 [0.6346, 0.6680] | 0.948 [0.902, 0.993] | -0.0364 [-0.0532, -0.0202] | -0.1531 [-0.1711, -0.1353] | +0.217 [+0.163, +0.270] | 2.56 | 2.1 + 18.2 |
| 10 | `qwen3.5-35b-a3b-int4` | 0.6416 [0.6249, 0.6580] | 0.997 [0.947, 1.047] | -0.0461 [-0.0625, -0.0301] | -0.1627 [-0.1805, -0.1449] | +0.266 [+0.207, +0.323] | 13.79 | 2.2 + 3.4 |
| 11 | `qwen3.5-9b-bf16` | 0.6100 [0.5929, 0.6267] | 1.044 [1.003, 1.085] | -0.0777 [-0.0955, -0.0604] | -0.1943 [-0.2134, -0.1752] | +0.313 [+0.251, +0.372] | 10.15 | 1.8 + 4.6 |
| 12 | `qwen3.5-9b-int8` | 0.6090 [0.5917, 0.6257] | 1.042 [1.002, 1.084] | -0.0787 [-0.0967, -0.0613] | -0.1953 [-0.2143, -0.1762] | +0.312 [+0.250, +0.371] | 9.80 | 1.8 + 4.8 |
| 13 | `qwen3-32b-int4` | 0.6053 [0.5882, 0.6227] | 1.544 [1.459, 1.630] | -0.0824 [-0.1008, -0.0643] | -0.1990 [-0.2188, -0.1792] | +0.814 [+0.729, +0.900] | 8.39 | 1.5 + 5.6 |
| 14 | `qwen2.5-32b-int4` | 0.5986 [0.5813, 0.6159] | 3.052 [2.882, 3.222] | -0.0891 [-0.1086, -0.0697] | -0.2057 [-0.2257, -0.1863] | +2.322 [+2.160, +2.486] | 8.70 | 1.2 + 5.4 |
| 15 | `qwen3-30b-a3b-2507-int4-redhat` | 0.5970 [0.5797, 0.6144] | 2.966 [2.808, 3.127] | -0.0907 [-0.1101, -0.0719] | -0.2073 [-0.2280, -0.1869] | +2.236 [+2.088, +2.385] | 44.62 | 1.3 + 1.1 |
| 16 | `qwen3-14b-int4` | 0.5818 [0.5658, 0.5985] | 2.783 [2.633, 2.934] | -0.1058 [-0.1249, -0.0868] | -0.2225 [-0.2419, -0.2023] | +2.052 [+1.909, +2.199] | 18.57 | 1.2 + 2.5 |

Other paired macro Δ accuracy (a − b) near the cut, with Δ NLL:

| a − b | Δ acc. [95% CI] | Δ NLL [95% CI] |
| --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` − `qwen3.6-27b-int4-groxaxo` (#2 − #3) | +0.0077 [-0.0050, +0.0205] | -0.037 [-0.057, -0.016] |
| `qwen3.6-27b-int4-groxaxo` − `qwen3.8-27b-int4-palmfuture` (#3 − #4) | +0.0107 [-0.0043, +0.0254] | -0.041 [-0.072, -0.011] |
| `qwen3.6-27b-int4-groxaxo` − `qwen3.8-27b-int4-groxaxo` (#3 − #5) | +0.0110 [-0.0045, +0.0264] | -0.012 [-0.040, +0.017] |
| `qwen3.6-27b-int4-groxaxo` − `qwen3.8-27b-awq` (#3 − #7) | +0.0137 [-0.0018, +0.0297] | -0.036 [-0.066, -0.007] |
| `qwen3.6-27b-int4-groxaxo` − `qwen3.6-35b-a3b-int4-palmfuture` (#3 − #8) | +0.0177 [+0.0016, +0.0338] | -0.095 [-0.130, -0.060] |
| `qwen3.8-27b-int4-palmfuture` − `qwen3.8-27b-int4-groxaxo` (#4 − #5) | +0.0003 [-0.0115, +0.0120] | +0.029 [+0.015, +0.044] |
| `qwen3.8-27b-int4-palmfuture` − `qwen3.8-27b-int4-redhat` (#4 − #9) | +0.0100 [-0.0010, +0.0214] | -0.006 [-0.020, +0.010] |
| `qwen3.8-27b-int4-palmfuture` − `qwen3.6-35b-a3b-int4-palmfuture` (#4 − #8) | +0.0070 [-0.0094, +0.0234] | -0.054 [-0.088, -0.020] |
| `qwen3.6-35b-a3b-int4-palmfuture` − `qwen3.5-35b-a3b-int4` (#8 − #10) | +0.0127 [-0.0014, +0.0271] | -0.000 [-0.027, +0.025] |
| `qwen3.5-9b-bf16` − `qwen3.5-9b-int8` (#11 − #12) | +0.0010 [-0.0027, +0.0047] | +0.001 [-0.001, +0.003] |

Accuracy per benchmark. The IDK columns give the share of "I don't know" answers, which count as
wrong.

| Preset | GPQA | IDK | LEXam | IDK | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | 0.364 | 0.051 | 0.407 | 0.010 | 0.527 | 0.890 | 0.883 | 0.813 | 0.913 | 0.877 | 0.587 | 0.617 |
| `qwen3.6-27b-int4-quanttrio` | 0.364 | 0.081 | 0.370 | 0.017 | 0.503 | 0.897 | 0.887 | 0.817 | 0.913 | 0.867 | 0.593 | 0.587 |
| `qwen3.6-27b-int4-groxaxo` | 0.303 | 0.101 | 0.377 | 0.007 | 0.507 | 0.887 | 0.883 | 0.807 | 0.927 | 0.860 | 0.583 | 0.587 |
| `qwen3.8-27b-int4-palmfuture` | 0.283 | 0.202 | 0.363 | 0.007 | 0.467 | 0.867 | 0.870 | 0.810 | 0.903 | 0.853 | 0.587 | 0.610 |
| `qwen3.8-27b-int4-groxaxo` | 0.343 | 0.081 | 0.350 | 0.003 | 0.473 | 0.870 | 0.850 | 0.807 | 0.900 | 0.853 | 0.587 | 0.577 |
| `qwen3.8-27b-int4-btbtyler09` | 0.293 | 0.182 | 0.340 | 0.063 | 0.487 | 0.857 | 0.863 | 0.813 | 0.897 | 0.840 | 0.577 | 0.620 |
| `qwen3.8-27b-awq` | 0.313 | 0.141 | 0.373 | 0.003 | 0.463 | 0.850 | 0.863 | 0.813 | 0.893 | 0.843 | 0.583 | 0.587 |
| `qwen3.6-35b-a3b-int4-palmfuture` | 0.293 | 0.061 | 0.343 | 0.057 | 0.530 | 0.833 | 0.887 | 0.780 | 0.887 | 0.860 | 0.567 | 0.563 |
| `qwen3.8-27b-int4-redhat` | 0.263 | 0.242 | 0.347 | 0.107 | 0.483 | 0.857 | 0.870 | 0.810 | 0.897 | 0.843 | 0.577 | 0.567 |
| `qwen3.5-35b-a3b-int4` | 0.283 | 0.091 | 0.327 | 0.007 | 0.470 | 0.820 | 0.883 | 0.747 | 0.923 | 0.873 | 0.593 | 0.497 |
| `qwen3.5-9b-bf16` | 0.323 | 0.010 | 0.283 | 0.000 | 0.383 | 0.757 | 0.877 | 0.737 | 0.877 | 0.850 | 0.567 | 0.447 |
| `qwen3.5-9b-int8` | 0.323 | 0.000 | 0.267 | 0.000 | 0.383 | 0.753 | 0.873 | 0.743 | 0.873 | 0.850 | 0.563 | 0.460 |
| `qwen3-32b-int4` | 0.283 | 0.172 | 0.323 | 0.047 | 0.380 | 0.820 | 0.797 | 0.740 | 0.840 | 0.847 | 0.513 | 0.510 |
| `qwen2.5-32b-int4` | 0.263 | 0.242 | 0.300 | 0.047 | 0.407 | 0.797 | 0.820 | 0.717 | 0.837 | 0.823 | 0.573 | 0.450 |
| `qwen3-30b-a3b-2507-int4-redhat` | 0.333 | 0.010 | 0.283 | 0.007 | 0.383 | 0.743 | 0.817 | 0.690 | 0.870 | 0.847 | 0.543 | 0.460 |
| `qwen3-14b-int4` | 0.182 | 0.343 | 0.297 | 0.067 | 0.370 | 0.797 | 0.783 | 0.733 | 0.853 | 0.870 | 0.487 | 0.447 |
| **Jev** | 0.646 | 0.030 | 0.753 | 0.017 | 0.873 | 0.970 | 0.900 | 0.810 | 0.933 | 0.883 | 0.587 | 0.687 |

NLL per benchmark:

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | 1.437 | 1.264 | 1.443 | 0.341 | 0.480 | 1.004 | 0.343 | 0.335 | 1.036 | 1.055 |
| `qwen3.6-27b-int4-quanttrio` | 1.468 | 1.322 | 1.419 | 0.313 | 0.427 | 0.940 | 0.319 | 0.304 | 1.067 | 1.064 |
| `qwen3.6-27b-int4-groxaxo` | 1.563 | 1.295 | 1.500 | 0.366 | 0.452 | 0.911 | 0.372 | 0.331 | 1.071 | 1.150 |
| `qwen3.8-27b-int4-palmfuture` | 1.552 | 1.455 | 1.507 | 0.396 | 0.554 | 1.082 | 0.442 | 0.358 | 1.080 | 0.994 |
| `qwen3.8-27b-int4-groxaxo` | 1.469 | 1.426 | 1.496 | 0.420 | 0.526 | 0.943 | 0.421 | 0.352 | 0.990 | 1.084 |
| `qwen3.8-27b-int4-btbtyler09` | 1.561 | 1.541 | 1.492 | 0.424 | 0.524 | 1.010 | 0.415 | 0.362 | 1.050 | 0.996 |
| `qwen3.8-27b-awq` | 1.500 | 1.426 | 1.499 | 0.422 | 0.566 | 1.011 | 0.429 | 0.346 | 1.095 | 1.079 |
| `qwen3.6-35b-a3b-int4-palmfuture` | 1.503 | 1.463 | 1.539 | 0.466 | 0.578 | 1.133 | 0.450 | 0.381 | 1.216 | 1.236 |
| `qwen3.8-27b-int4-redhat` | 1.626 | 1.507 | 1.524 | 0.431 | 0.533 | 0.980 | 0.421 | 0.372 | 1.056 | 1.026 |
| `qwen3.5-35b-a3b-int4` | 1.559 | 1.412 | 1.594 | 0.480 | 0.531 | 1.279 | 0.420 | 0.364 | 1.073 | 1.256 |
| `qwen3.5-9b-bf16` | 1.414 | 1.470 | 1.776 | 0.761 | 0.584 | 1.257 | 0.527 | 0.359 | 1.044 | 1.244 |
| `qwen3.5-9b-int8` | 1.407 | 1.471 | 1.778 | 0.759 | 0.587 | 1.260 | 0.525 | 0.360 | 1.045 | 1.232 |
| `qwen3-32b-int4` | 2.081 | 1.929 | 2.463 | 0.724 | 0.815 | 2.141 | 0.846 | 0.521 | 1.854 | 2.071 |
| `qwen2.5-32b-int4` | 5.063 | 4.300 | 4.715 | 1.415 | 1.711 | 2.489 | 1.541 | 1.411 | 3.471 | 4.408 |
| `qwen3-30b-a3b-2507-int4-redhat` | 3.999 | 5.406 | 5.043 | 2.032 | 1.417 | 2.679 | 1.028 | 1.193 | 3.429 | 3.440 |
| `qwen3-14b-int4` | 5.205 | 3.179 | 4.151 | 1.384 | 1.673 | 2.417 | 0.976 | 1.148 | 3.857 | 3.840 |
| **Jev** | 0.978 | 0.786 | 0.619 | 0.156 | 0.565 | 1.275 | 0.408 | 0.279 | 1.372 | 0.870 |

ECE per benchmark:

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | 0.223 | 0.081 | 0.079 | 0.026 | 0.071 | 0.097 | 0.035 | 0.043 | 0.159 | 0.205 |
| `qwen3.6-27b-int4-quanttrio` | 0.168 | 0.112 | 0.063 | 0.041 | 0.058 | 0.090 | 0.026 | 0.053 | 0.169 | 0.240 |
| `qwen3.6-27b-int4-groxaxo` | 0.166 | 0.059 | 0.061 | 0.048 | 0.062 | 0.081 | 0.033 | 0.056 | 0.173 | 0.242 |
| `qwen3.8-27b-int4-palmfuture` | 0.166 | 0.074 | 0.081 | 0.043 | 0.094 | 0.096 | 0.038 | 0.066 | 0.162 | 0.203 |
| `qwen3.8-27b-int4-groxaxo` | 0.135 | 0.063 | 0.053 | 0.084 | 0.051 | 0.081 | 0.030 | 0.057 | 0.142 | 0.211 |
| `qwen3.8-27b-int4-btbtyler09` | 0.149 | 0.076 | 0.084 | 0.053 | 0.073 | 0.090 | 0.043 | 0.081 | 0.151 | 0.197 |
| `qwen3.8-27b-awq` | 0.145 | 0.039 | 0.082 | 0.056 | 0.081 | 0.076 | 0.040 | 0.067 | 0.152 | 0.224 |
| `qwen3.6-35b-a3b-int4-palmfuture` | 0.275 | 0.099 | 0.066 | 0.033 | 0.073 | 0.091 | 0.027 | 0.054 | 0.187 | 0.206 |
| `qwen3.8-27b-int4-redhat` | 0.244 | 0.059 | 0.061 | 0.080 | 0.065 | 0.076 | 0.032 | 0.048 | 0.163 | 0.201 |
| `qwen3.5-35b-a3b-int4` | 0.216 | 0.099 | 0.099 | 0.041 | 0.063 | 0.130 | 0.047 | 0.058 | 0.139 | 0.244 |
| `qwen3.5-9b-bf16` | 0.118 | 0.131 | 0.067 | 0.170 | 0.185 | 0.132 | 0.041 | 0.041 | 0.112 | 0.168 |
| `qwen3.5-9b-int8` | 0.110 | 0.116 | 0.067 | 0.165 | 0.184 | 0.120 | 0.043 | 0.035 | 0.111 | 0.156 |
| `qwen3-32b-int4` | 0.366 | 0.306 | 0.334 | 0.093 | 0.129 | 0.199 | 0.094 | 0.103 | 0.341 | 0.387 |
| `qwen2.5-32b-int4` | 0.572 | 0.536 | 0.469 | 0.171 | 0.165 | 0.236 | 0.143 | 0.156 | 0.394 | 0.502 |
| `qwen3-30b-a3b-2507-int4-redhat` | 0.511 | 0.609 | 0.481 | 0.221 | 0.160 | 0.258 | 0.103 | 0.133 | 0.399 | 0.441 |
| `qwen3-14b-int4` | 0.671 | 0.465 | 0.450 | 0.159 | 0.196 | 0.224 | 0.108 | 0.113 | 0.451 | 0.500 |
| **Jev** | 0.085 | 0.070 | 0.069 | 0.016 | 0.048 | 0.089 | 0.027 | 0.026 | 0.169 | 0.121 |

### Findings

1. The three Qwen3.6-27B builds lead. #1 (cyankiwi) and #2 (QuantTrio) are tied (Δ -0.0080 [-0.0201,
   +0.0038]). #1 beats every other preset: its Δ CI against each of them lies below zero. The five
   Qwen3.8-27B builds are within 0.010 of each other (0.6513 to 0.6613). The best Qwen3.6 build is
   0.0264 [0.0113, 0.0422] above the best Qwen3.8 build. Most of that gap is MMLU-Pro (0.50 to 0.53
   against 0.46 to 0.49), ARC (0.89 to 0.90 against 0.85 to 0.87) and IDK answers on GPQA. The
   comparison only partly controls for the quantizer. Groxaxo is the one quantizer with a build of
   both generations, and there Qwen3.6 − Qwen3.8 = +0.0110 [-0.0045, +0.0264].
2. The gap to Jev is on knowledge and reasoning. Against Jev, #1 is -0.28 on GPQA, -0.35 on LEXam,
   -0.35 on MMLU-Pro and -0.08 on ARC, all with McNemar p < 1e-4. On the classification benchmarks
   AG News, banking77, CLINC150, BoolQ and SST-5 the paired Δ CI includes 0 (banking77 +0.003, SST-5
   0.000). Yelp is the exception, at -0.070 [-0.123, -0.017]. On banking77 and SST-5, #1's NLL is
   lower than Jev's, partly because Jev rounds to 0.01 and these runs do not (`round_to` None).
3. The Qwen3.8 builds say "I don't know" more often. Their IDK rate on GPQA is 8% to 24%, and
   RedHat's is also 11% on LEXam. The Qwen3.6-27B builds are at 5% to 10% on GPQA; Jev is at 3%
   (GPQA) and 2% (LEXam). IDK answers count as wrong, so this explains part of Qwen3.8's GPQA
   deficit.
4. The dense 27B builds answer 2.5 to 2.9 items/s on the screen, and the two intent benchmarks take
   two thirds of that time. The mixture-of-experts (MoE) builds with the Qwen3.5 architecture
   (35B-A3B) are about 5× faster, at 13.3 to 13.8 items/s.
   - `qwen3.6-35b-a3b-int4-palmfuture` gives up 0.0334 [0.0181, 0.0488] against #1 and ties the
     Qwen3.8-27B builds (#4 − #8 +0.0070 [-0.0094, +0.0234]). It is the throughput pick if
     losing about 3 points of macro accuracy is acceptable.
   - `qwen3-30b-a3b-2507-int4-redhat` is the fastest, at 44.6 items/s, but 0.091 behind #1. It is an
     attention-only Qwen3 model without the hybrid's 784-token cache blocks, so the trie's queries
     share their prompt: 96% of its banking77 prompt tokens were cache hits, against 0% on the
     Qwen3.6-27B hybrid.
   - The 9B presets run at 10 items/s and are 0.078 behind. Their int8 build matches bf16
     (+0.0010 [-0.0027, +0.0047]).
5. Models outside the Qwen3.5 architecture are badly overconfident without calibration. Qwen3-32B,
   Qwen2.5-32B, Qwen3-30B-A3B-2507 and Qwen3-14B have a macro NLL of 1.54 to 3.05, against 0.86 to
   1.04 for the Qwen3.5-architecture models, and ECE up to 0.67. They also rank last on accuracy.

### Stage 2 recommendation (full `select` half)

- Take `qwen3.6-27b-int4-cyankiwi` and `qwen3.6-27b-int4-quanttrio`, #1 and #2, which are tied.
  QuantTrio has the lowest macro NLL of all 16 (0.864).
- The CIs leave the third slot open. #3 − #4 is +0.0107 [-0.0043, +0.0254], and #3 also ties #5
  and #7. Both #3 (-0.0157 [-0.0288, -0.0033]) and #4 are below #1. Since the screen cannot separate
  them, take `qwen3.8-27b-int4-palmfuture` (#4) for diversity. It adds a second model generation and
  a third quantizer (GPTQModel, group 32), where #1 to #3 are all Qwen3.6-27B. It also tests the
  Qwen3.6-over-Qwen3.8 finding on 7× the items. The alternative is `qwen3.6-27b-int4-groxaxo` (#3).
  It has the higher point estimate and a lower NLL than #4 (Δ NLL -0.041 [-0.072, -0.011]), but its
  base model is already covered. Run it as a fourth candidate if the budget allows.
- At the screen's measured prefill rate, one dense-27B preset on all of `select` (19,841 items,
  about 8.9 M prompt tokens under `auto_noecho`) takes about 1.9 h plus startup [INFERENCE:
  projected from the per-item token counts above; screen projections were within 8% of measured].
  Three presets take about 5.8 h, four about 7.8 h. The screen run directories resume into the same
  `runs/select/<preset>.auto_noecho/` with `--split select`.
- This stage used 3 h 45 min of GPU time: the cost check (23 min), a 1-min smoke test of the sweep
  driver on `qwen3-0.6b`, and the 16-preset sweep (3 h 21 min: 30 min of startups and 2 h 50 min of
  runs).

## Stage 2: layout and full `select` half

*Multi-call history (before one-call scoring): `auto_noecho` and the old prefill rendering.*

Stage 2 has two steps:

- 2a runs the three Stage 2 presets with the other renderer layout, `state_first`, on the same
  `screen` items, and pairs each with its `question_first` screen run.
- 2b runs each preset on all 19,841 `select` items with its better layout (`question_first` if the
  two tie).

Everything else is as in Stage 1: the `auto_noecho` policy, 16 items in flight, diagnostics on,
one fresh server per preset. The policy now lives in the library
(`jevemu.scoring.NoEchoAutoStrategy`, same manifest id `auto_noecho_tau0.001`), and every
Stage 2 run used it.

```bash
uv run python scripts/sweep_presets.py run --presets P1,P2,... --split screen --out runs/select \
    --strategy auto_noecho --renderer state_first    # -> runs/select/<preset>.auto_noecho.state_first/
uv run python scripts/run_split.py compare runs/select/P.auto_noecho.state_first \
    runs/select/P.auto_noecho --split screen         # paired, state_first - question_first
```

### Stage 2a: layout (screen)

The two layouts put the same text in a different order ([design.md](../implementation/design.md), "Renderer"):

- `question_first` (`question_first-6f75027397a3`) has a system message with the instructions,
  question and options, then a user message with the state.
- `state_first` (`state_first-5d29289f8b87`) has a single user message with the state, then
  the question, the options and "Respond with the letter only."

Both end with the `Answer:` prefill. On the four knowledge benchmarks (GPQA, LEXam, MMLU-Pro,
ARC) the state is the question stem and the benchmark's "question" is the option list. So
`question_first` shows the options before the stem, while `state_first` uses the usual
stem-then-options order. On the six classification benchmarks the state is the text to
classify, and the question is a fixed instruction.

As a render check, `/tokenize` rendered the first MMLU-Pro screen item on
`qwen3.6-27b-int4-cyankiwi` (Qwen3.6 template) and `qwen3.8-27b-int4-palmfuture` (Qwen3.8 template).
On both, `state_first` gives only user and assistant turns. No system message is injected: no
default system prompt and no Qwen3.8 reasoning-effort line. The generation prompt ends in the
thinking-off `<think>\n\n</think>\n\n`, then `Answer:`. So the Qwen2.5-style default-system-prompt
caveat (candidates.md, finding 6) does not apply to these templates. The item is 222 tokens under
`state_first` and 238 under `question_first`. `qwen3.6-27b-int4-quanttrio` and the 35B-A3B build
were not rendered separately [INFERENCE: same Qwen3.6 chat template].

The screen sweep ran from 07:26 to 08:22 UTC on Sep 25, 2026, with 0 item errors; every preset
finished on its first attempt. `qwen3.6-35b-a3b-int4-palmfuture`, the Stage 1 throughput pick,
was added from 08:23 to 08:29 so the speed alternative is measured with the same layout. All
rows are paired on the 2,799 screen items, with Δ = `state_first` (sf) − `question_first` (qf).
McNemar pools the discordant items of all ten benchmarks.

| Preset | Acc. `question_first` | Acc. `state_first` | Δ macro acc. [95% CI] | McNemar sf/qf, p | NLL qf | NLL sf | Δ macro NLL [95% CI] | Δ macro Brier [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | 0.6877 | 0.7341 | +0.0464 [+0.0303, +0.0617] | 281/150, 2.8e-10 | 0.874 | 0.757 | -0.117 [-0.147, -0.085] | -0.058 [-0.072, -0.044] |
| `qwen3.6-27b-int4-quanttrio` | 0.6797 | 0.7391 | +0.0594 [+0.0427, +0.0761] | 294/138, 4.7e-14 | 0.864 | 0.763 | -0.101 [-0.132, -0.071] | -0.055 [-0.069, -0.041] |
| `qwen3.8-27b-int4-palmfuture` | 0.6613 | 0.7208 | +0.0595 [+0.0424, +0.0762] | 292/144, 1.2e-12 | 0.942 | 0.795 | -0.147 [-0.177, -0.117] | -0.064 [-0.077, -0.051] |
| `qwen3.6-35b-a3b-int4-palmfuture` | 0.6543 | 0.7131 | +0.0588 [+0.0417, +0.0762] | 327/181, 9.3e-11 | 0.996 | 0.882 | -0.114 [-0.152, -0.076] | -0.053 [-0.070, -0.036] |

Δ accuracy per benchmark (`state_first` − `question_first`). Bold marks a paired CI that
excludes 0:

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | +0.040 | **+0.287** | **+0.110** | **+0.087** | +0.013 | -0.023 | -0.003 | +0.000 | -0.033 | -0.013 |
| `qwen3.6-27b-int4-quanttrio` | +0.111 | **+0.307** | **+0.127** | **+0.083** | +0.010 | -0.023 | -0.010 | +0.000 | -0.023 | +0.013 |
| `qwen3.8-27b-int4-palmfuture` | **+0.152** | **+0.267** | **+0.127** | **+0.097** | -0.013 | -0.017 | +0.003 | -0.003 | +0.003 | -0.020 |
| `qwen3.6-35b-a3b-int4-palmfuture` | **+0.152** | **+0.277** | **+0.107** | **+0.120** | -0.007 | -0.027 | -0.010 | -0.027 | -0.013 | +0.017 |

Δ NLL per benchmark (bold: CI excludes 0):

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-cyankiwi` | **-0.140** | **-0.449** | **-0.381** | **-0.235** | +0.057 | -0.041 | +0.008 | -0.001 | +0.011 | +0.007 |
| `qwen3.6-27b-int4-quanttrio` | **-0.217** | **-0.509** | **-0.344** | **-0.211** | **+0.128** | -0.029 | +0.018 | +0.046 | +0.063 | +0.039 |
| `qwen3.8-27b-int4-palmfuture` | **-0.346** | **-0.553** | **-0.330** | **-0.237** | **+0.101** | -0.115 | -0.065 | +0.023 | +0.004 | +0.043 |
| `qwen3.6-35b-a3b-int4-palmfuture` | **-0.248** | **-0.465** | **-0.373** | **-0.266** | +0.004 | -0.002 | +0.035 | **+0.078** | -0.004 | +0.099 |

The screen ranking under `state_first` (Δ vs #1 is paired with `qwen3.6-27b-int4-cyankiwi`
under `state_first`, and Δ vs Jev with Jev's screen answers):

| Preset | Macro acc. [95% CI] | Macro NLL [95% CI] | Δ acc. vs cyankiwi [95% CI] | Δ acc. vs Jev [95% CI] | Δ NLL vs Jev [95% CI] | Items/s sf (qf) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | 0.7391 [0.7227, 0.7556] | 0.763 [0.714, 0.812] | +0.0051 [-0.0054, +0.0158] | -0.0652 [-0.0826, -0.0478] | +0.032 [-0.017, +0.079] | 2.92 (2.81) |
| `qwen3.6-27b-int4-cyankiwi` | 0.7341 [0.7177, 0.7505] | 0.757 [0.709, 0.806] | — | -0.0702 [-0.0867, -0.0539] | +0.027 [-0.023, +0.075] | 2.91 (2.87) |
| `qwen3.8-27b-int4-palmfuture` | 0.7208 [0.7037, 0.7375] | 0.795 [0.748, 0.843] | -0.0133 [-0.0274, +0.0004] | -0.0835 [-0.1006, -0.0672] | +0.064 [+0.015, +0.112] | 2.67 (2.86) |
| `qwen3.6-35b-a3b-int4-palmfuture` | 0.7131 [0.6964, 0.7302] | 0.882 [0.829, 0.935] | -0.0210 [-0.0371, -0.0046] | -0.0912 [-0.1083, -0.0741] | +0.152 [+0.097, +0.204] | 11.81 (13.34) |

Findings:

1. `state_first` is better for every preset, by 0.046 to 0.060 macro accuracy, with all four CIs
   above 0 and McNemar p ≤ 3e-10. NLL falls by 0.10 to 0.15 and Brier by 0.05 to 0.06. The two
   Qwen3.6-27B builds swap places, but they remain tied.
2. All of the gain is on the knowledge benchmarks: LEXam +0.27 to +0.31, MMLU-Pro +0.11 to +0.13,
   ARC +0.08 to +0.12 and GPQA +0.04 to +0.15. On the six classification benchmarks no Δ accuracy CI
   excludes 0 (-0.033 to +0.017). With accuracy unchanged, NLL gets worse on AG News for QuantTrio
   and Qwen3.8 (+0.10 to +0.13) and on BoolQ for the MoE (+0.08). [INFERENCE: what helps is the
   stem-then-options order on the knowledge items; the classification prompts gain nothing from the
   reordering.]
3. There are fewer "I don't know" answers. The GPQA IDK rate falls from between 5% and 20% to
   between 1% and 3% (Qwen3.8 palmfuture: 0.202 to 0.030), and on LEXam it is 0.3% to 0.7% for every
   preset.
4. The gap to Jev shrinks by 4.6 to 6.0 points. For cyankiwi it goes from -0.117 to -0.070 [-0.087,
   -0.054], and macro NLL is now tied with Jev (+0.027 [-0.023, +0.075]). The Qwen3.6-27B builds
   reach 0.977 to 0.980 on ARC, against Jev's 0.970.
5. Throughput does not change with the dense 27B builds (2.67 to 2.92 items/s under either layout)
   [INFERENCE: the hybrid models cache prefixes only in 784-token blocks, so neither layout's shared
   prefix is reused at these prompt lengths]. The MoE build is 11% slower under `state_first` (11.8
   against 13.3 items/s).

So all three presets ran `select` with `state_first`.

### Stage 2b: finalists on `select`

Each preset answered all 19,841 `select` items with `state_first`, writing into the same
`runs/select/<preset>.auto_noecho.state_first/` directory as its 2a screen run. The runs went
from 08:29 to 13:59 UTC on Sep 25, 2026, one preset at a time. All three finished on the first
attempt with 0 item errors. The tables are `sweep_presets.py report` (saved in
`runs/select/sweep/report.select.md`); the paired rows are `run_split.py compare`.

```bash
uv run python scripts/sweep_presets.py run --split select --out runs/select --strategy auto_noecho \
    --renderer state_first \
    --presets qwen3.6-27b-int4-cyankiwi,qwen3.6-27b-int4-quanttrio,qwen3.8-27b-int4-palmfuture
uv run python scripts/sweep_presets.py report --split select --out runs/select --strategy auto_noecho \
    --renderer state_first --reference runs/select/jev --presets ...
```

| Rank | Preset | Macro acc. [95% CI] | Macro NLL [95% CI] | Macro Brier [95% CI] | Δ acc. vs #1 [95% CI] | Δ acc. vs Jev [95% CI] | Δ NLL vs Jev [95% CI] | Items/s | Startup + run, min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `qwen3.6-27b-int4-quanttrio` | 0.7357 [0.7235, 0.7479] | 0.804 [0.776, 0.832] | 0.374 [0.361, 0.386] | — | -0.0598 [-0.0740, -0.0456] | +0.053 [+0.024, +0.082] | 3.13 | 2.1 + 105.7 |
| 2 | `qwen3.6-27b-int4-cyankiwi` | 0.7285 [0.7167, 0.7403] | 0.792 [0.765, 0.820] | 0.371 [0.359, 0.383] | -0.0072 [-0.0160, +0.0014] | -0.0670 [-0.0800, -0.0538] | +0.042 [+0.013, +0.071] | 3.18 | 2.2 + 104.3 |
| 3 | `qwen3.8-27b-int4-palmfuture` | 0.7180 [0.7055, 0.7303] | 0.816 [0.793, 0.840] | 0.384 [0.373, 0.394] | -0.0177 [-0.0290, -0.0065] | -0.0775 [-0.0912, -0.0641] | +0.065 [+0.037, +0.093] | 2.92 | 2.1 + 113.2 |
| | **Jev** `jev-1.13.0` | 0.7955 [0.7837, 0.8069] | 0.751 [0.715, 0.788] | 0.301 [0.288, 0.314] | | | | | |

Paired between the finalists. The macro Δ weighs every benchmark equally. McNemar pools the
discordant items of all ten benchmarks, so it weighs every item equally, and MMLU-Pro is 30% of
the items:

| a − b | Δ macro acc. [95% CI] | McNemar a/b, p | Δ macro NLL [95% CI] | Δ macro Brier [95% CI] |
| --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` − `qwen3.6-27b-int4-cyankiwi` | +0.0072 [-0.0014, +0.0160] | 418/431, 0.68 | +0.011 [+0.002, +0.021] | +0.003 [-0.002, +0.007] |
| `qwen3.6-27b-int4-quanttrio` − `qwen3.8-27b-int4-palmfuture` | +0.0177 [+0.0065, +0.0290] | 1156/715, 1.6e-24 | -0.012 [-0.027, +0.003] | -0.010 [-0.018, -0.003] |
| `qwen3.6-27b-int4-cyankiwi` − `qwen3.8-27b-int4-palmfuture` | +0.0105 [-0.0010, +0.0220] | 1133/679, 1.0e-26 | -0.024 [-0.040, -0.008] | -0.013 [-0.020, -0.005] |

Accuracy per benchmark. The IDK columns give the share of "I don't know" answers, which count
as wrong:

| Preset | GPQA | IDK | LEXam | IDK | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | 0.465 | 0.010 | 0.671 | 0.003 | 0.628 | 0.971 | 0.886 | 0.795 | 0.900 | 0.887 | 0.566 | 0.588 |
| `qwen3.6-27b-int4-cyankiwi` | 0.384 | 0.020 | 0.690 | 0.003 | 0.634 | 0.971 | 0.884 | 0.788 | 0.897 | 0.888 | 0.553 | 0.596 |
| `qwen3.8-27b-int4-palmfuture` | 0.444 | 0.030 | 0.623 | 0.003 | 0.586 | 0.956 | 0.860 | 0.787 | 0.898 | 0.874 | 0.579 | 0.574 |
| **Jev** | 0.646 | 0.030 | 0.745 | 0.016 | 0.826 | 0.974 | 0.884 | 0.804 | 0.924 | 0.908 | 0.576 | 0.666 |

Δ accuracy vs Jev per benchmark (bold: the paired CI excludes 0):

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | **-0.182** | **-0.074** | **-0.198** | -0.003 | +0.002 | -0.008 | **-0.024** | **-0.021** | -0.010 | **-0.078** |
| `qwen3.6-27b-int4-cyankiwi` | **-0.263** | **-0.055** | **-0.192** | -0.003 | +0.000 | **-0.016** | **-0.027** | **-0.020** | -0.023 | **-0.071** |
| `qwen3.8-27b-int4-palmfuture` | **-0.202** | **-0.123** | **-0.241** | **-0.019** | **-0.024** | **-0.017** | **-0.026** | **-0.034** | +0.004 | **-0.093** |

Yelp is dropped. It shows its levels as `0) 1 star` ... `4) 5 stars`, a defect in our prompt: the
models also write `5`, which is not a label, and `4` is ambiguous. Jev receives the Score question
natively and is not affected. Without Yelp (9 benchmarks, 17,341 items; `run_split.py compare` now
skips it by default):

- QuantTrio 0.7521, Jev 0.8098 (one-call rerun: 0.7520, in the [Summary](#summary)).
- QuantTrio − Jev: accuracy -0.0577 [-0.0734, -0.0420] (with Yelp: -0.0598), NLL +0.032 [+0.001,
  +0.063] (with Yelp: +0.053), Brier +0.065 [+0.053, +0.078].

The tables in this section include Yelp as run.

NLL per benchmark:

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | 1.262 | 0.833 | 1.137 | 0.122 | 0.587 | 1.000 | 0.398 | 0.332 | 1.180 | 1.184 |
| `qwen3.6-27b-int4-cyankiwi` | 1.296 | 0.821 | 1.126 | 0.122 | 0.553 | 1.032 | 0.403 | 0.327 | 1.096 | 1.145 |
| `qwen3.8-27b-int4-palmfuture` | 1.205 | 0.914 | 1.240 | 0.154 | 0.644 | 0.984 | 0.430 | 0.339 | 1.108 | 1.140 |
| **Jev** | 0.978 | 0.803 | 0.743 | 0.153 | 0.665 | 1.158 | 0.430 | 0.257 | 1.378 | 0.942 |

ECE per benchmark:

| Preset | GPQA | LEXam | MMLU-Pro | ARC | AG News | banking77 | CLINC150 | BoolQ | SST-5 | Yelp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | 0.138 | 0.069 | 0.052 | 0.014 | 0.079 | 0.090 | 0.018 | 0.038 | 0.175 | 0.244 |
| `qwen3.6-27b-int4-cyankiwi` | 0.195 | 0.072 | 0.062 | 0.013 | 0.074 | 0.094 | 0.024 | 0.041 | 0.154 | 0.216 |
| `qwen3.8-27b-int4-palmfuture` | 0.150 | 0.079 | 0.032 | 0.020 | 0.090 | 0.080 | 0.024 | 0.028 | 0.114 | 0.229 |
| **Jev** | 0.085 | 0.064 | 0.049 | 0.011 | 0.069 | 0.088 | 0.020 | 0.019 | 0.184 | 0.134 |

Findings:

1. The two Qwen3.6-27B builds are tied; QuantTrio's lead comes from GPQA. QuantTrio is #1 by 0.0072
   [-0.0014, +0.0160]. That lead is GPQA's 99 items (+0.081, 12/4 discordant), which count for a
   tenth of the macro. On the other nine benchmarks the mean is 0.7658 for QuantTrio and 0.7668 for
   cyankiwi. Item-weighted accuracy is 0.7442 against 0.7449 (McNemar 418/431, p = 0.68). Cyankiwi
   has the lower NLL (by 0.011 [0.002, 0.021]) and is 1.6% faster. The order matches the
   `state_first` screen, where QuantTrio led by +0.0051 [-0.0054, +0.0158]. The screen items are a
   subset of `select`, so this is not an independent replication.
2. Qwen3.6 beats Qwen3.8 again, on 7× the items. `qwen3.8-27b-int4-palmfuture` is below QuantTrio
   (-0.0177 [-0.0290, -0.0065]). Against cyankiwi the macro Δ crosses 0 (-0.0105 [-0.0220,
   +0.0010]), but it loses 1,133 to 679 discordant items (p = 1e-26). The difference is mostly
   MMLU-Pro (0.586 against 0.628 to 0.634), LEXam (0.623 against 0.671 to 0.690) and AG News.
3. The screen predicted `select`. The `state_first` screen macros were 0.7391, 0.7341 and 0.7208. On
   `select` they are 0.7357, 0.7285 and 0.7180: the same order, within 0.6 points.
4. For #1 the gap to Jev is -0.060 [-0.074, -0.046], mostly on the knowledge benchmarks. The largest
   parts are MMLU-Pro (-0.198) and GPQA (-0.18), then Yelp (-0.078) and LEXam (-0.074). ARC, AG
   News, banking77 and SST-5 are tied with Jev; CLINC150 and BoolQ are 0.02 behind. Macro NLL is
   0.053 [0.024, 0.082] above Jev's. The emulator's NLL is below Jev's on AG News, banking77 and
   SST-5, partly because Jev rounds to 0.01. On the ordinal benchmarks the emulator's MAE is 0.48 to
   0.49 on SST-5 and 0.43 to 0.46 on Yelp (Jev: 0.470 and 0.387). It is within one level on 95% to
   96% of items (Jev: 97%).
5. Without calibration, the mean per-benchmark ECE is 0.084 to 0.095, against Jev's 0.073. The
   excess is on GPQA (0.14 to 0.20 against 0.085) and Yelp (0.22 to 0.24 against 0.134). The holdout
   calibration step targets this.
6. A dense Qwen3.6-27B `select` run takes 1 h 44 min to 1 h 46 min plus 2.1 min of startup (3.13 to
   3.18 items/s). Qwen3.8 takes 1 h 53 min (2.92 items/s). The intent benchmarks still dominate:
   banking77 and CLINC150 took 3,943 of cyankiwi's 6,255 s. The projection of about 1.9 h per preset
   held (1.7 to 1.9 h).

### Recommendation

- For the best accuracy, use `qwen3.6-27b-int4-quanttrio` with the `state_first` layout. It was
  chosen with `auto_noecho` scoring (τ = 1e-3) and now runs with one-call scoring
  ([above](#scoring-now-one-call-per-question)). In code:
  `Emulator(backend, renderer="state_first", strategy=SingleCallStrategy())`.
  - In Stage 2 (multi-call, 10 benchmarks with Yelp) on `select`: macro accuracy 0.7357
    [0.7235, 0.7479], NLL 0.804, 3.13 items/s. It ranked #1 on both the `state_first` screen
    and `select`.
  - With one-call scoring on `select` (9 benchmarks): 0.7520 [0.7385, 0.7653], NLL 0.780,
    4.44 items/s.
  - In Stage 2 it was statistically tied with `qwen3.6-27b-int4-cyankiwi` (0.7285), the
    equal-accuracy alternative. Cyankiwi had a lower NLL, was 1.6% faster, and runs on the
    standard 27B launch (0.95 of the GPU, `--max-num-batched-tokens=1024`); QuantTrio needs 0.97
    and 512 (candidates.md). [INFERENCE: calibration fitted on `holdout` should absorb the 0.011
    NLL difference; if it does not, or if memory headroom matters, cyankiwi is the pick.] With
    one-call scoring the two are tied on `holdout` (-0.0014 [-0.0101, +0.0069]), with calibrated
    NLL 0.691 against 0.688 ([calibration.md](calibration.md#current-results-one-call-per-question-holdout)).
- The layout matters more than the quantizer. `state_first` added 0.046 to 0.060 macro accuracy on
  every preset tested, while in Stage 2 the two Qwen3.6-27B builds differed by 0.007.
  `PromptRenderer()` and `Emulator(renderer="default")` now default to `state_first`
  (`jevemu.render.DEFAULT_LAYOUT`). Stage 1 ran with the earlier `question_first` default, which the
  commands above now pass explicitly. The emulator looks up calibrators by `template_id`, so fit
  them on `state_first` output (`state_first-5d29289f8b87`).
- For the best speed/accuracy trade, use `qwen3.6-35b-a3b-int4-palmfuture` with `state_first`
  (chosen with `auto_noecho`).
  - In Stage 1 (`question_first` screen) it scored 0.6543 at 13.34 items/s, 0.0334 behind #1.
  - With `state_first` on the screen it scores 0.7131 [0.6964, 0.7302] at 11.81 items/s, 4.0× the
    dense 27B builds' 2.91 to 2.92 items/s on the same items. That is 0.0260 [0.0097, 0.0418] below
    QuantTrio and tied with the Qwen3.8-27B finalist (-0.0077 [-0.0234, +0.0085]).
  - Its NLL is 0.119 [0.086, 0.153] higher than QuantTrio's. It was not run on `select`.
  - `qwen3-30b-a3b-2507-int4-redhat` (44.6 items/s in Stage 1) is 0.091 behind #1 and badly
    overconfident, so it is not a candidate.

### GPU time for Stage 2: 6 h 32 min

The GPU ran from 07:26:42 to 13:59:05 UTC on Sep 25, 2026:

- the 2a layout sweep, 56 min: three startups and three screen runs of 16.0 to 17.5 min;
- the 35B-A3B screen, 6 min;
- the 2b `select` sweep, 5 h 30 min: 6.5 min of startups and 5 h 23 min of runs.

The `/tokenize` render checks and a 4-item smoke test of `run_split.py` with the PriDe debiaser
(`--debiaser pride`) ran against servers that were already up for the sweep.

## Cost and token statistics

*Multi-call history (before one-call scoring): the tables are `auto_noecho` runs, and their GPU
$ use the old 280 W placeholder ($0.0756 per GPU-hour). Multiply those GPU $ by 1.464 for the
current 410 W estimate.*

`scripts/run_split.py stats` reads the records and manifests of one or more run directories and
reports, per system and per benchmark, the tokens, latency, throughput and cost of the answers
(`jevemu.bench.compare.stats`; prices and per-item costs in `jevemu.eval.costs`). All tables in
this section are its output.

```bash
uv run python scripts/run_split.py stats runs/select/jev \
    runs/select/qwen3.6-27b-int4-quanttrio.auto_noecho.state_first \
    runs/select/qwen3.6-27b-int4-cyankiwi.auto_noecho.state_first \
    runs/select/qwen3.8-27b-int4-palmfuture.auto_noecho.state_first --split select
# GPU $ = GPU-hours x electricity estimate: --gpu-watts (default 410) x
# --usd-per-wh (default 0.00027 = NJ average); --gpu-usd-per-hour RATE overrides both
```

How the numbers are computed:

- Jev is priced per token: $0.042 per 1M input tokens, output free
  ([docs.typesafe.ai/models](https://docs.typesafe.ai/models), as of Sep 25, 2026). An item's cost
  is the charge the client recorded for it. Response-cache hits (recorded as $0) are costed at the
  list price, so a cached run does not look cheaper than a fresh one.
  - On `select` the recorded spend is $0.5094 for 19,802 billed items. The price table gives
    $0.5094 for the same items; no item differs by more than 1%.
  - The 39 cache hits add $0.0010 at list price, for $0.5104 in total.
- The emulator is priced by GPU time on the local GPU (one RTX 3090).
  - A benchmark's GPU time is the start-to-end wall time of the runner invocation that wrote it,
    from the manifest. This excludes vLLM server startup (1.2 to 2.2 min per preset). It also
    assumes nothing else used the GPU during the run. The sweeps ran one preset at a time; only the
    brief checks listed under "GPU time for Stage 2" shared a server.
  - Each benchmark's GPU time is split over its items in proportion to input + output tokens, so
    the item costs add up to the run's total. This was checked on every emulator run below.
  - GPU $ is an electricity estimate, GPU-hours × average board power × electricity price: 410 W
    (the RTX 3090's draw while running, observed by the user; not metered per run) and $0.00027/Wh
    ($0.27/kWh, the New Jersey average). Together that is $0.1107 per GPU-hour. The tables below
    still use the old 280 W placeholder ($0.0756 per GPU-hour). The estimate covers only the GPU's
    electricity: no hardware amortization, rest of the machine or cooling. Change it with
    `--gpu-watts`/`--usd-per-wh` (or `JEVEMU_GPU_WATTS`/`JEVEMU_USD_PER_WH`); `--gpu-usd-per-hour`
    sets the rate directly.
- Tokens are each system's own count.
  - Jev's input tokens are the tokens it bills. Its output tokens are not billed and grow with the
    number of options: 17 per SST-5/Yelp item, 20 for BoolQ, 45 to 52 for 4 or 5 options, 828 for
    banking77's 77, and 1,294 for CLINC150's 150.
  - The emulator's input tokens are the prompt tokens of all its backend calls for the item. Its
    output tokens are the generated ones, one per call.
  - Calls/item is above 1 only where the trie scores more than 32 options (banking77 and
    CLINC150).
  - Cached tok/item counts vLLM prefix-cache hits. It is 0 for every Qwen3.6, Qwen3.8 and
    Qwen3.5-35B-A3B preset, 151 per item for Qwen3.5-9B, and 244 to 285 for the Qwen3 and Qwen2.5
    models.
- Latency is per item with 16 items in flight. For Jev it is the HTTP round trip. For the emulator
  it is the end-to-end time of the call, while 15 other items share the server; its p95 comes from
  banking77 and CLINC150.
- Throughput (q/s, questions per second) is the items the runner sent to the system divided by the
  summed start-to-end wall time of the invocations that sent them (from the manifest; vLLM server
  startup excluded). It is given per benchmark, and pooled as total items over total seconds. "In
  flight" is the runner's concurrency (16 for every run here). An invocation served entirely from
  the response cache measures nothing and is left out, so Jev's `screen` row has no q/s. Jev's q/s
  measures our client's rate limit and says nothing about Jev's capacity (footnote †).
- $ / 1k correct is the cost per correct answer × 1,000. It is pooled over items, while the macro
  accuracy weighs every benchmark equally.

### `select`: Jev and the Stage 2b finalists

| System | Price | Items | Macro acc. | In tok/item | Out tok/item | Cached tok/item | Calls/item | p50 ms | p95 ms | q/s | In flight | GPU-h | Cost $ | $ / 1k items | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Jev** `jev-1.13.0` | jev-1.13.0 | 19,841 | 0.7955 | 612.5 | 252.5 |  |  | 135 | 191 | 19.65 † | 16 |  | 0.5104 | 0.0257 | 0.0313 |
| `qwen3.6-27b-int4-quanttrio` | local-gpu | 19,841 | 0.7357 | 430.0 | 1.3 | 0.0 | 1.28 | 2343 | 19831 | 3.13 | 16 | 1.760 | 0.1330 | 0.0067 | 0.0090 |
| `qwen3.6-27b-int4-cyankiwi` | local-gpu | 19,841 | 0.7285 | 421.5 | 1.3 | 0.0 | 1.26 | 2365 | 19081 | 3.18 | 16 | 1.736 | 0.1312 | 0.0066 | 0.0089 |
| `qwen3.8-27b-int4-palmfuture` | local-gpu | 19,841 | 0.7180 | 464.1 | 1.3 | 0.0 | 1.34 | 2301 | 25471 | 2.92 | 16 | 1.885 | 0.1425 | 0.0072 | 0.0099 |

† Jev's q/s measures our client's rate limit and says nothing about Jev's compute. The runs sent one
question per request, and JevClient caps request starts at max_rpm = 1,200/min (20 requests/s),
Jev's documented limit (1,200 requests/min and 250,000 tokens/s). Jev evaluates the questions of one
request in parallel, so with N questions per request its ceiling is about 20·N q/s, subject to
250,000 tokens/s (about 500 q/s at 500 tokens per question). This is inferred from the docs and was
not measured.

- The Qwen3.6-27B builds need 1.74 to 1.76 GPU-hours for the 19,841 items, which is 0.087 to 0.089
  GPU-h per 1,000 items. Qwen3.8 needs 1.89 (0.095 per 1,000).
- QuantTrio matches Jev's list price per item ($0.0257 per 1,000) at $0.29 per GPU-hour. Per correct
  answer it breaks even at $0.26, because it gets fewer items right (14,766 against Jev's 16,302).
  Cyankiwi breaks even at $0.29 per item and $0.27 per correct answer; Qwen3.8 at $0.27 and $0.24.
  These figures hold for 16 items in flight on one RTX 3090. [INFERENCE: more items in flight would
  lower the GPU time per item.]
- At 16 in flight on the RTX 3090 the dense 27B finalists answer 2.92 to 3.18 q/s pooled; QuantTrio
  ranges from 0.88 q/s (banking77, 3.07 calls per item) to 13.53 (SST-5). Jev's 19.65 q/s is the
  client's 20 requests/s cap at one question per request (†).
- At the electricity estimate ($0.1107 per GPU-hour, the table's GPU $ × 1.464), QuantTrio costs
  $0.0098 per 1,000 items against Jev's $0.0257: 38% of Jev's price per item and 42% per correct
  answer. Breaking even would need a 2.6× higher rate: about 1,070 W at the New Jersey price, or
  $0.71/kWh at 410 W. So on electricity alone the local emulator is cheaper; with hardware
  amortization it may not be.
- The intent benchmarks dominate both costs.
  - banking77 and CLINC150 are 19% of the items. They take 1.115 of QuantTrio's 1.760 GPU-hours
    (63%): the trie makes 2 or 3 calls per item, 1,370 to 1,560 prompt tokens in all.
  - For Jev the same two benchmarks cost $0.2000 of $0.5104 (39%). Their 77 and 150 options
    make the prompts long: 1,024 and 1,416 input tokens per item.

### Per benchmark

Jev, then `qwen3.6-27b-int4-quanttrio` (GPU $ at the old 280 W estimate). The last row of
each table pools every benchmark, with the macro accuracy:

| Benchmark | Items | Acc. | In tok/item | Out tok/item | In tok total | Out tok total | Cached tok/item | Calls/item | p50 ms | p95 ms | q/s | In flight | GPU-h | Cost $ | $ / 1k items | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.6465 | 595.1 | 52.0 | 58,911 | 5,148 |  |  | 149 | 857 | 16.47 † | 16 |  | 0.0025 | 0.0250 | 0.0387 |
| lexam_en_idk | 310 | 0.7452 | 515.3 | 52.0 | 159,742 | 16,120 |  |  | 133 | 181 | 20.68 † | 16 |  | 0.0067 | 0.0216 | 0.0290 |
| mmlu_pro | 6,016 | 0.8263 | 554.6 | 83.4 | 3,336,346 | 501,594 |  |  | 131 | 175 | 20.00 † | 16 |  | 0.1401 | 0.0233 | 0.0282 |
| arc_challenge | 586 | 0.9744 | 380.9 | 45.0 | 223,180 | 26,384 |  |  | 134 | 177 | 20.04 † | 16 |  | 0.0094 | 0.0160 | 0.0164 |
| ag_news | 3,800 | 0.8845 | 358.4 | 47.4 | 1,362,095 | 180,174 |  |  | 139 | 198 | 19.99 † | 16 |  | 0.0572 | 0.0151 | 0.0170 |
| banking77 | 1,540 | 0.8039 | 1,023.6 | 828.3 | 1,576,302 | 1,275,566 |  |  | 138 | 198 | 16.36 † | 16 |  | 0.0662 | 0.0430 | 0.0535 |
| clinc150 | 2,250 | 0.9240 | 1,416.0 | 1,293.5 | 3,186,048 | 2,910,326 |  |  | 139 | 208 | 19.97 † | 16 |  | 0.1338 | 0.0595 | 0.0644 |
| boolq | 1,635 | 0.9083 | 403.8 | 20.0 | 660,271 | 32,700 |  |  | 132 | 197 | 20.01 † | 16 |  | 0.0277 | 0.0170 | 0.0187 |
| sst5 | 1,105 | 0.5756 | 338.7 | 17.0 | 374,237 | 18,785 |  |  | 134 | 183 | 20.01 † | 16 |  | 0.0157 | 0.0142 | 0.0247 |
| yelp_stars | 2,500 | 0.6664 | 486.0 | 17.0 | 1,214,898 | 42,500 |  |  | 135 | 202 | 20.00 † | 16 |  | 0.0510 | 0.0204 | 0.0306 |
| **all (10), macro acc.** | 19,841 | 0.7955 | 612.5 | 252.5 | 12,152,030 | 5,009,297 |  |  | 135 | 191 | 19.65 † | 16 |  | 0.5104 | 0.0257 | 0.0313 |

† As above: the q/s measures our client's rate limit (20 requests/s at one question per request) and
says nothing about Jev's compute.

| Benchmark | Items | Acc. | In tok/item | Out tok/item | In tok total | Out tok total | Cached tok/item | Calls/item | p50 ms | p95 ms | q/s | In flight | GPU-h | Cost $ | $ / 1k items | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpqa_diamond_idk | 99 | 0.4646 | 316.9 | 1.0 | 31,369 | 99 | 0.0 | 1.00 | 3501 | 5073 | 4.19 | 16 | 0.007 | 0.0005 | 0.0050 | 0.0108 |
| lexam_en_idk | 310 | 0.6710 | 241.6 | 1.0 | 74,881 | 310 | 0.0 | 1.00 | 2875 | 3249 | 5.63 | 16 | 0.015 | 0.0012 | 0.0037 | 0.0056 |
| mmlu_pro | 6,016 | 0.6283 | 250.8 | 1.0 | 1,508,540 | 6,016 | 0.0 | 1.00 | 2807 | 4776 | 5.37 | 16 | 0.311 | 0.0235 | 0.0039 | 0.0062 |
| arc_challenge | 586 | 0.9710 | 109.8 | 1.0 | 64,325 | 586 | 0.0 | 1.00 | 1402 | 1640 | 11.27 | 16 | 0.014 | 0.0011 | 0.0019 | 0.0019 |
| ag_news | 3,800 | 0.8863 | 109.9 | 1.0 | 417,483 | 3,800 | 0.0 | 1.00 | 1372 | 1605 | 11.27 | 16 | 0.094 | 0.0071 | 0.0019 | 0.0021 |
| banking77 | 1,540 | 0.7955 | 1,557.0 | 3.1 | 2,397,839 | 4,729 | 0.0 | 3.07 | 15716 | 35550 | 0.88 | 16 | 0.489 | 0.0369 | 0.0240 | 0.0301 |
| clinc150 | 2,250 | 0.9000 | 1,366.9 | 2.0 | 3,075,560 | 4,532 | 0.0 | 2.01 | 4114 | 88503 | 1.00 | 16 | 0.626 | 0.0473 | 0.0210 | 0.0234 |
| boolq | 1,635 | 0.8869 | 163.9 | 1.0 | 267,977 | 1,635 | 0.0 | 1.00 | 2034 | 2383 | 7.94 | 16 | 0.057 | 0.0043 | 0.0026 | 0.0030 |
| sst5 | 1,105 | 0.5656 | 88.0 | 1.0 | 97,222 | 1,105 | 0.0 | 1.00 | 1137 | 1332 | 13.53 | 16 | 0.023 | 0.0017 | 0.0016 | 0.0027 |
| yelp_stars | 2,500 | 0.5880 | 238.9 | 1.0 | 597,305 | 2,500 | 0.0 | 1.00 | 2733 | 3601 | 5.61 | 16 | 0.124 | 0.0094 | 0.0037 | 0.0064 |
| **all (10), macro acc.** | 19,841 | 0.7357 | 430.0 | 1.3 | 8,532,501 | 25,312 | 0.0 | 1.28 | 2343 | 19831 | 3.13 | 16 | 1.760 | 0.1330 | 0.0067 | 0.0090 |

### Stage 1 and Stage 2a screens

The same statistics on the `screen` split (2,799 items), GPU $ at the old 280 W estimate
($0.0756 per GPU-hour; × 1.464 for 410 W). Rows are in Stage 1 rank order (`question_first`),
then the four Stage 2a `state_first` runs. Jev's screen answers came from its response cache
($0 spent) and are costed at list price.

| System | Price | Items | Macro acc. | In tok/item | Out tok/item | Cached tok/item | Calls/item | p50 ms | p95 ms | q/s | In flight | GPU-h | Cost $ | $ / 1k items | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Jev** `jev-1.13.0` | jev-1.13.0 | 2,799 | 0.8043 | 609.6 | 259.5 |  |  | 135 | 196 | — | — |  | 0.0717 | 0.0256 | 0.0314 |
| `qwen3.6-27b-int4-cyankiwi` | local-gpu | 2,799 | 0.6877 | 457.3 | 1.3 | 0.0 | 1.30 | 2501 | 21906 | 2.87 | 16 | 0.271 | 0.0205 | 0.0073 | 0.0103 |
| `qwen3.6-27b-int4-quanttrio` | local-gpu | 2,799 | 0.6797 | 476.4 | 1.3 | 0.0 | 1.33 | 2563 | 22084 | 2.81 | 16 | 0.277 | 0.0209 | 0.0075 | 0.0107 |
| `qwen3.6-27b-int4-groxaxo` | local-gpu | 2,799 | 0.6720 | 505.5 | 1.4 | 0.0 | 1.38 | 2559 | 27515 | 2.65 | 16 | 0.294 | 0.0222 | 0.0079 | 0.0114 |
| `qwen3.8-27b-int4-palmfuture` | local-gpu | 2,799 | 0.6613 | 464.1 | 1.3 | 0.0 | 1.31 | 2464 | 22845 | 2.86 | 16 | 0.272 | 0.0206 | 0.0074 | 0.0107 |
| `qwen3.8-27b-int4-groxaxo` | local-gpu | 2,799 | 0.6610 | 527.4 | 1.4 | 0.0 | 1.42 | 2547 | 29272 | 2.54 | 16 | 0.306 | 0.0231 | 0.0083 | 0.0121 |
| `qwen3.8-27b-int4-btbtyler09` | local-gpu | 2,799 | 0.6586 | 497.0 | 1.4 | 0.0 | 1.37 | 2479 | 26304 | 2.67 | 16 | 0.291 | 0.0220 | 0.0079 | 0.0115 |
| `qwen3.8-27b-awq` | local-gpu | 2,799 | 0.6583 | 500.1 | 1.4 | 0.0 | 1.37 | 2559 | 27044 | 2.62 | 16 | 0.297 | 0.0224 | 0.0080 | 0.0117 |
| `qwen3.6-35b-a3b-int4-palmfuture` | local-gpu | 2,799 | 0.6543 | 550.8 | 1.4 | 0.0 | 1.45 | 468 | 5957 | 13.34 | 16 | 0.058 | 0.0044 | 0.0016 | 0.0023 |
| `qwen3.8-27b-int4-redhat` | local-gpu | 2,799 | 0.6513 | 520.4 | 1.4 | 0.0 | 1.40 | 2553 | 29014 | 2.56 | 16 | 0.303 | 0.0229 | 0.0082 | 0.0121 |
| `qwen3.5-35b-a3b-int4` | local-gpu | 2,799 | 0.6416 | 529.2 | 1.4 | 0.0 | 1.41 | 470 | 5514 | 13.79 | 16 | 0.056 | 0.0043 | 0.0015 | 0.0023 |
| `qwen3.5-9b-bf16` | local-gpu | 2,799 | 0.6100 | 527.1 | 1.4 | 151.1 | 1.42 | 855 | 6890 | 10.15 | 16 | 0.077 | 0.0058 | 0.0021 | 0.0033 |
| `qwen3.5-9b-int8` | local-gpu | 2,799 | 0.6090 | 527.2 | 1.4 | 150.9 | 1.42 | 881 | 7295 | 9.80 | 16 | 0.079 | 0.0060 | 0.0021 | 0.0034 |
| `qwen3-32b-int4` | local-gpu | 2,799 | 0.6053 | 410.9 | 1.2 | 284.5 | 1.22 | 1535 | 4048 | 8.39 | 16 | 0.093 | 0.0070 | 0.0025 | 0.0040 |
| `qwen2.5-32b-int4` | local-gpu | 2,799 | 0.5986 | 365.7 | 1.2 | 243.6 | 1.16 | 1478 | 3603 | 8.70 | 16 | 0.089 | 0.0068 | 0.0024 | 0.0039 |
| `qwen3-30b-a3b-2507-int4-redhat` | local-gpu | 2,799 | 0.5970 | 375.6 | 1.2 | 253.6 | 1.18 | 247 | 622 | 44.62 | 16 | 0.017 | 0.0013 | 0.0005 | 0.0008 |
| `qwen3-14b-int4` | local-gpu | 2,799 | 0.5818 | 381.2 | 1.2 | 255.7 | 1.18 | 666 | 1551 | 18.57 | 16 | 0.042 | 0.0032 | 0.0011 | 0.0019 |
| `qwen3.6-27b-int4-quanttrio`, `state_first` | local-gpu | 2,799 | 0.7391 | 456.2 | 1.3 | 0.0 | 1.33 | 2422 | 23567 | 2.92 | 16 | 0.266 | 0.0201 | 0.0072 | 0.0095 |
| `qwen3.6-27b-int4-cyankiwi`, `state_first` | local-gpu | 2,799 | 0.7341 | 455.1 | 1.3 | 0.0 | 1.33 | 2310 | 23954 | 2.91 | 16 | 0.267 | 0.0202 | 0.0072 | 0.0095 |
| `qwen3.8-27b-int4-palmfuture`, `state_first` | local-gpu | 2,799 | 0.7208 | 502.6 | 1.4 | 0.0 | 1.42 | 2372 | 29338 | 2.67 | 16 | 0.291 | 0.0220 | 0.0079 | 0.0106 |
| `qwen3.6-35b-a3b-int4-palmfuture`, `state_first` | local-gpu | 2,799 | 0.7131 | 625.9 | 1.6 | 0.0 | 1.62 | 450 | 7126 | 11.81 | 16 | 0.066 | 0.0050 | 0.0018 | 0.0024 |

- The MoE builds are about 5× cheaper in GPU time. The Qwen3.x-35B-A3B builds used 0.056 to 0.066
  GPU-h per screen, against 0.266 to 0.306 for the dense 27B builds.
  `qwen3-30b-a3b-2507-int4-redhat` used the least, at 0.017. In throughput that is 11.81 to 13.79
  q/s for the 35B-A3B builds against 2.54 to 2.92 for the dense 27B ones (44.62 for
  `qwen3-30b-a3b-2507-int4-redhat`), all at 16 in flight. Jev's screen answers were replayed from
  its response cache, so it has no screen q/s.
- The layout changes GPU time little. From `question_first` to `state_first`, GPU-h per screen goes
  from 0.277 to 0.266 (QuantTrio), 0.271 to 0.267 (cyankiwi), 0.272 to 0.291 (Qwen3.8) and 0.058 to
  0.066 (Qwen3.6-35B-A3B). Where it rises, so do calls and prompt tokens per item (Qwen3.8: 1.31 to
  1.42 calls, 464 to 503 tokens).

## GPT-6 Luna (OpenAI API, logprob-scored)

Sep 25, 2026; rerun Sep 27 with Structured Outputs

`gpt-6-luna` answers through the same `Emulator`, over `jevemu.backends.openai_chat`:

- prompt: `state_first` without the `Answer:` prefill (`state_first-noprefill-5d29289f8b87`);
- answer: Structured Outputs. The reply must be `{"answer": "<label>"}`, with the label from a
  JSON-schema `enum`. The label is read from the top logprobs of the token where the value
  starts (`--strategy constrained`). The mask is reflected in those logprobs, so only labels
  compete there;
- settings: reasoning off (`reasoning_effort="none"`), `temperature=0`, `top_logprobs=5`,
  `max_completion_tokens=16`, `prompt_cache_options.mode="explicit"`;
- no debiaser, no calibration.

OpenAI usually lists only 1 or 2 labels at the value (83% of `select` responses list 1). A label
missing from the list gets the probability the listed labels leave over, at most 2%. Under the mask
that is a true upper bound, but Luna's raw NLL depends on it. On `holdout` its calibrated NLL moves
by at most 0.02 when the missing labels get 0 or an equal share of the leftover instead, and its
accuracy not at all.

[openai_probe_report.md](openai_probe_report.md) covers what the API allows and why these
settings were chosen.

```bash
uv run python scripts/run_split.py run --system openai --strategy constrained \
    --prompt-cache-mode explicit --system-id gpt-6-luna.constrained.state_first \
    --benchmarks all --split select --out runs/select --concurrency 16 --max-usd 2
```

OpenAI models cannot run 3 of the 9 benchmarks. The API returns at most 5 top logprobs (the
reference documents 20). Without a prefill there is no other way to score a label (no echo, no
trie). So questions with more than 5 options cannot show every label, and MMLU-Pro (up to 10
options), banking77 (77) and CLINC150 (150) are skipped. Yelp is left out, as for every system.
Every number in this section covers the other 6 benchmarks (7,535 `select` items). Jev and the
emulator are paired with Luna on those items only, so their 9-benchmark numbers are not comparable.
The QuantTrio columns are its one-call run (`auto_single`).

In the earlier free read, the first runs (Sep 25) read the first token of a free reply instead:
macro 0.7400 on `select`, raw NLL 1.041. They are archived in
`runs/archive_noprefill/select/gpt-6-luna.first_token.state_first`.

### Accuracy (`select`, paired)

Δ = Luna minus the other system. McNemar counts items only Luna got right / only the other got right.

| Benchmark | n | Luna | Jev | Δ vs Jev [95% CI] | McNemar | QuantTrio | Δ vs QuantTrio [95% CI] | McNemar |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| gpqa_diamond_idk | 99 | 0.4646 | 0.6465 | -0.1818 [-0.3030, -0.0606] | 11/29, p 0.006 | 0.4747 | -0.0101 [-0.1212, +0.1010] | 16/17 |
| lexam_en_idk | 310 | 0.7290 | 0.7452 | -0.0161 [-0.0742, +0.0419] | 39/44 | 0.6774 | +0.0516 [+0.0000, +0.1032] | 43/27 |
| arc_challenge | 586 | 0.9608 | 0.9744 | -0.0137 [-0.0307, -0.0017] | 6/14 | 0.9710 | -0.0102 [-0.0273, +0.0017] | 6/12 |
| ag_news | 3,800 | 0.8839 | 0.8845 | -0.0005 [-0.0066, +0.0050] | 65/67 | 0.8876 | -0.0037 [-0.0100, +0.0018] | 59/73 |
| boolq | 1,635 | 0.8795 | 0.9083 | -0.0287 [-0.0428, -0.0165] | 36/83, p 2.0e-05 | 0.8875 | -0.0080 [-0.0214, +0.0043] | 53/66 |
| sst5 | 1,105 | 0.5557 | 0.5756 | -0.0199 [-0.0462, +0.0036] | 89/111 | 0.5665 | -0.0109 [-0.0389, +0.0154] | 110/122 |
| **macro (6)** | 7,535 | **0.7456** | 0.7891 | **-0.0435 [-0.0661, -0.0204]** |  | 0.7441 | **+0.0015 [-0.0197, +0.0234]** |  |

Raw probability quality (no calibration), macro over the 6: Luna's NLL is 1.224 [1.126, 1.328]
and Brier 0.410 [0.381, 0.438]. Against Jev: Δ NLL +0.518 [+0.433, +0.606], Δ Brier +0.103
[+0.076, +0.129]. Against QuantTrio: Δ NLL +0.506 [+0.421, +0.596], Δ Brier +0.051
[+0.025, +0.076]. The `screen` run (1,599 items on the 6) gave the same picture: macro 0.7452,
-0.0447 [-0.0698, -0.0201] vs Jev, +0.0022 [-0.0223, +0.0262] vs QuantTrio.

### Cost and latency (`select`, the same 6 benchmarks)

Costs are `run_split.py stats` per benchmark, summed over the 6. $ / 1k correct divides by the
pooled correct answers. Latency is pooled over the same items: HTTP round trip for Luna and Jev,
time per question at 16 in flight for QuantTrio.

| System | Items | Macro acc. | Cost $ | $ / 1k items | $ / 1k correct | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev `jev-1.13.0` | 7,535 | 0.7891 | 0.1192 | 0.0158 | 0.0188 | 136 | 197 |
| **GPT-6 Luna** | 7,535 | 0.7456 | 0.1520 | 0.0202 | 0.0243 | 704 | 930 |
| QuantTrio (one-call run; 0.210 GPU-h at $0.1107/h) | 7,535 | 0.7441 | 0.0232 | 0.0031 | 0.0037 | 1,460 | 2,535 |

- Luna bills 146.7 input and 11.0 output tokens per item (Standard price: $0.10 and $0.50 per 1M).
  The schema adds about 35 input tokens and the JSON reply 7 output tokens over the free read. Jev
  bills 376.7 input tokens per item (2.6x, from its own prompting) at $0.042 per 1M, with free
  output. So Luna costs 28% more per item and 29% more per correct answer. The `screen` subset was
  replayed from the response cache and is costed at list price.
- Luna's p50 latency is 5.2x Jev's. No reply was empty or had to be sent again.

### Findings

- Luna is tied with QuantTrio on every benchmark (+0.002 macro) and 0.044 behind Jev, with the gap
  on GPQA (-0.18) and BoolQ (-0.03). Answering in JSON does not close the GPQA knowledge gap that
  motivated this run.
- Structured Outputs moved accuracy little: +0.006 macro over the free read (GPQA 0.465 against
  0.434). Raw NLL got worse (1.224 against 1.041).
- Luna's raw probabilities are overconfident: its NLL is 0.52 worse than Jev's at 0.044 lower
  accuracy. Calibration on `holdout` recovers most of it ([summary
  report](../../reports/summary/index.html)).
- Luna now costs more than Jev per item and per correct answer, and about 6.5x the local emulator at
  410 W.
- In the probe, 2 of 100 items changed their top answer between identical structured calls. So
  single-item agreement with Jev is noisier than for the local emulator; the paired intervals above
  include that variance.

## DeepSeek V4.1 Flash (OpenRouter, logprob-scored)

Sep 25, 2026; rerun Sep 27 with Structured Outputs on another provider

`deepseek/deepseek-v4.1-flash` answers through the same `Emulator`, over
`jevemu.backends.openrouter_chat`, pinned to the `makora` provider. Every response came from
Makora.

- prompt: `state_first` without the `Answer:` prefill;
- answer: Structured Outputs, as for Luna. The reply must be `{"answer": "<label>"}`, and the
  label is read from the top 20 logprobs of the token where the value starts
  (`--strategy constrained`);
- settings: reasoning off (`reasoning: {"enabled": false}`), `temperature=0`, `max_tokens=16`
  (64 when a stuck reply is sent again);
- no debiaser, no calibration.

The first runs used Wafer. For the structured read, a provider must return each position's own top
logprobs. Wafer's list at the value was often a copy of an earlier position's (31 of 40 replies), so
it would read the wrong position. Makora's lists were fresh in every probe reply, and it takes the
likeliest token at temperature 0. It charges $0.30 per 1M input tokens, 3x Wafer's.
[openrouter_probe_report.md](openrouter_probe_report.md) covers the other providers, what the API
allows and why these settings were chosen.

```bash
uv run python scripts/run_split.py run --system openrouter --provider makora --strategy constrained \
    --system-id deepseek-v4.1-flash@makora.constrained.state_first \
    --benchmarks all --split select --out runs/select --concurrency 16 --max-usd 3
```

DeepSeek runs 7 of the 9 benchmarks. OpenRouter returns up to 20 top logprobs, so MMLU-Pro (up to 10
options) runs here, unlike on Luna. Banking77 (77) and CLINC150 (150) are skipped. Yelp is left out,
as for every system. Every number in this section covers those 7 benchmarks (13,551 `select` items),
with the other systems paired on the same items. Against Luna, the pairing covers Luna's 6. The
QuantTrio and fast MoE numbers are their one-call runs.

On 3 SST-5 items Makora's reply loops on whitespace before the value, even when sent again at
`max_tokens` 64 (about 20 attempts each). They are left unanswered: DeepSeek is scored on the other
13,548, and paired comparisons use the items both systems answered.

In the earlier free read, the Wafer runs (Sep 25) read the first token of a free reply: macro 0.7267
on `select`, raw NLL 1.397. They are archived in
`runs/archive_noprefill/select/deepseek-v4.1-flash@wafer.first_token.state_first`.

### Accuracy (`select`, paired)

Δ = DeepSeek minus the other system. McNemar counts items only DeepSeek got right / only the
other got right.

| Benchmark | n | DeepSeek | Jev | Δ vs Jev [95% CI] | McNemar | QuantTrio | Δ vs QuantTrio [95% CI] | McNemar |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| gpqa_diamond_idk | 99 | 0.4949 | 0.6465 | -0.1515 [-0.2727, -0.0303] | 11/26, p 0.020 | 0.4747 | +0.0202 [-0.0808, +0.1212] | 15/13 |
| lexam_en_idk | 310 | 0.6419 | 0.7452 | -0.1032 [-0.1581, -0.0484] | 23/55, p 3.8e-04 | 0.6774 | -0.0355 [-0.0903, +0.0194] | 31/42 |
| mmlu_pro | 6,016 | 0.7380 | 0.8263 | -0.0883 [-0.0989, -0.0778] | 289/820, p 2.4e-59 | 0.6297 | +0.1084 [+0.0961, +0.1203] | 1034/382, p 1.5e-69 |
| arc_challenge | 586 | 0.9471 | 0.9744 | -0.0273 [-0.0461, -0.0154] | 2/18, p 4.0e-04 | 0.9710 | -0.0239 [-0.0410, -0.0137] | 1/15, p 5.2e-04 |
| ag_news | 3,800 | 0.8916 | 0.8845 | +0.0071 [+0.0005, +0.0132] | 89/62, p 0.034 | 0.8876 | +0.0039 [-0.0021, +0.0097] | 72/57 |
| boolq | 1,635 | 0.8991 | 0.9083 | -0.0092 [-0.0220, +0.0012] | 38/53 | 0.8875 | +0.0116 [-0.0006, +0.0220] | 56/37 |
| sst5 | 1,102 | 0.5554 | 0.5762 | -0.0209 [-0.0472, +0.0027] | 90/113 | 0.5672 | -0.0118 [-0.0381, +0.0136] | 101/114 |
| **macro (7)** | 13,548 | **0.7383** | 0.7945 | **-0.0562 [-0.0754, -0.0375]** |  | 0.7279 | **+0.0104 [-0.0071, +0.0277]** |  |

Against the fast MoE emulator (`qwen3.6-35b-a3b-fast`, 0.7064): +0.0319 [+0.0125, +0.0514].
Against GPT-6 Luna on Luna's 6 benchmarks (7,532 items): 0.7383 vs 0.7457, -0.0074
[-0.0282, +0.0126]. DeepSeek is ahead on AG News (+0.008) and BoolQ (+0.020) and behind on
LEXam (-0.087).

Raw probability quality (no calibration), macro over the 7: NLL 0.806 [0.763, 0.848], Brier
0.370 [0.351, 0.389]. Against Jev: Δ NLL +0.095 [+0.053, +0.136], Δ Brier +0.069
[+0.050, +0.087]. Against QuantTrio: Δ NLL +0.028 [-0.004, +0.060], Δ Brier -0.007
[-0.023, +0.008]. Against Luna: Δ NLL -0.449 [-0.536, -0.366]. The `screen` run (1,899 items on
the 7) gave the same picture: macro 0.7417, -0.0602 [-0.0814, -0.0382] vs Jev, +0.0148
[-0.0058, +0.0355] vs QuantTrio.

### Cost and latency (`select`, the same 7 benchmarks)

Costs are `run_split.py stats` per benchmark, summed over the 7. $ / 1k correct divides by the
pooled correct answers. DeepSeek's cost includes the billed attempts of the 3 unanswered items.

| System | Items | Macro acc. | Cost $ | $ / 1k items | $ / 1k correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev `jev-1.13.0` | 13,551 | 0.7944 | 0.2593 | 0.0191 | 0.0229 |
| **DeepSeek V4.1 Flash** (Makora) | 13,551 | 0.7383 | 0.7943 | 0.0586 | 0.0741 |
| QuantTrio (one-call run; 0.522 GPU-h at $0.1107/h) | 13,551 | 0.7278 | 0.0577 | 0.0043 | 0.0057 |
| Fast MoE (one-call run; 0.138 GPU-h at $0.1107/h) | 13,551 | 0.7063 | 0.0153 | 0.0011 | 0.0016 |

- DeepSeek bills 158.7 input and 9.1 output tokens per item ($0.30 and $1.20 per 1M on Makora). Jev
  bills 455.7 input tokens per item (2.9x) at $0.042 per 1M. So DeepSeek costs 3.1x as much per item
  and 3.2x per correct answer. The 1,899 `screen` items were replayed from the response cache and
  are costed at list price.
- Latency (HTTP round trip, 16 in flight) was p50 221 ms, p95 636 ms over the 13,548 answered items,
  at 7.4 q/s. Jev: p50 134 ms, p95 186 ms.

### Findings

- Like Luna, DeepSeek V4.1 Flash is tied with QuantTrio (+0.010) and behind Jev (by 0.056). It is
  also tied with Luna on the shared 6 (-0.007). The gap is on the knowledge-heavy multiple-choice
  sets (GPQA -0.15, LEXam -0.10, MMLU-Pro -0.09). On AG News it beats Jev (+0.007).
- With 20 top logprobs DeepSeek can run MMLU-Pro (Luna cannot), and there it is 0.108 ahead of
  QuantTrio. That offsets its ARC, LEXam and SST-5 deficits in the macro.
- Structured Outputs fixed the prose starts. On Wafer's free read, only 72% of the greedy first
  tokens on GPQA were labels in the probe; the model often started hard questions with words. With
  the value forced to a label, macro accuracy rose by 0.012 (MMLU-Pro from 0.698 to 0.738, GPQA from
  0.485 to 0.495) and raw NLL fell from 1.397 to 0.806.
- Its raw probabilities are now about as good as the local emulator's (Δ NLL +0.028 against
  QuantTrio), and 0.095 NLL worse than Jev's.
- It costs about 3x Jev per item and per correct answer; at the 410 W electricity estimate it costs
  about 14x QuantTrio and 53x the fast MoE.
- In the probe, no top answer changed in 100 identical repeated calls on Makora, against 2.4% on
  Wafer's free read.

## CLM-8B (dual encoder)

Sep 28, 2026. [CLM-8B](https://github.com/Contrastive-LM/CLM) (Apache-2.0) is a dual encoder.
Qwen3-8B, served by vLLM's pooling runner with last-token pooling, embeds the state plus the
question once and each option's text on its own. A 20M-parameter projection head scores each option
by scaled cosine similarity, and a softmax over the scores gives the distribution. Its server,
`clm-serve`, answers TypeSafe's `POST /v1/systemone` natively, so the Jev client drives it
(`--system clm`, `jevemu.clm_client.ClmClient`, `docker/clm/compose.yaml`, `scripts/serve_clm.sh`).
It works as a drop-in on the wire, but it fails the screen by a wide margin and is not a finalist.

The runs pin the head `Contrastive-LM/CLM-v0.1-8B` @ `e939398d4556fcd9400c76fa8c5a513202f42b0a`
(sha256 `b2b4a8c9c2d3…`), the encoder `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`
and `contrastive-lm` 0.1.0, in the same vLLM v0.30.0 image as the emulator. The server names the
model `clm-v0.1-8b-b2b4a8c9c2d3`. It ran on one RTX 3090 with 16 requests in flight, on `screen`,
`select` and `holdout` (`runs/select/clm/`):

```bash
eval "$(scripts/serve_clm.sh | grep '^export ')"
uv run python scripts/run_split.py run --system clm --benchmarks all --split select --out runs/select
```

| Split | Macro acc. (9) [95% CI] |
| --- | --- |
| `screen` (2,499 items) | 24.7% [23.2, 26.2] |
| `select` | 24.9% [24.0, 25.9] |
| `holdout` (17,340 items) | 24.9% [24.0, 25.8] |

On `holdout` it is 57.6 points below Jev, paired (−57.6 [−59.1, −56.2]; Jev 82.5%, QuantTrio
Qwen3.6-27B 76.6% on the same 9 benchmarks). Per benchmark: GPQA 4.0% (it picks IDK on 89.9% of the
questions), LEXam 24.0%, MMLU-Pro 14.6%, ARC 37.5%, AG News 35.7%, banking77 3.5%, CLINC150 8.8%,
BoolQ 76.9%, SST-5 18.6%. Its `holdout` NLL is 3.615 and its Brier score 1.046. It answered the
17,340 `holdout` items in 433 s (40.1 q/s at 16 in flight), which costs $0.0008 per 1k items at the
410 W electricity estimate.

The screen answers show where it fails. On SST-5 it nearly always picks an extreme level (284 of 300
"very positive", 16 "very negative"). On AG News it never picks Business (161 Sci/Tech, 137 Sports,
2 World). The option labels are embedded without the question, so on the reasoning benchmarks
(MMLU-Pro, GPQA, ARC) an option such as a bare number carries no context.

An audit of the setup found no integration bug. Our server matches upstream's own captured output:
their playground screenshot (a real `clm-serve`, `clm-latest`) shows urgency 84.8%, billing 98.8%
and frustration 2.00, and we get 0.84, 0.988 and 2.00. The numbers in the README's code comments
(urgency 0.41, billing 0.939, frustration 1.98) match neither the screenshot nor any run, so they
are not a usable reference. The audit also found:

- The encoder input matches the pre-training embeddings the head was trained on. Qwen3-8B vectors
  from vLLM agree with an in-process transformers last-token embedding at cosine 0.9999, and fresh
  question/answer pairs sit where the stored pre-training pairs do. Appending an end token or using
  the chat template moves them away.
- The PyPI wheel scores exactly like GitHub main (`bb42c6c`), the head file is the only one ever
  uploaded to the HF repo, and turning off the server's vector cache or sending one request at a
  time leaves accuracy within 1.5 points on 200-item samples.
- On `LocalLLaMA/typed-decisions` (test, 2,000 decisions), run through upstream's own code, it
  scores 35.8%, below the per-question majority label (52.2%). Jev 1.13 is listed at 72.7% there.
- Most of the GPQA result comes from the prompt. The IDK instructions name the option ("E (I don't
  know) earns 0"), and a similarity model latches onto that phrase. With the phrase removed, IDK
  drops from 0.899 to 0.152 and accuracy rises to 0.283 (chance for 4 options). Jev gets the same
  instructions, so the benchmark stays as it is.
- Two cold servers disagree on up to 7.5% of predictions (bf16 embedding noise amplified by a logit
  scale of 100); the response cache keeps the recorded answers fixed.

CLM's own claims (on par with Jev, up to 9x faster) are on agentic, tool-calling and game tasks,
which this benchmark does not cover.

## Speed/quality trade-off

Sep 25, 2026; rerun Sep 26 to 27 with one-call scoring. To map how much accuracy each step of
throughput costs, every candidate answered the `screen` split with the same settings: `state_first`,
`auto_single` (first run with the multi-call `auto_noecho`), no debiaser, 16 items in flight, one
fresh server per preset. This added `state_first` screens for 7 existing presets (the 4 of Stage 2a
already had one) and 7 new small models (2B to 12B:
[candidates.md](candidates.md#small-models-for-the-speedquality-study)).

The report is [reports/speed_quality/](../../reports/speed_quality/README.md)
(`scripts/make_tradeoff_report.py`). It has scatter plots of q/s against accuracy, ECE and Brier
(raw and cross-fitted `temperature@signature`), the Pareto frontier, and the best model per
speed tier. Yelp is left out (label wording, a user decision), so the macros cover 9
benchmarks (2,499 items). The multi-call columns are the same presets' `auto_noecho` screens
on the same 9 benchmarks; the Gemma 4 12B and 26B-A4B ones also predate the prompt fix.

```bash
uv run python scripts/sweep_presets.py run --split screen --out runs/select --strategy auto_single \
    --renderer state_first --concurrency 16 --presets P1,P2,...
uv run --extra plot python scripts/make_tradeoff_report.py
```

| Preset | Macro acc. (9) | Multi-call macro acc. (9) | q/s (9) | Multi-call q/s (9) | p50 ms |
| --- | --- | --- | --- | --- | --- |
| `qwen3.6-27b-int4-quanttrio` | 0.7538 | 0.7546 | 4.47 | 2.77 | 2,334 |
| `qwen3.6-27b-int4-cyankiwi` | 0.7460 | 0.7486 | 4.43 | 2.77 | 2,274 |
| `qwen3.6-35b-a3b-fast` (tuned, below) | 0.7349 | 0.7360 | 19.25 | 9.70 | 552 |
| `qwen3.8-27b-int4-palmfuture` | 0.7290 | 0.7353 | 4.50 | 2.52 | 2,280 |
| `qwen3.5-35b-a3b-int4` | 0.7276 | 0.7324 | 24.07 | 10.18 | 426 |
| `qwen3.6-35b-a3b-int4-palmfuture` | 0.7272 | 0.7279 | 23.84 | 11.01 | 421 |
| `gemma-4-26b-a4b-int4-cyankiwi` | 0.7127 | 0.7060 | 19.67 | 19.77 | 418 |
| `gemma-4-12b-int4` | 0.7078 | 0.6723 | 8.97 | 9.27 | 965 |
| `qwen2.5-32b-int4` | 0.6997 | 0.7034 | 3.88 | 4.00 | 2,477 |
| `qwen3.5-9b-int8` | 0.6922 | 0.6937 | 12.80 | 7.52 | 755 |
| `qwen3.5-9b-bf16` | 0.6908 | 0.6937 | 13.18 | 7.62 | 752 |
| `qwen3-30b-a3b-2507-int4-redhat` | 0.6860 | 0.6904 | 25.59 | 25.59 | 358 |
| `qwen3-32b-int4` | 0.6853 | 0.6882 | 3.88 | 3.97 | 2,545 |
| `qwen3-14b-int4` | 0.6652 | 0.6722 | 8.98 | 9.53 | 1,081 |
| `qwen3.5-4b-bf16` | 0.6586 | 0.6637 | 22.90 | 13.23 | 444 |
| `qwen3-4b-2507-bf16` | 0.6504 | 0.6615 | 27.52 | 26.35 | 334 |
| `gemma-4-e4b-bf16` | 0.6467 | 0.6526 | 22.14 | 21.33 | 377 |
| `qwen3-4b-bf16` | 0.6312 | 0.6497 | 27.27 | 26.53 | 341 |
| `phi-4-mini-bf16` | 0.5380 | 0.5550 | 31.28 | 27.88 | 286 |
| `smollm3-3b-bf16` | 0.5034 | 0.5474 | 32.90 | 30.84 | 275 |
| `llama-3.2-3b-bf16` | 0.4941 | 0.5300 | 35.58 | 32.61 | 252 |
| `qwen3.5-2b-bf16` | 0.4922 | 0.5511 | 49.09 | 20.62 | 196 |

- By a user decision, the other five dense 27B builds (Qwen3.8 groxaxo, btbtyler09, cyankiwi AWQ,
  RedHat; Qwen3.6 groxaxo) were not re-run. On the Stage 1 screen each is within about 0.01 of a
  plotted build of the same model. SmolLM3-3B and Llama 3.2 3B were screened after the cache move
  (22:28 to 22:34 UTC, 6 min of GPU). Ministral 3 cannot load in the pinned image (candidates.md).
- `state_first` helps the attention-only models too (multi-call runs, 10 benchmarks, against Stage
  1: Qwen3-30B-A3B-2507 from 0.5970 to 0.6704, Qwen3-14B from 0.5818 to 0.6500). But it halved their
  throughput (from 44.6 to 26.2 and from 18.6 to 9.6 q/s). `question_first` let the prefix cache
  share the fixed instructions and options across items; `state_first` puts the per-item text first.
- One-call scoring makes the Qwen3.5, Qwen3.6 and Qwen3.8 models 1.6x to 2.4x faster and moves the
  others by at most about 12% either way. It costs the smallest models a lot of accuracy on
  banking77 and CLINC150 ([What one-call scoring changed](#what-one-call-scoring-changed)).
- The multi-call screens took 1 h 40 min of GPU time (20:37 to 22:17 UTC): 58 min of existing-preset
  screens, 14 min of render/label preflights, 29 min of small-model screens. The one-call screens
  took 2.6 h of server time: 24 runs, the 22 presets plus two Gemma screens made before the prompt
  fix.

## Fast MoE tuning (Qwen3.6-35B-A3B)

Sep 25 to 26, 2026. The goal was the fastest Qwen3.6-35B-A3B configuration that keeps the accuracy
of the screened palmfuture build. It becomes the "fast" emulator of the Jev vs Qwen report.
Everything ran on the pinned vLLM 0.30.0 image and one RTX 3090, with one fresh server per
configuration, `state_first`, `auto_noecho` and no debiaser.

*Multi-call history (before one-call scoring): the tuning used `auto_noecho`, so banking77 and
CLINC150 took several prompt-only calls per item. The throughput findings below describe that
load.*

Method:

- A throwaway driver (not in the repo) served each configuration with `scripts/serve_vllm.sh`
  from environment overrides (no preset file). It ran `scripts/run_split.py run --split screen
  --limit 50` into the gitignored `runs/tune/<name>/`. That is a fixed 500-item subset: the
  first 50 screen items of each of the 10 benchmarks, Yelp included (this ran before Yelp left
  the protocol).
- It logged GPU memory (`nvidia-smi`), the vLLM log's KV cache and backend lines, and the
  scheduler's running/waiting counts.
- Each run is paired with the base on the same 500 items. It reports the macro accuracy
  difference (stratified bootstrap, 2,000 resamples) and how far the per-item answer
  distributions diverge: KL(base ‖ run) (Kullback-Leibler divergence), total variation (TV)
  and top-1 agreement.
- The base is the `qwen3.6-35b-a3b-int4-palmfuture` preset: 0.95 of the GPU,
  `--max-model-len=8192`, `--max-num-seqs=16`, `--max-num-batched-tokens=1024`, prefix caching
  on, 16 items in flight.
- The noise floor comes from the base restarted with identical flags, and from a second identical
  pair (`b2048` against `b2048 + throughput`, which changes nothing; see below): KL 0.005 to 0.006,
  TV 0.024 to 0.025, top-1 agreement 0.974 to 0.976, accuracy ±0.006. A flag that moves the answer
  distributions beyond that floor is rejected even when accuracy is unchanged.

### Tuning table (palmfuture, 500-item screen subset, 16 in flight unless noted)

| Configuration (change from base) | q/s | p50 / p95 ms | GPU MiB idle / peak | KV tokens | Accuracy | Δ acc. vs base [95% CI] | KL mean (p95) | TV | Top-1 agree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base (the palmfuture preset) | 9.08 | 495 / 9,076 | 22,262 / 22,386 | 20,480 | 0.682 | | | | |
| base, restarted (noise floor) | 9.24 | 496 / 8,845 | 22,262 / 22,388 | 20,480 | 0.686 | +0.004 [-0.008, +0.016] | 0.0052 (0.023) | 0.024 | 0.976 |
| batched tokens 2048 | 9.46 | 524 / 8,965 | 22,326 / 22,674 | 21,650 | 0.686 | +0.004 [-0.008, +0.014] | 0.0058 (0.022) | 0.026 | 0.976 |
| batched tokens 4096 | 8.67 | 507 / 11,744 | 22,450 / 23,352 | 22,235 | 0.682 | +0.000 [-0.010, +0.010] | 0.0062 (0.024) | 0.025 | 0.982 |
| batched tokens 8192 | 8.58 | 556 / 11,075 | 22,666 / 23,640 | 21,065 | 0.684 | +0.002 [-0.010, +0.014] | 0.0060 (0.024) | 0.025 | 0.974 |
| 2048 + `--performance-mode=throughput` | 9.70 | 522 / 8,346 | 22,326 / 22,746 | 21,650 | 0.680 | -0.002 [-0.012, +0.008] | 0.0063 (0.026) | 0.026 | 0.982 |
| 2048 + `--no-enable-prefix-caching` | 9.68 | 523 / 8,710 | 22,326 / 22,658 | 27,554 | 0.684 | +0.002 [-0.010, +0.014] | 0.0067 (0.028) | 0.026 | 0.976 |
| 2048 + `--prefix-match-unit=32` | 9.51 | 508 / 8,691 | 22,326 / 22,742 | 21,650 | 0.682 | +0.000 [-0.008, +0.008] | 0.0055 (0.024) | 0.025 | 0.986 |
| 2048, 0.97 of the GPU | 9.51 | 480 / 8,690 | 22,822 / 23,166 | 35,693 | 0.676 | -0.006 [-0.016, +0.002] | 0.0061 (0.024) | 0.025 | 0.980 |
| 2048, 0.98 of the GPU | 9.53 | 479 / 8,690 | 23,048 / 23,404 | 42,130 | 0.678 | -0.004 [-0.016, +0.006] | 0.0057 (0.024) | 0.025 | 0.976 |
| 2048, 0.97, `--max-cudagraph-capture-size=512` | 9.49 | 523 / 9,120 | 22,802 / 23,114 | 18,139 | 0.680 | -0.002 [-0.010, +0.006] | 0.0057 (0.025) | 0.025 | 0.984 |
| 2048 + `--moe-backend=humming` | 9.38 | 520 / 8,444 | 22,266 / 22,616 | 19,309 | 0.680 | -0.002 [-0.006, +0.000] | 0.0061 (0.025) | 0.025 | 0.992 |
| 2048 + `--kv-cache-dtype=fp8` (FlashInfer) | 9.58 | 527 / 8,400 | 22,746 / 23,078 | 29,491 | 0.680 | -0.002 [-0.016, +0.012] | **0.0084 (0.039)** | **0.030** | **0.970** |
| 2048, 0.97, `--max-model-len=4096`, 32 seqs, 32 in flight | 7.52 | 1,079 / 19,848 | 22,822 / 23,220 | 22,937 | 0.688 | +0.006 [-0.006, +0.020] | 0.0067 (0.025) | 0.026 | 0.972 |

Did not start:

- 64 seqs at 64 in flight (`--max-model-len=4096`, 0.97): "available Mamba cache blocks (47)",
  one per decode sequence, too few to capture CUDA graphs;
- capture size 512 or 2048 at 0.95: the graphs leave 0.15 GiB or less for the KV cache, short of
  one request;
- `--moe-backend=triton`: "WNA16 MoE backend 'TRITON' does not support ... expert bias".

The image's other WNA16 MoE backends are Blackwell-only (FlashInfer TRT-LLM), ROCm (RDNA3) or
emulation.

Findings:

- Throughput is bound by prefill compute and barely moves. Every configuration that started ran at
  8.6 to 9.7 q/s, against 9.1 to 9.2 for the base and its restart. Each item is 1 to 4 prompt-only
  calls: the trie strategy re-sends the prompt for multi-token labels, and banking77 and CLINC150
  took 42 of the base's 56 s. So the GPU runs prefill at 4k to 6k prompt tokens/s whatever the batch
  shape.
  - More KV cache (0.97 to 0.98 of the GPU: 36k to 42k tokens) raised concurrency on the server but
    not q/s.
  - Larger batched-token budgets were slower (8.6 to 8.7 q/s).
  - 32 in flight only queued more (7.5 q/s, p50 doubled).
- Prefix caching has no effect here. This hybrid model caches Mamba state in "align" mode, at
  1,056-token block boundaries, so the base hit 0 cached tokens (`state_first` puts the per-item
  text first). `--prefix-match-unit=32` let the trie's follow-up calls hit (110 cached tokens per
  item) but gained nothing measurable. Turning prefix caching off frees 27% more KV tokens in the
  same memory, within the noise floor.
- `--performance-mode=throughput` has no effect when `--max-num-seqs` and `--max-num-batched-tokens`
  are set: in 0.30.0 it only doubles those two defaults (read from the image's
  `engine/arg_utils.py`). Its row is a second noise-floor sample.
- FP8 KV cache is rejected. On Ampere it runs only through FlashInfer (FlashAttention 2 has no FP8
  KV path; Triton attention needs SM 8.9+). It moves the answer distributions beyond the restart
  floor (KL 0.008, p95 0.039, top-1 agreement 0.962 to 0.970) and gains no speed.

### Choosing the quantized build (full screen split, 2,799 items, 16 in flight)

| Build | Weights on GPU | KV cache | Macro acc. (9) | Macro acc. (10, with Yelp) | NLL (10) | q/s (10) | p50 / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` (its preset) | 20.27 GiB | 0.72 GiB, 20,480 tokens | 0.7279 [0.7101, 0.7461] | 0.7131 [0.6964, 0.7302] | 0.882 | 11.81 | 450 / 7,126 |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` @ `119886a107` (the fast flags) | 21.38 GiB | 0.33 GiB, 9,362 tokens | 0.7338 [0.7160, 0.7520] | 0.7211 [0.7044, 0.7382] | 0.869 | 10.44 | 571 / 7,559 |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` @ `00fcea2d3b` | 22.03 GiB | does not fit | | | | | |

The palmfuture row is its earlier screen with the preset's (base) flags. The tuning table shows
those are within the noise floor of the tuned flags.

- QuantTrio − palmfuture, paired: accuracy +0.0059 [-0.0052, +0.0164] over 9 benchmarks (+0.0080
  [-0.0024, +0.0177] with Yelp), NLL -0.013. The answer distributions differ well beyond the noise
  floor (KL 0.033, TV 0.059, top-1 agreement 0.935), as expected of a different quantization.
- QuantTrio was served at 0.98 of the GPU with `--max-model-len=4096` (the longest
  `select`/`holdout` prompt is 2,825 tokens). At 0.97, its measured 21.63 GiB of weights and
  non-torch memory, 1.13 GiB activation peak and 0.1 GiB of CUDA graphs would leave no KV cache
  [INFERENCE, not run]. Its layer-0 experts stay bf16 (TritonExperts); the other routed experts run
  AWQ-Marlin.
- cyankiwi does not fit. Its group-32 build loaded 22.03 GiB and left -0.32 GiB for the KV cache at
  0.98 (-0.20 GiB with 512 batched tokens and 8 seqs). CPU offload was not considered.

QuantTrio was chosen. It has the higher accuracy (tied within its interval), then the lower NLL;
both tie-breaks come before speed. It runs at 12% fewer q/s than palmfuture.

All weights are on the GPU. vLLM logged "Model loading took 21.38 GiB" (the safetensors
language-model share is 21.30 GiB), with no offload line. `nvidia-smi` showed the vLLM process
holding 23,064 MiB of 24,576 at 100% utilization during the `select` run (no `--cpu-offload-gb`).

### The fast preset: `qwen3.6-35b-a3b-fast`

[docker/vllm/presets/qwen3.6-35b-a3b-fast.env](../../docker/vllm/presets/qwen3.6-35b-a3b-fast.env):
QuantTrio AWQ, bf16 activations, 0.98 of the GPU, `--max-model-len=4096`, `--max-num-seqs=16`,
`--max-num-batched-tokens=1024`, `--no-enable-prefix-caching`, 16 items in flight (the tuned
concurrency; the preset file explains why). It is not for echo strategies: the prompt-logprob spike
does not fit the remaining headroom.

```bash
for s in screen select holdout; do
  uv run python scripts/sweep_presets.py run --split $s --out runs/select --strategy auto_single \
      --renderer state_first --concurrency 16 --presets qwen3.6-35b-a3b-fast
done
```

| Split | Items | Macro acc. (9) | NLL (9) | − Jev (9) | − QuantTrio 27B (9) | q/s (9) | q/s (6 shared) | p50 / p95 ms (9) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `screen` | 2,499 | 0.7349 [0.7171, 0.7528] | 0.886 | -0.0824 [-0.1007, -0.0645] | -0.0189 [-0.0350, -0.0033] | 19.25 | 28.15 | 552 / 1,733 |
| `select` | 17,341 | 0.7264 [0.7129, 0.7401] | 0.904 | -0.0834 [-0.0981, -0.0686] | -0.0256 [-0.0387, -0.0123] | 20.07 | 29.67 | 581 / 1,755 |
| `holdout` | 17,340 | 0.7304 [0.7167, 0.7443] | 0.900 | -0.0947 [-0.1084, -0.0810] | -0.0352 [-0.0488, -0.0216] | 19.70 | 28.07 | 586 / 1,758 |

These are the one-call runs. For reference on `select`: Jev 0.8098; QuantTrio 27B 0.7520 at
4.44 q/s, p50 2,374 ms. Differences are paired on shared items (`compare`, stratified
bootstrap). q/s is items over the summed runner wall time (startup excluded), pooled over the
9 benchmarks or the 6 that every system answers. Neither `holdout` run has a debiaser.

The fast preset gives up about 0.03 accuracy against the dense QuantTrio 27B for 4.5x its
throughput (20.07 against 4.44 q/s on `select`). In the multi-call runs it gave up about 0.02
for 3.8x (11.07 against 2.94 q/s), and its screen accuracy repeated the tuning run's (10
benchmarks: 0.7238 against 0.7211, a restart).

- The multi-call work took 2 h 19 min of GPU time (22:34 UTC Sep 25 to 00:59 UTC Sep 26, less 6 min
  of idle gaps): tuning 70 min (18 palmfuture configurations, 4 of which did not start, the
  QuantTrio screen and two cyankiwi attempts; it includes 6 min before a restart and an interrupted
  restart run), the fast preset's screen, `select` and `holdout` 66 min (three fresh servers, 6 min
  of startups), the registry smoke test 3 min.

### The GPTQ build as a finalist: `qwen3.6-35b-a3b-int4-palmfuture`

Sep 26, 2026. The build choice above ranked by accuracy, then NLL, and only then speed. Its speed
column pooled q/s over all 10 benchmarks, and that hid most of the gap. In those multi-call runs
banking77 and CLINC150 took most of the wall time, and they ran at the same speed on both builds
(3.4 to 3.7 q/s on `holdout`). On the 6 benchmarks every system answers (GPQA, LEXam, ARC, AG News,
BoolQ, SST-5; one call per question even then), the screen already showed the GPTQ build at 1.65x
the AWQ's speed (44.67 against 27.06 q/s), against 1.13x pooled over 10 (11.81 against 10.44). So
the GPTQ build answered `select` and `holdout` too, with its own preset (0.95 of the GPU,
`--max-model-len=8192`, prefix caching on; the tuning table shows these flags are within the noise
floor of the tuned ones). All numbers below are from the one-call reruns unless marked.

```bash
for s in select holdout; do
  uv run python scripts/sweep_presets.py run --split $s --out runs/select --strategy auto_single \
      --renderer state_first --concurrency 16 --presets qwen3.6-35b-a3b-int4-palmfuture
done
```

| Build | Split | Items (9) | Macro acc. (9) | NLL (9) | − AWQ fast (9) | − Jev (9) | q/s (9) | q/s (6 shared) | p50 / p95 ms (9) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPTQ | `select` | 17,341 | 0.7224 [0.7087, 0.7359] | 0.922 | -0.0041 [-0.0129, +0.0045] | -0.0875 [-0.1019, -0.0730] | 25.11 | 52.99 | 422 / 1,673 |
| GPTQ | `holdout` | 17,340 | 0.7265 [0.7126, 0.7403] | 0.915 | -0.0039 [-0.0146, +0.0068] | -0.0986 [-0.1114, -0.0855] | 25.10 | 53.17 | 426 / 1,669 |
| AWQ fast | `select` | 17,341 | 0.7264 [0.7129, 0.7401] | 0.904 | | -0.0834 [-0.0981, -0.0686] | 20.07 | 29.67 | 581 / 1,755 |
| AWQ fast | `holdout` | 17,340 | 0.7304 [0.7167, 0.7443] | 0.900 | | -0.0947 [-0.1084, -0.0810] | 19.70 | 28.07 | 586 / 1,758 |

Both runs answered every item with 0 errors. Differences are paired (`compare`). "q/s (6
shared)" is items over the summed runner wall time of those 6 benchmarks. The calibration
study gives GPTQ − Jev on `holdout` as -0.0989, because it renormalizes Jev's probabilities
([calibration.md](calibration.md#current-results-one-call-per-question-holdout)).

- GPTQ's accuracy is tied with AWQ fast on both splits and on the screen (-0.0078 [-0.0182,
  +0.0027]). Per benchmark, BoolQ is lower on both splits (`select` -0.0110 [-0.0202, -0.0037],
  `holdout` -0.0196 [-0.0300, -0.0116]), and CLINC150 on `holdout` (-0.0089 [-0.0173, -0.0013]). In
  the multi-call runs GPTQ was worse on `select` (-0.0113 [-0.0204, -0.0025]) and tied on `holdout`.
- GPTQ's NLL is higher on both splits: +0.018 [+0.009, +0.027] on `select`, +0.015 [+0.005, +0.024]
  on `holdout`. After calibration the gap closes (+0.003 [-0.005, +0.010];
  [calibration.md](calibration.md#current-results-one-call-per-question-holdout)).
- On `select` GPTQ runs 52.99 against 29.67 q/s on the 6 shared benchmarks (1.8x), and 25.11 against
  20.07 over all 9 (1.25x). banking77 and CLINC150 take about half of the GPTQ build's `holdout`
  wall time [INFERENCE: 1,540 items at 13.29 q/s plus 2,250 at 9.54 q/s, out of about 690 s]. There
  the two builds run at almost the same speed (AWQ fast: 12.49 and 9.22 q/s).
- The cost on `holdout` (`run_split.py stats`, 410 W) is 0.192 against 0.244 GPU-h, $0.0012 against
  $0.0016 per 1k items.
- The multi-call run needed a manifest edit. The run directory was first written by the screen run
  (Sep 25), which recorded `confidence_fn` `peak_linear`. The default is now `mode_distance`, so the
  first launch stopped at the manifest check before answering any item. The manifest field was
  changed to `mode_distance` (backup:
  `runs/select/sweep/qwen3.6-35b-a3b-int4-palmfuture.auto_noecho.state_first.manifest.pre_select.json`).
  The screen records' SST-5 `confidence` values therefore use `peak_linear`. Probabilities and every
  metric (accuracy, NLL, Brier, ECE, calibration) do not read `confidence` and are unaffected.
  `select` and `holdout` use `mode_distance`, like AWQ fast.
- The multi-call runs took about 56 min of GPU time on Sep 26: `select` from 16:27 to 16:53 UTC (25
  min 41 s, 135 s of startup), `holdout` from 16:53 to 17:19 UTC (25 min 37 s, 126 s of startup),
  plus 4.5 min for the failed first launch (two server starts, 0 items).

The GPTQ build is as accurate as the AWQ fast preset on both splits, with slightly worse raw
probabilities (NLL; ECE 0.087 against 0.077 on `holdout`). It is about 1.8x faster on the 6 shared
benchmarks and 1.25x faster over all 9. The AWQ build stays the `qwen3.6-35b-a3b-fast` preset. The
GPTQ build has its own calibration registry,
[`calibration/qwen3.6-35b-a3b-int4-palmfuture/registry.json`](../../calibration/qwen3.6-35b-a3b-int4-palmfuture/registry.json).

### Gemma 4 26B-A4B as a finalist

Sep 26, 2026. The fast Qwen emulators are mixture-of-experts (MoE) models with about 3B active
parameters. Gemma 4 26B-A4B is Google's model of the same kind: 3.8B of 25.2B parameters active per
token. Google publishes no W4A16 build of it, so the preset `gemma-4-26b-a4b-int4-cyankiwi` serves a
community INT4 build of Google's quantization-aware-trained checkpoint
(`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`, compressed-tensors W4A16). It loads at 0.90 of the GPU
with the Gemma 4 12B flags ([candidates.md](candidates.md#gemma-4-26b-a4b)). It answered all three
splits with the same protocol as the Qwen builds: `state_first`, `auto_single`, no debiaser, 16 in
flight. Its runs were redone after the Gemma 4 prompt fix
([above](#scoring-now-one-call-per-question)).

```bash
for s in screen select holdout; do
  uv run python scripts/sweep_presets.py run --split $s --out runs/select --strategy auto_single \
      --renderer state_first --concurrency 16 --presets gemma-4-26b-a4b-int4-cyankiwi
done
```

| Build | Split | Items (9) | Macro acc. (9) | NLL (9) | − AWQ fast (9) | − GPTQ (9) | − Jev (9) | q/s (9) | q/s (6 shared) | p50 / p95 ms (9) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma | `screen` | 2,499 | 0.7127 [0.6948, 0.7305] | 1.286 | -0.0223 [-0.0410, -0.0037] | -0.0145 [-0.0335, +0.0038] | -0.1047 [-0.1229, -0.0872] | 19.67 | 45.23 | 418 / 2,280 |
| Gemma | `select` | 17,341 | 0.7095 [0.6957, 0.7229] | 1.295 | -0.0170 [-0.0319, -0.0023] | -0.0129 [-0.0274, +0.0013] | -0.1004 [-0.1150, -0.0857] | 20.47 | 52.29 | 424 / 2,308 |
| Gemma | `holdout` | 17,340 | 0.7349 [0.7209, 0.7482] | 1.270 | +0.0045 [-0.0107, +0.0196] | +0.0084 [-0.0067, +0.0235] | -0.0901 [-0.1048, -0.0757] | 20.44 | 52.33 | 425 / 2,310 |

Every run answered every item with 0 errors. Differences are paired (`compare`). The Qwen rows
for `select` and `holdout` are in the GPTQ table above. On the screen (9 benchmarks), AWQ fast
ran 19.25 q/s (28.15 on the 6 shared) and GPTQ 23.84 (44.96).

- Gemma is below AWQ fast on the screen and on `select`, and tied with both Qwen MoEs on `holdout`.
  Gemma moves more between splits than the Qwen MoEs: +0.025 from `select` to `holdout`, against
  +0.004 for both Qwen builds. Per benchmark on `holdout`, Gemma is better on the two intent
  benchmarks: banking77 +0.0266 [+0.0071, +0.0448] and CLINC150 +0.0160 [+0.0022, +0.0289] against
  AWQ fast. It is worse on AG News (-0.0203 [-0.0271, -0.0142]) and BoolQ (-0.0232 [-0.0379,
  -0.0098]). The knowledge questions are tied (GPQA +0.040, LEXam +0.010, MMLU-Pro +0.001; every
  interval includes 0). Before the prompt fix Gemma was worse there (LEXam -0.081, MMLU-Pro -0.016).
  Against Jev it is lower on all 9 benchmarks, most on GPQA (-0.24) and MMLU-Pro (-0.23).
- Gemma's raw probabilities are strongly overconfident. Its NLL is 0.37 above AWQ fast's on
  `holdout`. It is higher even on banking77, where Gemma is more accurate (+0.366). One temperature
  per signature removes this (NLL from 1.270 to 0.833), and calibrated it is tied with AWQ fast
  (-0.002 [-0.027, +0.023];
  [calibration.md](calibration.md#current-results-one-call-per-question-holdout)).
- Over all 9 benchmarks Gemma runs at AWQ fast's speed and below GPTQ's: 20.44 against 19.70 and
  25.10 q/s on `holdout`. On the 6 shared benchmarks it runs at GPTQ's speed (52.33 against 53.17).
  On banking77 and CLINC150 it is the slowest of the three: 9.07 and 6.96 q/s, against 12.49 and
  9.22 (AWQ fast) and 13.29 and 9.54 (GPTQ). Its tokenizer makes these prompts longer: 729.9 and
  948.0 input tokens per question, against 581.6 and 827.0 for Qwen. Its p95 latency is 2,310
  against 1,758 ms (AWQ fast).
  - Multi-call history (before one-call scoring): Gemma was about 2x as fast as the Qwen MoEs
    over 9 (22.03 against 11.02 q/s on `holdout`). Its trie needed fewer calls per question
    (2.03 against 4.46 on banking77, 1.62 against 3.03 on CLINC150), and it got 118.9 cached
    input tokens per question from the prefix cache. With one call per question it gets 1.4
    [INFERENCE: the hits came from the later trie calls reusing the first call's prompt].
- The cost on `holdout` (`run_split.py stats`, 410 W) is 0.236 GPU-h against 0.244 (AWQ fast) and
  0.192 (GPTQ); $0.0015 against $0.0016 and $0.0012 per 1k items; $0.0020 against $0.0021 and
  $0.0017 per 1k correct answers.
- In the probe, 16 identical requests sent at once spread the top logprob by 0.117, against 0.0066
  (AWQ fast), 0.0044 (GPTQ) and 0.011 (Gemma 4 12B)
  ([vllm_probe_report.md](vllm_probe_report.md#gemma-4-26b-a4b-awq-int4-preset-gemma-4-26b-a4b-int4-cyankiwi)).
  So Gemma's answers move more with what else is in flight [INFERENCE: batch-dependent Marlin MoE
  kernels; not isolated].
- The first (multi-call) runs took about 36 min of GPU time on Sep 26: the sweep ran from 17:59 to
  18:33 UTC (runner time 128 + 788 + 793 s, three server starts of 98 to 101 s), plus a first load
  check (104 s) and the probe.

Gemma 4 26B-A4B is tied with the AWQ Qwen MoE on `holdout` (+0.0045) at about the same speed over
all 9 benchmarks, and as fast as the GPTQ build on the 6 shared ones. It is below AWQ fast on the
screen and on `select`. It is better on banking77 and CLINC150, and worse on AG News and BoolQ. Its
raw probabilities are much more overconfident than the Qwen builds' (ECE 0.165 against 0.077 to
0.087); calibrated, its NLL is tied with AWQ fast's. The AWQ Qwen build stays the
`qwen3.6-35b-a3b-fast` preset. Gemma has its own calibration registry,
[`calibration/gemma-4-26b-a4b-int4-cyankiwi/registry.json`](../../calibration/gemma-4-26b-a4b-int4-cyankiwi/registry.json).
