# Quantization report: Qwen3.5-9B bf16 vs FP8, INT8 and INT4

This report tests whether serving Qwen3.5-9B quantized (FP8, INT8 or INT4) changes the
emulator's accuracy or its reported probabilities by more than the bf16 model's own run-to-run
noise. Accuracy does not change measurably at any bit width. 8-bit keeps probabilities at or near
the noise floor. Both 4-bit builds move each item's probabilities well beyond it, and
cyankiwi's INT4, the scheme of the 27B preset, also costs log-likelihood.

Measured on 2026-09-25 with `scripts/quant_compare.py` (vLLM 0.30.0, RTX 3090). The question:
does serving a quantized checkpoint change (a) the emulator's accuracy or (b) the label
probabilities and logprobs it reports, by more than the bf16 model's own run-to-run noise?

Why the 9B: the top rung of the model ladder, Qwen3.8-27B, only fits a 24 GB GPU as a community
INT4 quant (`qwen3.8-27b-awq`); its bf16 weights (51.7 GiB) cannot be served here. Qwen3.5-9B has
the same hybrid architecture (Gated DeltaNet plus gated attention, 248k vocabulary) and its bf16
checkpoint fits, so its quantization ladder stands in for the 27B's.

## Conclusion

Numbers are for S2 (constrained first-token letters) on both banks (817 items) unless noted. The
"floor" is the `bf16-restart` run: the same bf16 server started again. `bf16-c16` (concurrency
16) gives about the same values.

Terms: NLL is negative log-likelihood of the gold answer, KL is Kullback-Leibler divergence from
bf16, TV is total variation distance, MAE is mean absolute error, CI is confidence interval,
ECE is expected calibration error, and Brier is the Brier score (see Metrics below).

| Run vs `bf16` | Δ accuracy [95% CI] | Δ NLL [95% CI] | KL mean (nats) | TV mean | Mean \|Δ p(gold)\| | Raw logprob MAE (nats) | Top-1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor: `bf16-restart` | −0.002 [−0.013, +0.005] | +0.001 [−0.002, +0.004] | 0.0009 | 0.014 | 0.007 | 0.031 | 0.969 |
| `int8` | +0.007 [−0.004, +0.017] | −0.001 [−0.004, +0.003] | 0.0011 | 0.016 | 0.008 | 0.037 | 0.957 |
| `fp8` | −0.002 [−0.018, +0.011] | **+0.006 [+0.001, +0.011]** | 0.0024 | 0.026 | 0.012 | 0.060 | 0.928 |
| `int4-quanttrio` | −0.006 [−0.032, +0.018] | −0.007 [−0.021, +0.008] | 0.021 | 0.081 | 0.039 | 0.191 | 0.750 |
| `int4-cyankiwi` | −0.012 [−0.037, +0.010] | **+0.052 [+0.035, +0.069]** | 0.030 | 0.091 | 0.043 | 0.216 | 0.781 |

1. **Accuracy: no measurable change at any bit width.** For every quantized run, the 95%
   interval of the accuracy difference includes zero on every slice (both banks, each bank,
   echo). QuantTrio's echo interval just touches zero: +0.061 [0, +0.121]. No McNemar test
   reaches p < 0.05. The resolution is about ±2-3 points. Qwen3.5-9B without chain-of-thought is
   close to chance here (accuracy 0.28, mean p(gold) 0.255), so these banks cannot detect small
   accuracy losses. Noise alone can look significant: the concurrency-16 bf16 run gains +0.025
   [+0.005, +0.051] on GPQA (5 items gained, 0 lost).
2. **8-bit: probabilities stay at or near the noise floor.** INT8 (W8A16, Marlin) cannot be told
   apart from a bf16 restart on any probability metric, nor on NLL and Brier. FP8 moves
   probabilities about 2-3x the floor (KL 0.0024 vs 0.0009), with a small but significant NLL
   increase (+0.006 nats, 0.4%). On this Ampere GPU FP8 runs weight-only, and its kernel is not
   deterministic (below).
3. **4-bit: probabilities move well beyond noise; accuracy does not.** Both INT4 checkpoints
   move each item's distribution by 6-7x the floor in TV (0.08-0.09) and 23-33x in KL. They move
   the raw label logprobs by about 0.2 nats (floor 0.03). The top answer flips on 22-27% of items
   (floor 3%). On the Jev doc examples, a Score's expected value moves by 0.10 on average (up to
   0.49 for QuantTrio and 0.61 for cyankiwi; floor 0.009, up to 0.026). A Noul's P(yes) moves by
   0.03-0.05 (up to 0.10 and 0.22; floor 0.002).
4. **QuantTrio's INT4 is the closer one.**
   - Against bf16 it has lower KL than cyankiwi's (0.021 vs 0.030 on both banks; 0.019 vs 0.042
     on GPQA), lower TV and raw-logprob error, and no proper-score cost (NLL −0.007 [−0.021,
     +0.008], Brier −0.001).
   - cyankiwi's INT4 is significantly worse in NLL (+0.052, 3.5%) and Brier (+0.017 [+0.009,
     +0.025]), and more overconfident on GPQA (mean top probability 0.421 vs 0.397, ECE 0.133 vs
     0.097).
   - What each quantizes: QuantTrio keeps attention, DeltaNet and layer 0 in bf16 (only MLPs are
     4-bit, group 128). cyankiwi also quantizes attention and most DeltaNet projections (group
     32).
   - QuantTrio's checkpoint makes vLLM run in fp16. Served with bf16 activations
     (`int4-quanttrio-bf16`) it measures the same (KL 0.0216 vs 0.0214), so the difference is the
     weights.
   - QuantTrio flips slightly more top answers (0.750 vs 0.781 agreement): its per-item
     perturbations are unbiased, not smaller on every item.
5. **What this means for the 27B-AWQ emulator [INFERENCE].** `qwen3.8-27b-awq` uses cyankiwi's
   scheme (compressed-tensors W4A16, group 32, Marlin). If the 27B responds like the 9B, its
   aggregate accuracy is within a few points of bf16 Qwen3.8-27B. But each question's
   probabilities differ from bf16's by roughly 0.1 in TV and 0.2 nats per label logprob, with a
   small systematic NLL penalty. So: use its accuracy as representative, calibrate on its own
   outputs (calibrators are fitted per model anyway), and never read its per-item probabilities
   as bf16 Qwen3.8-27B's. Larger models usually lose less to 4-bit weights, so the 9B gap is
   plausibly an upper bound; this was not measured. On the same 817 items the 27B-AWQ rung is
   the stronger model (accuracy 0.368 vs 0.284 for 9B bf16, NLL 1.452 vs 1.475, ECE 0.044 vs
   0.100), so its 4-bit perturbation does not erase the size gain.

**Noise-floor findings.** These hold for any comparison on this stack, not only quantization.

- **A container restart alone changes almost every item.** Two bf16 servers with identical
  flags agree bit-for-bit on 1 of 817 items (KL 0.0009, TV 0.014, raw logprob MAE 0.03 nats, 3%
  of top answers flipped). FP8 did the same (first attempt vs rerun: 3 of 817 identical, KL
  0.0009, TV 0.013). Within one server, 8 sequential repeats of the same constrained and echo
  request (5 LEXam items) were bit-identical on bf16, so the variation is fixed at startup. The
  likely cause is autotuned Triton kernels picking different tile configurations per process:
  the Gated DeltaNet prefill uses flash-linear-attention ops decorated with `triton.autotune`
  [INFERENCE]. Concurrency 16 adds no more than a restart does.
- **FP8 on Ampere is not repeatable within a server.** The same 8 sequential repeats gave 8
  distinct results (constrained top-20 logprobs spread up to 0.125 nats, echo up to 0.10). In
  the runs, this shows up as echo's five per-option requests for one item disagreeing: only 13%
  of FP8 items stay on the logit grid below, against 98-100% for the other runs at concurrency 1
  (the fp16 QuantTrio run on its own, finer grid).
- **Label logprob differences lie on the bf16 logit grid.** Under S2, for 99-100% of items in
  every 9B run with bf16 activations, the differences between label logprobs are multiples of
  0.125 nats (the bf16 spacing of logits between 16 and 32). So probabilities move in discrete
  steps. The fp16 QuantTrio run sits on a 1/64-nat grid instead.

**Serving notes.** All five 9B checkpoints loaded on the first try with the shared flags, with no
kernel fallbacks beyond FP8's weight-only path. Load facts per run are in the setup table below.
The `qwen3.8-27b-awq` preset lacks `--generation-config=vllm`, so vLLM warns that the
checkpoint's `temperature=1.0, top_k=20, top_p=0.95` replace its sampling defaults. The
emulator's numbers are unaffected: it sends `temperature=0` and reads `raw_logprobs`, which are
taken before top-k/top-p.

## Method

**Checkpoints** (all Qwen3.5-9B with the same chat template; prompt token counts are identical
to `bf16` on every item; revisions pinned in `docker/vllm/presets/qwen3.5-9b-*.env`):

| Run | Checkpoint | Format | Quantized | Kept in bf16 | Snapshot |
| --- | --- | --- | --- | --- | --- |
| `bf16` | `Qwen/Qwen3.5-9B` | bf16 (official) | nothing | everything | 18.0 GiB |
| `fp8` | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | compressed-tensors FP8, per-channel weights, dynamic per-token activations; on SM 8.6 vLLM runs it weight-only (W8A16) | MLPs and gated-attention projections, all 32 layers | DeltaNet projections, `lm_head`, embeddings | 13.0 GiB |
| `int8` | `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8` | compressed-tensors W8A16, symmetric, group 32 | same as `fp8` | same as `fp8` | 13.2 GiB |
| `int4-cyankiwi` | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | compressed-tensors W4A16, symmetric, group 32 (the quantizer of the 27B preset) | MLPs, gated attention, DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj` | DeltaNet `in_proj_a`/`in_proj_b`, `lm_head`, embeddings | 8.4 GiB |
| `int4-quanttrio` | `QuantTrio/Qwen3.5-9B-AWQ` | AutoAWQ GEMM INT4, group 128, zero point; the config declares float16, so `--dtype auto` serves it in fp16 | MLPs of layers 1-31 | attention, DeltaNet, layer 0, `lm_head`, embeddings | 11.5 GiB |
| `int4-quanttrio-bf16` | same | same, served with `--dtype bfloat16` | same | same | same |

The 9B presets share every launch flag: 0.90 of the GPU, thinking off by server default,
`--language-model-only`, `--max-num-batched-tokens=1024`, and `--generation-config=vllm` (so no
checkpoint's sampling defaults apply). Only the model, the revision and, for
`int4-quanttrio-bf16`, the dtype differ.

The FP8 linear kernel is not logged. In vLLM 0.30.0 the first CUDA candidate for W8A16 FP8 is
`HummingFP8ScaledMMLinearKernel` (supported from SM 7.5; `humming` is installed in the image),
ahead of Marlin, and Marlin's "weight-only FP8" warning never appears. So the FP8 run used
Humming [INFERENCE from the image's source].

**Protocol.** Each run gets a new container, so every run starts with an empty prefix cache. The
emulator (unrounded probabilities, diagnostics on) then answers, in this order:

1. the same 817 bank items (GPQA-Diamond 198 + LEXam-en 619; seed 0, "I don't know" option E,
   evaluate-idk option order) with S2 constrained letters, the default `auto` choice for these
   questions;
2. every Choice, Score and Noul question of the 21 Jev doc request fixtures (41 questions), with
   S2;
3. the 198 GPQA items with S4 echo (`mode="sum"`), which scores each option key from prompt
   logprobs.

Only one HTTP request is in flight at a time, so batch composition cannot move logprobs. For
each item the runner stores the renormalized probabilities, the raw label logprobs before
renormalization, the observed label mass, the answer, the gold key, and the prompt and cached
tokens (`runs/quant/<ts>/<run>.jsonl`, gitignored).

Every run was checked to render thinking off (`<think>\n\n</think>` after the generation prompt)
and to keep a trailing prefill space. The Qwen3.5 template trims `Answer: ` to `Answer:` like
Qwen3.8's, and the adapter appends the space token. Unlike Qwen3.8's, the Qwen3.5 template adds
no reasoning-effort instruction when thinking is on. After an assistant prefill it renders the
empty think block either way (checked on the bf16 server with `enable_thinking=true`).

**Noise floors.** Two extra bf16 runs show what "no change" looks like:

- `bf16-restart` repeats the reference on a new container (same flags, same order,
  concurrency 1).
- `bf16-c16` repeats it with 16 questions and 16 HTTP requests in flight, where batch
  composition changes the kernels' reduction order.

A quantization delta matters only if it clears these floors. The within-server repeat check
above was a separate manual run (`fp8` and `bf16` presets, first 5 LEXam items, 8 sequential
repeats each of the constrained request and of the echo of `" A"`).

**Metrics** (`src/jevemu/eval/metrics.py`; definitions from the design's "Paired comparison
harness"):

- Accuracy counts E as wrong.
- NLL = −log max(p(gold), 1e-6).
- Brier score: multiclass, summed over A-E.
- ECE: top-label, over 10 equal-mass bins (the design default) and 10 equal-width bins.
- KL(bf16 || run) in nats, after 1e-12 additive smoothing.
- Total variation.
- Paired bootstrap with 10,000 resamples (seed 0; percentile for GPQA's 198 items, BCa from 500
  items).
- Exact McNemar test on per-item correctness.

Echo scores the option keys (`" A"`..`" E"`), not the option texts. With letter keys, S4 reads
the unconstrained letter logprobs through the prompt-logprob path, and its observed mass (0.90 on
bf16) is the probability the model puts on a letter at all.

## Results

<!-- quant_compare:begin -->

<!-- Generated by `uv run python scripts/quant_compare.py analyze runs/quant/20260925T010117Z`; edit the prose outside this block. -->

### Setup

- Data: `runs/quant/20260925T010117Z` (gitignored; per-item outputs stay there).
- vLLM `0.30.0`, image `vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB, driver 580.173.02.
- `bf16` launch (the other 9B runs change only the model, revision or dtype): `vllm serve Qwen/Qwen3.5-9B --revision=c202236235762e1c871ad0ccb60c8ee5ba337b9a --port=8000 --max-logprobs=64 --logprobs-mode=raw_logprobs --enable-prefix-caching --enable-prompt-tokens-details --max-model-len=8192 --seed=0 --dtype=auto --gpu-memory-utilization=0.90 --default-chat-template-kwargs={"enable_thinking":false} --language-model-only --max-num-seqs=16 --max-num-batched-tokens=1024 --generation-config=vllm`.
- Banks: seed 0, IDK option, evaluate-idk order; `gpqa_diamond` 198 items (sha256 `b32d8024a59f`), `lexam_en` 619 items (sha256 `65e2cb91a688`).
- Doc examples: every question of the 21 Jev doc request fixtures.

| Run | What | Checkpoint @ revision | Quantization: linear kernel (vLLM log) | Weights GiB | KV cache | Startup s | Concurrency | ms/question S2 / S4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | bf16 (reference) | `Qwen/Qwen3.5-9B` @ `c202236235` | none (bf16) | 16.80 | 2.28 GiB, 52,503 tokens | 95 | 1 | 79 / 619 |
| `bf16-restart` | bf16, new container | `Qwen/Qwen3.5-9B` @ `c202236235` | none (bf16) | 16.80 | 2.28 GiB, 52,503 tokens | 95 | 1 | 79 / 621 |
| `bf16-c16` | bf16, concurrency 16 | `Qwen/Qwen3.5-9B` @ `c202236235` | none (bf16) | 16.80 | 2.28 GiB, 52,503 tokens | 95 | 16 | 67 (wall / n) / 429 (wall / n) |
| `fp8` | FP8 (RedHatAI) | `RedHatAI/Qwen3.5-9B-FP8-dynamic` @ `790f0576d2` | `compressed-tensors`: no kernel line logged | 11.88 | 7.18 GiB, 165,701 tokens | 113 | 1 | 72 / 481 |
| `int8` | INT8 g32 (cyankiwi) | `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8` @ `c798ade7cb` | `compressed-tensors`: MarlinLinearKernel (CompressedTensorsWNA16) | 12.19 | 6.84 GiB, 157,882 tokens | 104 | 1 | 80 / 529 |
| `int4-cyankiwi` | INT4 g32 (cyankiwi) | `cyankiwi/Qwen3.5-9B-AWQ-4bit` @ `156edc4bbe` | `compressed-tensors`: MarlinLinearKernel (CompressedTensorsWNA16) | 7.55 | 11.51 GiB, 265,867 tokens | 104 | 1 | 69 / 454 |
| `int4-quanttrio` | INT4 AWQ g128 (QuantTrio), fp16 activations (checkpoint dtype) | `QuantTrio/Qwen3.5-9B-AWQ` @ `938f8e3ef8` | `auto_awq`: MarlinLinearKernel (AutoAWQMarlinLinearMethod) | 10.35 | 8.71 GiB, 201,076 tokens | 104 | 1 | 73 / 460 |
| `int4-quanttrio-bf16` | INT4 AWQ g128 (QuantTrio), bf16 activations | `QuantTrio/Qwen3.5-9B-AWQ` @ `938f8e3ef8` | `auto_awq`: MarlinLinearKernel (AutoAWQMarlinLinearMethod) | 10.35 | 8.72 GiB, 201,076 tokens | 104 | 1 | 73 / 500 |
| `27b-awq` | Qwen3.8-27B INT4 (cyankiwi) | `cyankiwi/Qwen3.8-27B-AWQ-INT4` @ `6e134bae81` | `compressed-tensors`: MarlinLinearKernel (CompressedTensorsWNA16) | 18.37 | 1.33 GiB, 13,010 tokens | 128 | 1 | 208 / 1244 |

### Accuracy and calibration

Accuracy counts IDK (E) as wrong; A-D accuracy excludes items answered E. NLL clips p(gold) at 1e-6. Brier is the multiclass sum over A-E. ECE is top-label with 10 equal-mass bins (design default) and 10 equal-width bins; confidence is the top-label probability.

**GPQA-Diamond, constrained (S2)**

| Run | n | Accuracy | A-D accuracy | IDK rate | Mean p(gold) | NLL | Brier | ECE (mass) | ECE (width) | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | 198 | 0.318 | 0.320 | 0.005 | 0.277 | 1.437 | 0.737 | 0.097 | 0.093 | 0.397 |
| `bf16-restart` | 198 | 0.323 | 0.325 | 0.005 | 0.277 | 1.437 | 0.736 | 0.121 | 0.091 | 0.397 |
| `bf16-c16` | 198 | 0.343 | 0.345 | 0.005 | 0.277 | 1.433 | 0.735 | 0.089 | 0.065 | 0.395 |
| `fp8` | 198 | 0.323 | 0.325 | 0.005 | 0.276 | 1.439 | 0.738 | 0.115 | 0.093 | 0.396 |
| `int8` | 198 | 0.333 | 0.333 | 0.000 | 0.278 | 1.435 | 0.735 | 0.088 | 0.078 | 0.397 |
| `int4-cyankiwi` | 198 | 0.293 | 0.293 | 0.000 | 0.273 | 1.479 | 0.759 | 0.133 | 0.134 | 0.421 |
| `int4-quanttrio` | 198 | 0.323 | 0.328 | 0.015 | 0.278 | 1.421 | 0.732 | 0.102 | 0.071 | 0.390 |
| `int4-quanttrio-bf16` | 198 | 0.328 | 0.332 | 0.010 | 0.279 | 1.419 | 0.731 | 0.090 | 0.077 | 0.391 |
| `27b-awq` | 198 | 0.348 | 0.390 | 0.106 | 0.283 | 1.464 | 0.740 | 0.124 | 0.088 | 0.419 |

**LEXam-en, constrained (S2)**

| Run | n | Accuracy | A-D accuracy | IDK rate | Mean p(gold) | NLL | Brier | ECE (mass) | ECE (width) | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | 619 | 0.273 | 0.273 | 0.000 | 0.248 | 1.487 | 0.774 | 0.112 | 0.105 | 0.378 |
| `bf16-restart` | 619 | 0.268 | 0.268 | 0.000 | 0.248 | 1.487 | 0.774 | 0.111 | 0.110 | 0.378 |
| `bf16-c16` | 619 | 0.271 | 0.271 | 0.000 | 0.247 | 1.489 | 0.775 | 0.116 | 0.111 | 0.379 |
| `fp8` | 619 | 0.268 | 0.268 | 0.000 | 0.246 | 1.494 | 0.776 | 0.114 | 0.114 | 0.380 |
| `int8` | 619 | 0.278 | 0.278 | 0.000 | 0.248 | 1.486 | 0.773 | 0.102 | 0.099 | 0.377 |
| `int4-cyankiwi` | 619 | 0.265 | 0.266 | 0.003 | 0.238 | 1.542 | 0.790 | 0.117 | 0.111 | 0.376 |
| `int4-quanttrio` | 619 | 0.263 | 0.263 | 0.000 | 0.250 | 1.483 | 0.774 | 0.117 | 0.109 | 0.373 |
| `int4-quanttrio-bf16` | 619 | 0.279 | 0.279 | 0.000 | 0.249 | 1.483 | 0.775 | 0.097 | 0.094 | 0.373 |
| `27b-awq` | 619 | 0.375 | 0.378 | 0.010 | 0.270 | 1.448 | 0.737 | 0.041 | 0.024 | 0.379 |

**Both banks, constrained (S2)**

| Run | n | Accuracy | A-D accuracy | IDK rate | Mean p(gold) | NLL | Brier | ECE (mass) | ECE (width) | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | 817 | 0.284 | 0.284 | 0.001 | 0.255 | 1.475 | 0.765 | 0.100 | 0.100 | 0.383 |
| `bf16-restart` | 817 | 0.282 | 0.282 | 0.001 | 0.255 | 1.475 | 0.765 | 0.107 | 0.103 | 0.383 |
| `bf16-c16` | 817 | 0.289 | 0.289 | 0.001 | 0.255 | 1.476 | 0.765 | 0.103 | 0.096 | 0.383 |
| `fp8` | 817 | 0.282 | 0.282 | 0.001 | 0.253 | 1.480 | 0.767 | 0.103 | 0.105 | 0.384 |
| `int8` | 817 | 0.291 | 0.291 | 0.000 | 0.255 | 1.474 | 0.764 | 0.090 | 0.093 | 0.382 |
| `int4-cyankiwi` | 817 | 0.272 | 0.272 | 0.002 | 0.246 | 1.527 | 0.782 | 0.115 | 0.116 | 0.387 |
| `int4-quanttrio` | 817 | 0.278 | 0.279 | 0.004 | 0.256 | 1.468 | 0.764 | 0.104 | 0.099 | 0.377 |
| `int4-quanttrio-bf16` | 817 | 0.291 | 0.292 | 0.002 | 0.257 | 1.468 | 0.764 | 0.093 | 0.086 | 0.377 |
| `27b-awq` | 817 | 0.368 | 0.381 | 0.033 | 0.273 | 1.452 | 0.738 | 0.044 | 0.027 | 0.389 |

**GPQA-Diamond, echo sum (S4)**

| Run | n | Accuracy | A-D accuracy | IDK rate | Mean p(gold) | NLL | Brier | ECE (mass) | ECE (width) | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16` | 198 | 0.288 | 0.291 | 0.010 | 0.273 | 1.445 | 0.747 | 0.132 | 0.136 | 0.401 |
| `bf16-restart` | 198 | 0.293 | 0.296 | 0.010 | 0.275 | 1.442 | 0.745 | 0.129 | 0.144 | 0.400 |
| `bf16-c16` | 198 | 0.293 | 0.296 | 0.010 | 0.275 | 1.439 | 0.744 | 0.118 | 0.121 | 0.399 |
| `fp8` | 198 | 0.293 | 0.296 | 0.010 | 0.272 | 1.447 | 0.748 | 0.162 | 0.135 | 0.397 |
| `int8` | 198 | 0.288 | 0.291 | 0.010 | 0.274 | 1.441 | 0.744 | 0.118 | 0.121 | 0.399 |
| `int4-cyankiwi` | 198 | 0.278 | 0.279 | 0.005 | 0.278 | 1.458 | 0.756 | 0.146 | 0.149 | 0.424 |
| `int4-quanttrio` | 198 | 0.348 | 0.356 | 0.020 | 0.274 | 1.444 | 0.745 | 0.089 | 0.072 | 0.397 |
| `int4-quanttrio-bf16` | 198 | 0.328 | 0.335 | 0.020 | 0.274 | 1.447 | 0.747 | 0.094 | 0.077 | 0.398 |
| `27b-awq` | 198 | 0.328 | 0.346 | 0.051 | 0.300 | 1.410 | 0.724 | 0.144 | 0.128 | 0.434 |

### Jev doc examples (constrained), paired with bf16

| Run | Choice (n=14): agreement, TV mean / max | Score (n=13): \|Δ expected score\| mean / max | Noul (n=14): \|Δ P(yes)\| mean / max |
| --- | --- | --- | --- |
| `bf16-restart` | 1.000, 0.0059 / 0.0312 | 0.0090 / 0.0261 | 0.0021 / 0.0224 |
| `bf16-c16` | 1.000, 0.0064 / 0.0311 | 0.0132 / 0.0302 | 0.0008 / 0.0060 |
| `fp8` | 1.000, 0.0196 / 0.0585 | 0.0118 / 0.0568 | 0.0088 / 0.0298 |
| `int8` | 1.000, 0.0112 / 0.0397 | 0.0138 / 0.0320 | 0.0037 / 0.0253 |
| `int4-cyankiwi` | 1.000, 0.0489 / 0.1419 | 0.1080 / 0.6144 | 0.0492 / 0.2151 |
| `int4-quanttrio` | 1.000, 0.0446 / 0.1105 | 0.1051 / 0.4869 | 0.0285 / 0.0995 |
| `int4-quanttrio-bf16` | 1.000, 0.0473 / 0.1085 | 0.1029 / 0.4621 | 0.0312 / 0.0951 |

### Paired with bf16: accuracy and proper scores

Differences are run minus `bf16` over the same items, with 95% paired-bootstrap intervals (10,000 resamples, seed 0; percentile below 500 items, BCa from 500). McNemar is exact; b/c = items only this run / only `bf16` got right.

**GPQA-Diamond, constrained (S2)**

| Run | Δ accuracy [95% CI] | McNemar p (b/c) | Δ NLL [95% CI] | Δ Brier [95% CI] | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| `bf16-restart` | +0.005 [-0.010, +0.025] | 1 (2/1) | +4.9e-04 [-0.006, +0.007] | -3.0e-04 [-0.003, +0.003] | 0.965 |
| `bf16-c16` | +0.025 [+0.005, +0.051] | 0.062 (5/0) | -0.003 [-0.010, +0.004] | -0.002 [-0.005, +0.002] | 0.944 |
| `fp8` | +0.005 [-0.025, +0.035] | 1 (5/4) | +0.003 [-0.008, +0.013] | +9.4e-04 [-0.004, +0.006] | 0.904 |
| `int8` | +0.015 [-0.005, +0.040] | 0.375 (4/1) | -0.002 [-0.009, +0.006] | -0.002 [-0.005, +0.001] | 0.939 |
| `int4-cyankiwi` | -0.025 [-0.086, +0.035] | 0.522 (17/22) | +0.043 [+0.004, +0.082] | +0.023 [+0.002, +0.043] | 0.697 |
| `int4-quanttrio` | +0.005 [-0.045, +0.056] | 1 (13/12) | -0.016 [-0.044, +0.012] | -0.005 [-0.019, +0.010] | 0.742 |
| `int4-quanttrio-bf16` | +0.010 [-0.040, +0.061] | 0.845 (14/12) | -0.018 [-0.045, +0.010] | -0.006 [-0.020, +0.008] | 0.737 |

**LEXam-en, constrained (S2)**

| Run | Δ accuracy [95% CI] | McNemar p (b/c) | Δ NLL [95% CI] | Δ Brier [95% CI] | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| `bf16-restart` | -0.005 [-0.018, +0.003] | 0.549 (4/7) | +7.2e-04 [-0.003, +0.004] | +3.9e-05 [-0.002, +0.002] | 0.971 |
| `bf16-c16` | -0.002 [-0.015, +0.008] | 1 (6/7) | +0.002 [-9.6e-04, +0.005] | +9.6e-04 [-6.1e-04, +0.003] | 0.964 |
| `fp8` | -0.005 [-0.023, +0.010] | 0.701 (12/15) | +0.007 [+0.002, +0.012] | +0.003 [+1.7e-04, +0.005] | 0.935 |
| `int8` | +0.005 [-0.008, +0.015] | 0.581 (8/5) | -4.4e-04 [-0.004, +0.003] | -6.9e-04 [-0.002, +0.001] | 0.963 |
| `int4-cyankiwi` | -0.008 [-0.034, +0.015] | 0.603 (27/32) | +0.055 [+0.037, +0.074] | +0.016 [+0.007, +0.024] | 0.808 |
| `int4-quanttrio` | -0.010 [-0.040, +0.018] | 0.586 (39/45) | -0.004 [-0.020, +0.013] | +9.1e-05 [-0.008, +0.008] | 0.753 |
| `int4-quanttrio-bf16` | +0.006 [-0.026, +0.036] | 0.755 (48/44) | -0.003 [-0.019, +0.013] | +7.6e-04 [-0.007, +0.009] | 0.729 |

**Both banks, constrained (S2)**

| Run | Δ accuracy [95% CI] | McNemar p (b/c) | Δ NLL [95% CI] | Δ Brier [95% CI] | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| `bf16-restart` | -0.002 [-0.013, +0.005] | 0.791 (6/8) | +6.6e-04 [-0.002, +0.004] | -4.4e-05 [-0.002, +0.001] | 0.969 |
| `bf16-c16` | +0.005 [-0.006, +0.015] | 0.481 (11/7) | +1.0e-03 [-0.002, +0.004] | +3.2e-04 [-0.001, +0.002] | 0.960 |
| `fp8` | -0.002 [-0.018, +0.011] | 0.868 (17/19) | +0.006 [+0.001, +0.011] | +0.002 [-1.8e-05, +0.004] | 0.928 |
| `int8` | +0.007 [-0.004, +0.017] | 0.238 (12/6) | -7.6e-04 [-0.004, +0.003] | -0.001 [-0.003, +5.2e-04] | 0.957 |
| `int4-cyankiwi` | -0.012 [-0.037, +0.010] | 0.363 (44/54) | +0.052 [+0.035, +0.069] | +0.017 [+0.009, +0.025] | 0.781 |
| `int4-quanttrio` | -0.006 [-0.032, +0.018] | 0.702 (52/57) | -0.007 [-0.021, +0.008] | -0.001 [-0.008, +0.006] | 0.750 |
| `int4-quanttrio-bf16` | +0.007 [-0.020, +0.032] | 0.645 (62/56) | -0.007 [-0.021, +0.008] | -8.6e-04 [-0.008, +0.006] | 0.731 |

**GPQA-Diamond, echo sum (S4)**

| Run | Δ accuracy [95% CI] | McNemar p (b/c) | Δ NLL [95% CI] | Δ Brier [95% CI] | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| `bf16-restart` | +0.005 [-0.015, +0.025] | 1 (3/2) | -0.003 [-0.009, +0.003] | -0.002 [-0.005, +8.1e-04] | 0.955 |
| `bf16-c16` | +0.005 [-0.020, +0.030] | 1 (4/3) | -0.006 [-0.011, +1.9e-04] | -0.003 [-0.006, +9.9e-05] | 0.934 |
| `fp8` | +0.005 [-0.020, +0.030] | 1 (4/3) | +0.002 [-0.010, +0.014] | +7.4e-04 [-0.005, +0.007] | 0.924 |
| `int8` | 0 [-0.030, +0.030] | 1 (5/5) | -0.003 [-0.011, +0.004] | -0.003 [-0.006, +7.0e-04] | 0.934 |
| `int4-cyankiwi` | -0.010 [-0.061, +0.040] | 0.845 (12/14) | +0.013 [-0.025, +0.051] | +0.009 [-0.012, +0.030] | 0.783 |
| `int4-quanttrio` | +0.061 [0, +0.121] | 0.065 (24/12) | -5.9e-04 [-0.033, +0.032] | -0.002 [-0.019, +0.015] | 0.687 |
| `int4-quanttrio-bf16` | +0.040 [-0.015, +0.096] | 0.200 (19/11) | +0.003 [-0.029, +0.035] | -3.5e-04 [-0.017, +0.016] | 0.727 |

### Paired with bf16: probabilities

Per item over the A-E distribution: KL(bf16 || run) in nats, total variation, |Δ p(gold)|. Raw label logprobs are before renormalization (constrained: the merged `A` and ` A` surfaces; echo: the ` A`..` E` continuation totals); observed mass is the label mass before renormalization.

**GPQA-Diamond, constrained (S2)**

| Run | KL mean | KL median | KL p95 | KL max | TV mean | Mean \|Δ p(gold)\| | Raw logprob MAE | Δ observed mass (mean / abs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16-restart` | 0.0010 | 0.0010 | 0.0019 | 0.0052 | 0.0146 | 0.0074 | 0.0334 | +3.9e-05 / 0.0002 |
| `bf16-c16` | 0.0010 | 0.0008 | 0.0024 | 0.0063 | 0.0143 | 0.0073 | 0.0327 | +1.2e-05 / 0.0002 |
| `fp8` | 0.0031 | 0.0021 | 0.0073 | 0.0239 | 0.0297 | 0.0142 | 0.0733 | -0.0005 / 0.0007 |
| `int8` | 0.0013 | 0.0012 | 0.0032 | 0.0063 | 0.0174 | 0.0084 | 0.0414 | +1.4e-05 / 0.0002 |
| `int4-cyankiwi` | 0.0422 | 0.0287 | 0.1193 | 0.3981 | 0.1094 | 0.0541 | 0.2459 | -2.6e-05 / 0.0017 |
| `int4-quanttrio` | 0.0188 | 0.0132 | 0.0515 | 0.1053 | 0.0737 | 0.0389 | 0.1761 | -0.0003 / 0.0011 |
| `int4-quanttrio-bf16` | 0.0188 | 0.0132 | 0.0512 | 0.1004 | 0.0736 | 0.0383 | 0.1778 | -0.0003 / 0.0012 |

**LEXam-en, constrained (S2)**

| Run | KL mean | KL median | KL p95 | KL max | TV mean | Mean \|Δ p(gold)\| | Raw logprob MAE | Δ observed mass (mean / abs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16-restart` | 0.0009 | 0.0008 | 0.0019 | 0.0048 | 0.0134 | 0.0065 | 0.0297 | -2.6e-07 / 2.3e-05 |
| `bf16-c16` | 0.0008 | 0.0007 | 0.0019 | 0.0050 | 0.0130 | 0.0062 | 0.0289 | -1.7e-06 / 2.6e-05 |
| `fp8` | 0.0021 | 0.0018 | 0.0055 | 0.0236 | 0.0249 | 0.0115 | 0.0563 | +4.0e-06 / 4.5e-05 |
| `int8` | 0.0010 | 0.0011 | 0.0020 | 0.0052 | 0.0157 | 0.0075 | 0.0352 | -1.2e-05 / 2.9e-05 |
| `int4-cyankiwi` | 0.0258 | 0.0206 | 0.0664 | 0.1430 | 0.0857 | 0.0395 | 0.2067 | -0.0004 / 0.0004 |
| `int4-quanttrio` | 0.0222 | 0.0177 | 0.0527 | 0.2502 | 0.0831 | 0.0395 | 0.1962 | -0.0002 / 0.0002 |
| `int4-quanttrio-bf16` | 0.0225 | 0.0184 | 0.0547 | 0.2489 | 0.0833 | 0.0396 | 0.1984 | -0.0002 / 0.0002 |

**Both banks, constrained (S2)**

| Run | KL mean | KL median | KL p95 | KL max | TV mean | Mean \|Δ p(gold)\| | Raw logprob MAE | Δ observed mass (mean / abs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16-restart` | 0.0009 | 0.0009 | 0.0019 | 0.0052 | 0.0137 | 0.0067 | 0.0306 | +9.1e-06 / 7.0e-05 |
| `bf16-c16` | 0.0009 | 0.0007 | 0.0020 | 0.0063 | 0.0133 | 0.0065 | 0.0298 | +1.5e-06 / 7.5e-05 |
| `fp8` | 0.0024 | 0.0019 | 0.0059 | 0.0239 | 0.0261 | 0.0121 | 0.0604 | -0.0001 / 0.0002 |
| `int8` | 0.0011 | 0.0012 | 0.0021 | 0.0063 | 0.0161 | 0.0077 | 0.0367 | -5.8e-06 / 7.2e-05 |
| `int4-cyankiwi` | 0.0297 | 0.0221 | 0.0759 | 0.3981 | 0.0914 | 0.0430 | 0.2162 | -0.0003 / 0.0008 |
| `int4-quanttrio` | 0.0214 | 0.0169 | 0.0527 | 0.2502 | 0.0808 | 0.0394 | 0.1914 | -0.0002 / 0.0005 |
| `int4-quanttrio-bf16` | 0.0216 | 0.0170 | 0.0545 | 0.2489 | 0.0809 | 0.0393 | 0.1934 | -0.0003 / 0.0005 |

**GPQA-Diamond, echo sum (S4)**

| Run | KL mean | KL median | KL p95 | KL max | TV mean | Mean \|Δ p(gold)\| | Raw logprob MAE | Δ observed mass (mean / abs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16-restart` | 0.0009 | 0.0008 | 0.0019 | 0.0079 | 0.0136 | 0.0068 | 0.0329 | +0.0003 / 0.0031 |
| `bf16-c16` | 0.0009 | 0.0008 | 0.0020 | 0.0047 | 0.0139 | 0.0069 | 0.0335 | +0.0002 / 0.0060 |
| `fp8` | 0.0039 | 0.0024 | 0.0109 | 0.0448 | 0.0321 | 0.0158 | 0.0782 | -0.0098 / 0.0145 |
| `int8` | 0.0012 | 0.0012 | 0.0028 | 0.0070 | 0.0171 | 0.0088 | 0.0415 | -0.0016 / 0.0040 |
| `int4-cyankiwi` | 0.0410 | 0.0259 | 0.1202 | 0.3803 | 0.1047 | 0.0506 | 0.2592 | +0.0007 / 0.0219 |
| `int4-quanttrio` | 0.0250 | 0.0172 | 0.0725 | 0.3769 | 0.0828 | 0.0409 | 0.2026 | +0.0055 / 0.0234 |
| `int4-quanttrio-bf16` | 0.0250 | 0.0168 | 0.0819 | 0.3596 | 0.0822 | 0.0402 | 0.2041 | +0.0061 / 0.0238 |

### Checks

- Prompt token counts equal to `bf16` on every bank item (same template and tokenizer): `bf16-restart`, `bf16-c16`, `fp8`, `int8`, `int4-cyankiwi`, `int4-quanttrio`, `int4-quanttrio-bf16`
- `bf16`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.998, S4 0.995 of items
- `bf16-restart`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.998, S4 1.000 of items
- `bf16-c16`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.996, S4 0.480 of items
- `fp8`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.995, S4 0.126 of items
- `int8`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.998, S4 0.995 of items
- `int4-cyankiwi`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 1.000, S4 1.000 of items
- `int4-quanttrio`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.000, S4 0.000 of items
- `int4-quanttrio-bf16`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.993, S4 0.980 of items
- `27b-awq`: thinking off True; template trims `Answer: ` True, adapter keeps the space True; probed top-logprobs cap 64, mask reflected True; prefix-cache hits 0 tokens; label logprob differences on the 0.125-nat grid: S2 0.985, S4 0.944 of items

<!-- quant_compare:end -->

## Reproduce

```bash
uv run python scripts/quant_compare.py run                        # all 9 runs, ~50 min; resume with --out
uv run python scripts/quant_compare.py analyze runs/quant/<ts>    # rewrites the generated block
```

The generated block comes from `runs/quant/20260925T010117Z`. Its `fp8` run is a second attempt,
made to keep the full server log. The first attempt (`first-fp8/` there) is the FP8 restart pair
quoted in the noise-floor findings. The within-server repeat check is not part of the script.
