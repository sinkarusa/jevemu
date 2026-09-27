# vLLM probe report

This probe checks what the pinned vLLM 0.30.0 server returns for logprob scoring, first on
`Qwen/Qwen3-0.6B` and then on the Qwen3.8-27B AWQ INT4 preset. Everything the adapter needs
works: constrained decoding is enforced, the returned logprobs are already renormalized over the
allowed tokens, the top-logprob cap is a hard 64, and assistant prefill is honored. Two findings
change the design: echo requests never read the prefix cache, and reported logprobs move with
batch composition (and, on Qwen3-0.6B in bf16, with prefix-cache hits).

Measured on 2026-09-24 with `scripts/probe_vllm.py` against the pinned Docker server. The probe
answers the design risk "Unknown pre- vs post-mask logprobs" and the open question about
explicit-token logprobs over HTTP. Re-run it whenever the image, model or flags change. Results
are cached under `~/.cache/jevemu/probes/`, keyed by vLLM version, model, revision, image,
effective chat-template kwargs and declared flags.

The sections up to "Adapter decisions" are for `Qwen/Qwen3-0.6B`;
[Qwen3.8-27B](#qwen38-27b-awq-int4-preset-qwen38-27b-awq) and
[Gemma 4 26B-A4B](#gemma-4-26b-a4b-awq-int4-preset-gemma-4-26b-a4b-int4-cyankiwi) have their own
sections below. Strategy names follow the design doc: S1 reads the first token's top logprobs
over the labels, S2 does the same under a `choice` constraint, S3 walks a token trie, and S4
(echo) scores each option's text from prompt logprobs.

## Setup

| Item | Value |
| --- | --- |
| vLLM | 0.30.0 (`GET /version`) |
| Image | `vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` |
| Model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, dtype `auto` (bf16) |
| GPU | RTX 3090 24 GB, driver 580.173.02 |
| Launch | `vllm serve Qwen/Qwen3-0.6B --revision=main --max-logprobs=64 --logprobs-mode=raw_logprobs --enable-prefix-caching --enable-prompt-tokens-details --max-model-len=8192 --seed=0 --dtype=auto --gpu-memory-utilization=0.85` (`docker/vllm/compose.yaml`) |
| Discovered (`/metrics` `vllm:cache_config_info`, `/v1/models`) | `enable_prefix_caching=True`, `block_size=16`, `cache_dtype=auto`, `max_model_len=8192` |

The probe ran three times: the primary launch above, the same launch with
`--logprobs-mode=processed_logprobs`, and the same launch with `--dtype=float32`.

## Results

| Probe | Result (primary run) | processed_logprobs | float32 |
| --- | --- | --- | --- |
| a. Constraint enforced | **Yes.** `choice=["Q","Z"]` on a "which is a fruit, A or B" prompt generated `"Z"`; unconstrained greedy token was `" **"` | Yes | Yes |
| b. Mask reflected in logprobs | **Yes.** Constrained top-64: only `"A"`/`"B"` finite, 62/64 entries at the -9999 floor, allowed mass 1.000000 | Yes (identical values) | Yes |
| c. `--max-logprobs 64` | `top_logprobs=64` → HTTP 200 with 64 entries; `65` → **HTTP 400** `Requested sample logprobs of 65, which is greater than max allowed: 64 (parameter=logprobs, value=65)`. No silent truncation | same | same |
| d. Label tokens | `A..Z`, `a..z`, `Yes`, `No`: one token bare and with a leading space. `0..9`: one token bare, **two tokens with a leading space** (`" 0"` = `[220, 15]`) | — | — |
| e. Echo | `max_tokens=0` accepted, no generated token appended; first prompt logprob `null`; an identical second echo request reported `cached_tokens=0` | same | same |
| f. Prefill | Honored. `/tokenize` render ends `…assistant\n<think>\n\n</think>\n\nThe capital of France is`; chat `prompt_tokens` = `/tokenize` count = 24 | same | same |
| Prefix cache | Enabled (metrics); a repeated chat prompt reported `cached_tokens=32` of 46 | same | same |
| Explicit token ids | Server supports `logprob_token_ids`: requesting ids of `"A"`/`"B"` returned the sampled token plus both labels (`"B"` -21.22, `"A"` -32.04) | same | same |

Measured capabilities (all three runs): `top_logprobs_max=64, structured_choice=True,
mask_reflected_in_logprobs=True, echo_prompt_logprobs=True, explicit_token_logprobs=False,
assistant_prefill=True, prefix_caching=True`. `explicit_token_logprobs` stays False because the
adapter has no explicit-token method yet; the server already supports it.

### b. Mask evidence

Same prompt (`system`: answer with the letter; `user`: "Which of these is a fruit?\nA) Hammer\nB)
Apple"; prefill `Answer:`), `top_logprobs=64`:

| Request | Top entries (token, logprob as reported) |
| --- | --- |
| Unconstrained | `" **"` -0.283, `" B"` -1.408, `" $\\"` -6.783, `" $"` -8.783, `""` -9.283, `" A"` -10.283, `" *"` -11.908, `" \\"` -12.408 |
| `structured_outputs={"choice": ["A","B"]}` | `"B"` -2.0e-05, `"A"` -10.820, then 62 arbitrary tokens (`"!"`, `"\""`, `"1"`, `"2"`, ...) at -9999.0 |

vLLM applies the grammar bitmask to the logits before the sampler computes `raw_logprobs`, so
the reported logprobs are already renormalized over the allowed tokens. With greedy decoding
(`temperature=0`), `processed_logprobs` returns the same numbers. Masked tokens still fill the
top-k list at the -9999 floor; the adapter maps them to `-inf`.

What this means for scoring: under S2 the constrained distribution is exact (mass 1 over the
allowed first tokens). But a constrained call's `observed_mass` is then always about 1, so it
says nothing about what the model wanted. The unconstrained call above puts only
exp(-1.408) + exp(-10.283) ≈ 0.24 on letter tokens, because the model wants to write markdown
`**B**`. S1 on this model would therefore fire the `observed_mass < 0.5` warning. Measuring the
unconstrained mass needs a second call.

### d. Label tokenization

Qwen3's tokenizer splits digits one by one, so `" 7"` is `[" ", "7"]`. After an `Answer:`
prefill, an unconstrained model spends its first token on the space. Score labels (digits)
therefore need either the constraint (S2, which emits the bare digit right after `Answer:`) or a
prefill that ends in a space. Letters and `Yes`/`No` work either way.

Bare ids: `A..Z` = 32..57, `a..z` = 64..89, `0..9` = 15..24, `Yes` = 9454, `No` = 2753. Spaced:
`" A"` = 362, `" B"` = 425, `" Yes"` = 7414, `" No"` = 2308.

### e. Echo details

- `echo=true, logprobs=1, max_tokens=0` works. vLLM generates one token internally and drops it
  (`usage.completion_tokens=1`, but `logprobs.tokens` has exactly the prompt length).
- The first prompt position has `token_logprobs[0] = null`. The adapter never scores it, because
  the continuation always follows the chat-template prefix.
- No -9999 values appeared in echo responses. The floor was seen only for masked tokens in
  constrained chat top-k lists.
- **Echo never reads the prefix cache.** vLLM 0.30.0 sets `skip_reading_prefix_cache` for any
  request with prompt logprobs (`SamplingParams`: "the output of prompt logprobs may be less than
  n_prompt_tokens"). An identical repeat confirmed it (`cached_tokens=0`). So the design's "one
  request per option shares a cached prefix" does not hold: S4 costs K full prefills, one per
  option.
- Responses also carry vLLM-specific `prompt_logprobs` (per position: token id, logprob, rank,
  decoded token) and `prompt_token_ids`.
- Adapter-level totals after `The capital of France is`: `" Paris"` -2.936, `" London"` -9.992,
  `" Paris, the city of light."` -19.389.

### Determinism (not in the design's risk list)

In bf16, reported logprobs depend on prefix-cache hits and on batch composition:

| Measurement | bf16 (primary) | float32 |
| --- | --- | --- |
| Echo vs first-token logprob of `" Paris"` on a fresh prompt | 0.0 | 2.4e-07 |
| Same prompt uncached vs cached (probe, 48 cached tokens), max \|Δ\| over top-20 | 0.125 | 0.005 |
| 20 prompts (55–187 tokens, 48–176 cached), max \|Δ\| per prompt: median / max | 0.128 / 0.242 | 0.0045 / 0.010 |
| 16 concurrent identical chat requests, spread of the top-1 logprob (3 prompts) | 0.061, 0.076, 0.027 | 0.0006, 0.0030, 0.0035 |
| 8 concurrent identical echo requests, spread of the total (3 prompts) | 0.156, 0.054, 0.029 | 0.0020, 0.0008, 0.0052 |

Sequential repeats of a cached request are bit-identical. The probe's own 16-way spread was only
0.0009 (bf16), because all copies hit the cache the same way; the dedicated runs above are the
better estimate.

Implications:

- A question's probabilities can move by about 0.1 nat depending on what else is in flight or
  cached. That matters for paired comparisons and temperature fits at n ≈ 200.
- `--dtype float32` cuts the noise about 25× and fits Qwen3-0.6B easily. It does not fit a 7–8B
  model on 24 GB. Larger models need either the noise accounted for, or prefix caching disabled
  for reproducibility runs (batch noise remains).

## Adapter decisions driven by these results

- Constraint: send a top-level `structured_outputs={"choice": [...]}`, never a `guided_*` field.
  Verified live: the same prompt with `guided_choice=["Q","Z"]` returned HTTP 200 and `" **"`,
  and the server logged "Request contains the removed guided-decoding field(s)
  ['guided_choice'], which are ignored; output will NOT be constrained."
- `top_logprobs` is clamped to the probed cap. Going over the cap is an HTTP 400, not a silent
  truncation.
- Token ids: logprobs return either text or `"token_id:N"` placeholders
  (`return_tokens_as_token_ids`), never both. The completions top-k is a map, and keyed by text
  it keeps only one of several tokens that decode alike: Gemma 4's byte tokens (`<0x41>` next
  to `A`, `<0x31>` next to `1`) are both allowed by a choice constraint, and the text map kept
  the byte token's value (`"B"` -19.75 where the real `B` token had -0.00). The adapter
  therefore requests ids on `/v1/completions` and resolves each id's text by `/detokenize`
  (cached). Until 2026-09-27 it requested text on the whitespace-prefill path (`Answer: `
  before digits), which read Gemma's digit labels wrong.
- Echo: the adapter builds the prompt as token ids: `/tokenize` renders the conversation
  before the prefill with the generation prompt (`add_generation_prompt=true,
  add_special_tokens=false`), and the prefill and continuation are appended as plain-text
  tokens. It sends the ids to `/v1/completions` with `max_tokens=0` and finds the continuation
  tokens by the prefix token count. (Until 2026-09-27 the prefill was rendered by the template
  as a continued assistant message, `continue_final_message=true`; see the design's "Assistant
  prefill" row.)
- `cached_tokens` needs `--enable-prompt-tokens-details`. Without it the field is absent and
  `NextTokenDist.cached_tokens` is `None`.
- The model revision is not exposed over HTTP. `scripts/serve_vllm.sh` resolves it from the
  shared Hugging Face cache and exports `JEVEMU_VLLM_MODEL_REVISION`. The backend reports it in
  `health()`, with the image from `JEVEMU_VLLM_IMAGE`.
- Chat-template defaults (`--default-chat-template-kwargs`, e.g. thinking off) are set on the
  server and invisible over HTTP. `serve_vllm.sh` exports them as
  `JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS`. The backend reports the effective kwargs (server
  defaults merged with per-request ones) as `health().flags["chat-template-kwargs"]` and keys the
  probe cache on them.
- A prefill that ends in whitespace (`Answer: ` before Score digits) is never left to the chat
  template, which may trim it (Qwen3.8's does; see below). Instead, `/tokenize` renders the
  prompt without the whitespace, the whitespace's own tokens are appended, and first-token calls
  send the ids to `/v1/completions` (same sampling and `structured_outputs`). On Qwen3-0.6B,
  whose template keeps the space, the ids are identical to the chat render.

## Qwen3.8-27B (AWQ INT4, preset `qwen3.8-27b-awq`)

Measured on 2026-09-24/25 against `scripts/serve_vllm.sh --preset qwen3.8-27b-awq`.

| Item | Value |
| --- | --- |
| Model | `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision `6e134bae811fb5adac50ee042ae5f029ac6779aa`: community compressed-tensors W4A16 quant of `Qwen/Qwen3.8-27B` (hybrid: 48 Gated DeltaNet + 16 gated-attention layers, 248,320-token vocab); Marlin kernels on Ampere. The official bf16 (51.7 GiB) and FP8 (~28 GiB) checkpoints do not fit 24 GB |
| Launch | `vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 --revision=6e134bae… --max-logprobs=64 --logprobs-mode=raw_logprobs --enable-prefix-caching --enable-prompt-tokens-details --max-model-len=8192 --seed=0 --dtype=auto --gpu-memory-utilization=0.95 --default-chat-template-kwargs={"enable_thinking":false} --language-model-only --max-num-seqs=16 --max-num-batched-tokens=1024` |
| Discovered (`/metrics`) | `block_size=784` (attention pages sized to the Mamba state), `mamba_cache_mode=align`, `mamba_ssm_cache_dtype=float32`, `num_gpu_blocks=27`, `cache_dtype=auto`, `max_model_len=8192` |
| Memory | Weights 18.37 GiB (text only); KV cache 1.33 GiB = 13,010 tokens (1.59 requests of 8,192 tokens); CUDA graphs 0.2 GiB. `nvidia-smi`: 20.7 GiB right after start, 22.2 GiB peak during 4 concurrent 4,877-token echo requests, 23.6 GiB held by the allocator after a stress run (3 × 7,731-token echo + 24 echo + 16 chat requests at once: no errors) |
| Startup | 126 s from `serve_vllm.sh` to healthy (weight loading 11–15 s, torch.compile 24 s, profiling/warm-up 41 s, graph capture 6 s) |

Fit: at `--gpu-memory-utilization 0.93` the server failed to start. It had 0.85 GiB for the KV
cache, and one 8,192-token request needs 0.81 GiB plus Mamba pages (estimated maximum length
7,840). 0.95 serves the full 8,192.

`--max-num-batched-tokens 1024` bounds the memory spike from echo's prompt logprobs. vLLM
computes full-vocabulary logits and log-softmax for every prompt token in the chunk
(1,024 × 248,320 in fp32, about 1.4 GiB peak measured), and the startup memory profile does not
cover this.

The GPU is headless; a display would need a lower utilization and a shorter `--max-model-len`.

| Probe | Qwen3.8-27B AWQ | Delta vs Qwen3-0.6B |
| --- | --- | --- |
| a. Constraint enforced | Yes: `choice=["Q","Z"]` → `"Q"`; unconstrained `" B"` (correct) | Same; 0.6B wanted `" **"` |
| b. Mask reflected | Yes: constrained top-64 `"B"` -2.4e-05, `"A"` -10.64, 62/64 at -9999, allowed mass 1.000000. Unconstrained top-1 `" B"` -0.008 (letters ≈ 0.99 of the mass) | Same mechanism; S1 mass is ~1 on this model, not 0.24 |
| c. `--max-logprobs 64` | 64 → HTTP 200 with 64 entries; 65 → HTTP 400 | Same |
| d. Label tokens | `A..Z`, `a..z`, `Yes`, `No`: one token bare and spaced; `0..9`: one token bare, two spaced (`" 7"` = `[220, 22]`). Bare ids `A..Z` = 32..57, `a..z` = 64..89, `0..9` = 15..24 (as Qwen3); `Yes` 9175, `No` 2665; spaced `" A"` 357, `" a"` 264, `" Yes"` 7179, `" No"` 2233; `" "` = 220 | Same single-token pattern, so the same reading plans (digits after the gap) |
| e. Echo | `max_tokens=0` accepted, first logprob `null`; echo never reads the prefix cache (identical 2,477-token echo repeated: `cached_tokens` 0 both times) but writes it (a following chat request: 2,352 cached) | Same |
| f. Prefill | Honored: render ends `…assistant\n<think>\n\n</think>\n\nThe capital of France is`, `/tokenize` 24 = chat 24 tokens. **The template trims the final assistant message**: `Answer: ` renders exactly like `Answer:` | 0.6B keeps the trailing space |
| Thinking | Server default `{"enable_thinking":false}` applies to chat and `/tokenize`: the decoded render keeps the system message as sent (no reasoning instruction) and ends in an empty `<think>\n\n</think>` block. A request override `enable_thinking=true` prepends "Reasoning effort is set to xhigh. …" to the system message (38 → 76 tokens), so the default is required | 0.6B's template renders the empty think block for a prefill either way |
| Prefix cache | Enabled, in 784-token blocks: prompts under 784 tokens never hit (the probe's short repeat: `cached_tokens=0`); repeats of 1,278 / 2,480 / 4,879-token prompts hit 784 / 2,352 / 4,704 | 0.6B caches 16-token blocks (32 of 46) |
| Explicit token ids | `logprob_token_ids` supported (`" B"` -0.0081, `"B"` -8.383, `"A"` -19.02) | Same |

Measured capabilities are identical to Qwen3-0.6B's (`top_logprobs_max=64`, constraint enforced,
mask reflected, echo, prefill, prefix caching).

**Score prefill.** Before the adapter fix, every Score first-token read (S1/S2/S3) on this model
saw the trimmed `…Answer:` and read the bare digit there, which is the distortion the design
avoids. Reads now happen after the true `Answer: ` gap. Over 10 Score prompts (5 questions × 2
layouts), they match echo of `" d"` within 0.016 (`question_first`) / 0.007 (`state_first`) per
level. The trimmed read was off by up to 0.08 / 0.175 (review 0–9, `state_first`: `7` 0.461 vs
0.286 by echo).

Determinism (same measurements as the Qwen3-0.6B table; prompts are unique per run):

| Measurement | Qwen3.8-27B AWQ | Qwen3-0.6B bf16 |
| --- | --- | --- |
| Echo vs first-token logprob of `" Paris"`, fresh prompt | 4.8e-07 | 0.0 |
| Uncached vs cached, 12 prompts (1,878–5,172 tokens, 1,568–4,704 cached), max \|Δ\| over top-20: median / max | 0.0000 / 0.0000 | 0.128 / 0.242 |
| 16 concurrent identical chat requests, spread of the top-1 logprob (3 prompts) | 0.0014, 0.0022, 0.0091 | 0.061, 0.076, 0.027 |
| 16 concurrent identical constrained requests, 4-option questions with split answers: max \|Δp\| of a label | 0.031, 0.028, 0.0002 | — |
| 8 concurrent identical echo requests, spread of the last-5-token total (3 prompts) | 0.42, 1.00, 0.36 | 0.156, 0.054, 0.029 |
| Same, per token (7,912 positions): median / p90 / p99 / max | 0 / 0.18 / 1.1 / 14.3 | — |

Sequential repeats are bit-identical, and a request that runs alone reproduces the reference
exactly; the spread comes from batch composition. Prefix-cache hits do not move logprobs on this
model.

Echo is the noisy path. The large per-token moves are on tokens far in the tail (logprob below
about -10). S4 sends its K option requests concurrently, so echo scores on this model carry about
0.1–1 nat of batch noise per option. First-token strategies (the default S2) move by at most
about 0.03 in probability.

## Gemma 4 26B-A4B (AWQ INT4, preset `gemma-4-26b-a4b-int4-cyankiwi`)

Measured on 2026-09-26 against `scripts/serve_vllm.sh --preset gemma-4-26b-a4b-int4-cyankiwi`
(`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` @ `18a3c7285c33ee39d3e5e16ee6fb2c18f4955ef9`, a MoE
with 3.8B of 25.2B parameters active; load facts in
[candidates.md](candidates.md#gemma-4-26b-a4b)). Prefix caching uses 16-token blocks.

| Probe | Gemma 4 26B-A4B AWQ | Delta vs Qwen3.8-27B |
| --- | --- | --- |
| a. Constraint enforced | Yes: `choice=["Q","Z"]` → `"Q"`; unconstrained `" B"` (correct) | Same |
| b. Mask reflected | Yes: constrained top-64 `"B"` -0.00015, `"A"` -8.84, 60/64 at -9999, allowed mass 1.000000. Unconstrained top-1 `" B"` -0.0029 | Same |
| c. `--max-logprobs 64` | 64 → HTTP 200; 65 → HTTP 400 | Same |
| d. Label tokens | Letters, `Yes`, `No`: one token bare and spaced. `0..9`: one token bare, two spaced (`" 0"` = `[236743, 236771]`) | Same pattern; the first-token reader reads digits after the gap |
| e. Echo | `max_tokens=0` accepted, first logprob `null`; a repeat reported `cached_tokens=0`. Totals after `The capital of France is`: `" Paris"` -1.502, `" London"` -13.469 | Same |
| f. Prefill | Honored: render ends `<|turn>model\nThe capital of France is`, `/tokenize` 21 = chat 21 tokens. No thinking tokens with `enable_thinking=false` | Gemma template; no empty think block |
| Prefix cache | Enabled; a repeated chat prompt got 32 cached tokens | 16-token blocks, so short prompts hit |
| Explicit token ids | `logprob_token_ids` supported (`" B"` -0.0030, `"B"` -12.94, `"A"` -21.69) | Same |

Determinism (the probe's own measurements; `probe.py` `DeterminismProbe`):

| Measurement | Gemma 4 26B-A4B AWQ | Gemma 4 12B QAT | Qwen3.6-35B-A3B AWQ (fast) | Qwen3.6-35B-A3B GPTQ |
| --- | --- | --- | --- | --- |
| Echo vs first-token logprob of `" Paris"`, fresh prompt | 3.0e-06 | 0.0 | 0.124 | 0.249 |
| Repeat of the first-token request: cached tokens, max \|Δ\| over shared top-k | 32, 0.125 | 0, 0.0 | 0, 0.248 | 0, 0.313 |
| 16 concurrent identical requests, spread of the top-1 logprob | **0.117** | 0.011 | 0.0066 | 0.0044 |

The batch spread is 10-27x the other presets'. So on this model a question's probabilities move
more with what else is in flight [INFERENCE: batch-dependent Marlin MoE kernels; not isolated].
Echo and uncached first-token logprobs agree (gap 3.0e-06).

## Reproduce

```bash
eval "$(scripts/serve_vllm.sh --preset qwen3-0.6b | grep '^export ')"   # or qwen3.8-27b-awq
uv run python scripts/probe_vllm.py --refresh --json-out /tmp/probe.json
JEVEMU_VLLM_LOGPROBS_MODE=processed_logprobs scripts/serve_vllm.sh      # variant runs (0.6B)
JEVEMU_VLLM_DTYPE=float32 scripts/serve_vllm.sh
docker compose -f docker/vllm/compose.yaml down
```
