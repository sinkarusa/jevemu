# OpenAI probe report: gpt-6-luna

This probe checks whether GPT-6 Luna (`gpt-6-luna`) can be scored from the logprobs of OpenAI's
Chat Completions API, and it sets the options of `jevemu.backends.openai_chat`. Logprobs work
with reasoning off, but the API returns at most 5 alternatives per token and does not continue
an assistant prefill. So benchmarks with more than 5 options (MMLU-Pro, banking77, CLINC150)
cannot run on OpenAI.

The runs read the answer through Structured Outputs: the reply must be
`{"answer": "<label>"}` with the label from a JSON-schema `enum`, and the label distribution is
read at the token where the value starts. The mask is reflected in the logprobs, so that token's
list holds only labels, and the result is the model's distribution over the labels given that it
answers. Identical calls change the top answer on about 2% of items this way (4% for the earlier
free first-token read).

Measured with `scripts/probe_chat.py` against `POST https://api.openai.com/v1/chat/completions`:
the general probe on 2026-09-25 (`runs/probe_openai/report.json`), the structured read on
2026-09-27 (`runs/probe_openai_structured/report.json`). Both reports hold aggregates only, no
question text. Re-run them if the model, the API or the account tier changes:

```bash
uv run --with tiktoken python scripts/probe_chat.py --system openai --out runs/probe_openai --max-usd 1
uv run python scripts/probe_chat.py --system openai --sections structured \
  --prompt-cache-mode explicit --compare-run runs/archive_noprefill/select/gpt-6-luna.first_token.state_first \
  --out runs/probe_openai_structured --max-usd 0.5
```

## Setup

| Item | Value |
| --- | --- |
| Model | `gpt-6-luna` (every response reported exactly this id; no dated snapshot) |
| `system_fingerprint` | `null` on every probe response |
| Request | `reasoning_effort="none"`, `temperature=0`, `logprobs=true`, `top_logprobs=5`, `max_completion_tokens=16`, `prompt_cache_options.mode="explicit"`; prompts end in the user turn (`state_first-noprefill-5d29289f8b87`) |
| Structured read | `response_format: {"type": "json_schema", "json_schema": {"name": "answer", "strict": true, "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": [labels]}}, "required": ["answer"], "additionalProperties": false}}}` |
| Rate limits (headers) | 500 requests and 200,000 tokens per minute, for the account used here |
| Items | General probe: 50 `screen` items from each of 8 benchmarks. Structured read: 50 `screen` items from each of the 6 benchmarks Luna runs |
| Spend | $0.025 for the general probe (1,165 billed requests); $0.0091 for the structured read (400 requests) |

## Results

### Logprobs with reasoning off

Logprobs work with reasoning off, within these limits:

- `top_logprobs` is capped at 5, although the API reference says 0..20. Asking for 6 or
  more gives HTTP 400 `Invalid value for 'top_logprobs': must be less than or equal to 5.`
- `max_completion_tokens=1` gives HTTP 400
  `Could not finish the message because max_tokens or model output limit was reached`. With 16,
  a one-letter free answer stops on its own and bills 4 completion tokens; a structured reply
  bills 11. None of them are reasoning tokens.
- Values are float32: a near-certain token can come back at `+3.8e-06`. The parser clamps it
  to 0.

### Fewer alternatives than requested

The API cuts the list by probability, so it can hold fewer entries than requested. With 5
requested, an easy multiple-choice question (MCQ) returned only `B` at 0.0. A review between 3
and 4 stars returned `3` (-0.237) and `4` (-1.631), 0.985 of the mass. Under the structured read
the value's list usually holds one or two labels (see the table below).

### Structured Outputs schema root

Structured Outputs takes only object roots. A schema whose root is a string is refused with HTTP
400 (`schema must be a JSON Schema of 'type: "object"'`). The prompt does not need to mention JSON
(that is required only for `json_object` mode), so the user message is unchanged.

### Structured reply tokens

The value is a token of its own. Every reply was `{"answer":"<label>"}` in the same 5 tokens:
`{"`, `answer`, `":"`, the label, `"}`. The three tokens before the value came back at logprob 0.
The backend locates the value from the reply text and also handles a value that shares a token
with its opening quote (`":"B`); on Luna that never happened.

### Structured Outputs mask

The mask is reflected in the logprobs. At the value token, every listed alternative was a label:
no other token appeared in any of the 300 lists, and the listed labels held 0.994 to 0.9996 of
the mass on average. The leftover can only belong to the labels not listed, so each missing
label is bounded by it. An earlier check ("Capital of France?" with
`enum: ["London", "Berlin"]`) gave `London` -0.026 and `Berlin` -3.68, mass 0.9999995, and no
`Paris`.

### Prefill

The API does not continue a prefill. A final assistant message `The capital of France is` got a
new turn as the reply: `The capital of France is Paris.` (top first tokens: `The` -0.54, `Paris`
-0.88).

### Free first-token surfaces

Without a schema, all label mass sat on bare tokens (`A`, `3`, `Yes`). No top list held a
space-prefixed or decorated label token (`**A`, `(A`). Every greedy token was a label.

### Tokenizer

`tiktoken` 0.14.0 (latest) has no mapping for this model: `encoding_for_model("gpt-6-luna")`
raises `KeyError`. On all 400 prompts, though, the billed `prompt_tokens` minus the `o200k_base`
count of the message was exactly 6, so `o200k_base` matches. Under `o200k_base`, letters,
`Yes`/`No` and bare digits are one token each; `" 9"` is not.

### Nondeterminism

Identical calls can return different logprobs and a different top answer:

- Structured read: 100 ARC and AG News items sent again (cache bypassed) changed their top
  answer in 2 of 100.
- Free first-token read (2026-09-25): 50 items, each sent 5 more times and compared with the
  cached call, using label distributions as scoring reads them and total variation (TV)
  distance. Sequential: TV mean 0.049, p95 0.20, max 0.64; 30% of repeats identical, 54% within
  TV 0.01; the argmax flipped in 10 of 250. Concurrent (5 at once): TV mean 0.050, p95 0.26,
  max 0.54; 11 of 250 flips. The top token's logprob moved by up to 2.0 nats.

### Empty replies

One fresh call out of about 1,000 returned `content: ""`, no token logprobs,
`finish_reason: "stop"` and 3 billed completion tokens. The backend discards such a reply, does
not cache it, and sends the request again (at most 3 times). No structured call returned one.

### Prompt caching

Caching is implicit by default. A 1,530-token prompt reported `cache_write_tokens: 1527` on the
first call ($0.000193) and `cached_tokens: 1527` on the second ($0.0000176). Chat Completions
accepts `prompt_cache_options: {"mode": "explicit"}`, and then writes nothing (both calls
$0.000155). Logprobs were identical on the cache write and the cache hit (max abs difference
0.0).

### Temperature

Logprobs did not change between `temperature` 0 and 1 (max abs difference 0.0).

### Structured read per benchmark

50 `screen` items each, the run's exact requests. Every reply located its value (50 of 50), took
its likeliest label, and billed 0 reasoning tokens. "Label mass" is the summed probability of the
labels listed at the value token. "Agrees" compares the argmax with the free first-token read of
the same items in `runs/archive_noprefill/select/gpt-6-luna.first_token.state_first`.

| Benchmark | Labels | Label mass mean | p05 | min | Labels missing per item | Output tokens | $ per 1k items | Agrees with free read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpqa_diamond_idk | 5 | 0.9949 | 0.9824 | 0.9805 | 2.56 | 11 | 0.038 | 34/50 |
| lexam_en_idk | 5 | 0.9942 | 0.9814 | 0.9801 | 3.22 | 11 | 0.031 | 43/50 |
| arc_challenge | 4 | 0.9990 | 0.9891 | 0.9808 | 2.92 | 11 | 0.018 | 47/50 |
| ag_news | 4 | 0.9994 | 0.9969 | 0.9859 | 2.88 | 11 | 0.019 | 47/50 |
| boolq | 2 | 0.9996 | 0.9966 | 0.9931 | 0.90 | 11 | 0.023 | 48/50 |
| sst5 | 5 | 0.9976 | 0.9870 | 0.9809 | 3.24 | 11 | 0.017 | 38/50 |

On ARC and AG News the two reads disagree on 3 of 50 items each, about what two noisy reads
(2% and 4% flip rates) give. On GPQA and SST-5 they disagree far more than those rates explain:
there the JSON answer context moves the model's choice on hard or borderline items.

### Free first-token label mass per benchmark (2026-09-25)

Observed mass is the summed probability of the label tokens in the returned list (at most 5
entries).

| Benchmark | Labels | Observed mass mean | p05 | min | Share >= 0.99 | Labels missing per item |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpqa_diamond_idk | 5 | 0.9957 | 0.9840 | 0.9818 | 0.86 | 1.84 |
| lexam_en_idk | 5 | 0.9925 | 0.9822 | 0.9813 | 0.58 | 2.66 |
| mmlu_pro | 10 | 0.9891 | 0.9583 | 0.9016 | 0.58 | 7.42 |
| arc_challenge | 4 | 0.9996 | 0.9985 | 0.9871 | 0.98 | 2.90 |
| ag_news | 4 | 0.9988 | 0.9862 | 0.9826 | 0.94 | 2.96 |
| boolq | 2 | 0.9990 | 0.9934 | 0.9869 | 0.98 | 0.88 |
| sst5 | 5 | 0.9962 | 0.9839 | 0.9812 | 0.84 | 3.12 |

## Adapter decisions

- Runs use the structured read: `ConstrainedStrategy` (`run_split.py --strategy constrained`,
  run directory `gpt-6-luna.constrained.state_first`, manifest flag `constraint: json_schema`).
  `Capabilities`: `structured_choice=True`, `mask_reflected_in_logprobs=True`,
  `assistant_prefill=False`, `local_tokenizer=False`, `top_logprobs_exact=False`;
  `TOP_LOGPROBS_MAX = 5`, `MAX_COMPLETION_TOKENS = 16`. The prompt and its `template_id` are the
  free read's; the strategy and the constraint tell the runs apart. The free first-token read
  (`--strategy first_token`) still works; the earlier runs used it.
- A response that bills reasoning tokens, or whose list at the value is not that position's own
  (it must list the generated token at its own logprob), is rejected and not cached.
- A label missing from the value's list gets the leftover mass `log(1 - sum p)` (floored at
  float32 epsilon) and the item is marked truncated. Under the mask the leftover belongs only to
  the missing labels, so this is a true bound. The free read takes the smaller of that and the
  smallest returned logprob.
- MMLU-Pro (up to 10 options), banking77 (77) and CLINC150 (150) cannot run on OpenAI models.
  With at most 5 top logprobs, a question with more than 5 options cannot show every label, and
  without prefill there is no other way to score it (no echo, no trie).
  `run_split.py --system openai` skips such benchmarks; Luna's macro average covers the other 6.
- Runs use `prompt_cache_options.mode="explicit"` (`run_split.py --prompt-cache-mode explicit`).
  Each benchmark item is sent once, so in the default mode the cache writes are a 25% surcharge
  on prompts of 1,024+ tokens that are never read back.
- A structured call bills 11 output tokens instead of 4, about $0.02 to $0.04 per 1,000 items.
- About 2% of items change their top answer between identical structured calls. Read paired
  accuracy differences against Jev with that noise floor in mind.
