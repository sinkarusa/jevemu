# OpenRouter probe report: DeepSeek V4.1 Flash

This probe checks whether DeepSeek V4.1 Flash can be scored from logprobs through OpenRouter,
with the requests `jevemu.backends.openrouter_chat` sends. Logprobs work: 20 alternatives per
token, reasoning off, no prefill.

The runs read the answer through Structured Outputs: the reply must be
`{"answer": "<label>"}` with the label from a JSON-schema `enum`, and the label distribution is
read at the token where the value starts. That needs a provider whose top lists belong to the
position they are reported at. Makora passes: its lists are fresh, the mask is reflected (masked
tokens at -9999), it answers greedily at temperature 0, and the listed labels hold all the mass.
Wafer, the provider of the earlier free first-token runs, fails because its lists at the value
are often another position's. The structured runs are therefore pinned to Makora.

Measured with `scripts/probe_chat.py --system openrouter` and throwaway scripts against
`POST https://openrouter.ai/api/v1/chat/completions`: Wafer on 2026-09-25 (free read,
`runs/probe_openrouter/report.json`) and 2026-09-27 (structured), the other providers and Makora
on 2026-09-27 (`runs/probe_openrouter_structured/report.json`). Reports hold aggregates only, no
question text. Re-run the probe when the model, the provider or the settings change:

```bash
uv run python scripts/probe_chat.py --system openrouter --provider makora \
  --sections basic,structured \
  --compare-run "runs/archive_noprefill/select/deepseek-v4.1-flash@wafer.first_token.state_first" \
  --out runs/probe_openrouter_structured --max-usd 0.5
```

## Setup

| Item | Value |
| --- | --- |
| Model | `deepseek/deepseek-v4.1-flash` (every response reported exactly this id; the endpoints listing names the canonical slug `deepseek/deepseek-v4.1-flash-20260910`) |
| Provider | Pinned: `provider: {"require_parameters": true, "order": ["makora"], "allow_fallbacks": false}`; every response reported `provider: "Makora"` (earlier runs: `"wafer"`, every response `"Wafer"`) |
| `system_fingerprint` | `null` on every response |
| Request | `reasoning: {"enabled": false}`, `temperature=0`, `logprobs=true`, `top_logprobs=20`, `max_tokens=16`; prompts end in the user turn (`state_first`, no prefill) |
| Structured read | `response_format: {"type": "json_schema", "json_schema": {"name": "answer", "strict": true, "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": [labels]}}, "required": ["answer"], "additionalProperties": false}}}` |
| Price snapshot | Makora, per 1M tokens: input $0.30, cached input $0.006, output $1.20. Wafer: input $0.099, cached input $0.06, output $0.60 |
| Items | Structured read: 50 `screen` items from each of the 7 benchmarks DeepSeek runs. Free read on Wafer: 50 from each of 8 |
| Spend | Structured probe on Makora: $0.034 in all. Provider screen: $0.0045. Wafer and Luna structured tests: under $0.002. Wafer free-read probe: $0.0162 (907 requests) |

## Results

### Logprobs with `require_parameters`

Logprobs work on Wafer and Makora. `top_logprobs=20` returns exactly 20 alternatives, all
finite; on an easy multiple-choice question (MCQ) they hold 1.0000000 of the mass.
`top_logprobs=21` is refused by both (Wafer: HTTP 400 `model_request_rejected`; Makora: HTTP 200
with an error body, "Requested sample logprobs of 21, which is greater than max allowed: 20"), so
the backend keeps 20.

### Reasoning off

Every response on both providers has `reasoning_tokens: 0`. The backend rejects any response
that bills reasoning tokens. No call returned an empty reply.

### Billing

`usage.cost` equals the snapshot price on both providers (Makora, 39-token MCQ: billed and
computed $1.41e-05).

### Prefill

Wafer does not continue a prefill, although OpenRouter documents prefill. A final assistant
message `The capital of France is` got a new turn: `The capital of France is **Paris**.`

### Which providers can do the structured read

A provider qualifies if it serves the model with `logprobs`, `top_logprobs` and
`response_format`, returns each position's own top list, and takes the likeliest token at
temperature 0. Tested with the object schema above on `screen` items:

| Provider | Result |
| --- | --- |
| Makora (fp8) | Passes. 40 of 40 replies had fresh lists at every position and took the top token at every position; 350 of 350 in the probe below |
| Wafer | Fails: the value token's list was a copy of an earlier position's in 31 of 40 replies (for example the value `D` carried the list of the `":` position); it also sometimes generated a token that was not its most likely one at temperature 0 |
| DekaLLM | Fails: stale lists at the value in 8 of 10 |
| DigitalOcean | Fails: took a token other than its most likely one at temperature 0 (2 of 7 located replies) |
| NextBit | Fails: every listed alternative at logprob 0, with special tokens among them |
| Together | Fails: logprobs for the first token only |
| CoreWeave, Sail Research | Fail: no token logprobs under a schema |
| Fireworks | Fails: HTTP 400 for the request |

### Wafer's stale lists

On Wafer, tokens generated in the same decoding step as the one before them come back with a
copy of an earlier position's list, so under the object schema the list at the value is often
not the value's. A bare string schema (`"B"`) kept the lists fresh, because the value starts in
the first or second token. But DeepSeek's tokenizer merges the opening quote with some labels
(`"A`, `"C`, `"D`, `"Yes`, `"No` are single tokens; `"B`, `"E` and `"0`..`"4` are not). When the
reply takes a merged token, the labels that follow only the bare `"` are not listed, so the read
would be partial and biased toward the merged labels. That read was rejected; the earlier Wafer
runs keep the free first-token read.

### Makora's structured reply

The reply is `{ "answer": "B" }` with varying whitespace, 7 to 14 completion tokens (mean 8 to 10
per benchmark). The value is a token of its own (`B`) after ` "`. The backend handles two quirks:

- Before SST-5 values the model often hesitates between whitespace tokens (` "` at 0.4 to 0.85,
  the rest on `\r`, `\t`, spaces). The read is normalized over the alternatives that start the
  value right there, so it is the model's label distribution given that the value starts where
  the reply's does.
- Tokens the grammar forces are missing from the logprobs although the message text has them
  (for example `{\n\n` then `answer`; `\t` then `4` for `\t"4`). 4 of 350 replies had such a
  gap, 1 of them the value's opening quote. The locator tolerates missing braces and quotes, and
  a value read after a forced quote is conditioned on it.

### Nondeterminism

On Makora, 100 ARC and AG News items sent again (cache bypassed) never changed their top answer
(0 of 100). On Wafer's free read, the top answer flipped in 6 of 250 sequential repeats and 4 of
250 concurrent ones (TV mean 0.022; the top token's logprob moved by up to 2.8 nats).

### Prompt caching (Wafer)

Caching is automatic. A 1,588-token prompt sent twice reported `cached_tokens: 1536` the second
time and cost $0.0000979 instead of $0.000158. There is no cache-write surcharge.

### Structured read per benchmark (Makora)

50 `screen` items each, the run's exact requests. Every reply located its value (50 of 50), its
list held only labels and masked entries (at -9999), the listed labels held at least 0.9999999
of the mass, it took its likeliest label, and it billed 0 reasoning tokens. No label was missing
on any item. "Agrees" compares the argmax with Wafer's free first-token read of the same items in
`runs/archive_noprefill/select/deepseek-v4.1-flash@wafer.first_token.state_first`.

| Benchmark | Labels | Label mass mean | Labels missing per item | Output tokens | $ per 1k items | Agrees with free read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpqa_diamond_idk | 5 | 1.0000 | 0 | 9.1 | 0.090 | 31/50 |
| lexam_en_idk | 5 | 1.0000 | 0 | 8.0 | 0.075 | 39/50 |
| mmlu_pro | 10 | 1.0000 | 0 | 9.1 | 0.100 | 45/50 |
| arc_challenge | 4 | 1.0000 | 0 | 8.6 | 0.037 | 50/50 |
| ag_news | 4 | 1.0000 | 0 | 8.8 | 0.039 | 50/50 |
| boolq | 2 | 1.0000 | 0 | 8.8 | 0.053 | 49/50 |
| sst5 | 5 | 1.0000 | 0 | 9.9 | 0.034 | 45/50 |

On ARC and AG News the two reads agree on every item. On GPQA they disagree on 19 of 50: the
free read started a written answer there (see the Wafer free-read table below), so its letters
were a poor proxy.

### Replies stuck before the value (full runs, 2026-09-27)

On some items Makora's greedy path never opens the value: after `{ "answer": ` (or after `{` and
two newlines) it repeats `\r\n` until `max_tokens`. That happened on about 3% of SST-5 items and
on 3 of 12,032 MMLU-Pro items in select and holdout; no other benchmark had it, and Luna never
did. The backend treats such a reply like an empty one: it is not cached and is sent again, up to
3 requests per call, and after a stuck reply the resends raise `max_tokens` from 16 to 64
(manifest flag `stuck_reply_max_tokens: 64`). The cap does not change the distribution read at
the value token. Most stuck items answer on a resend or a later pass. 6 of the 2,210 SST-5
select and holdout items (3 in each split; 6 of the 27,101 items the two splits hold in all)
never answered in about 20 attempts, 9 of them at `max_tokens` 64, every reply the same
whitespace loop; they are left unanswered. The billed cost of stuck attempts is recorded on the
item.

### Free first-token label mass per benchmark (Wafer, 2026-09-25)

Observed mass is the summed probability of the label tokens in the top 20.

| Benchmark | Labels | Observed mass mean | p05 | min | Share >= 0.99 | Labels missing per item | Greedy token is a label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpqa_diamond_idk | 5 | 0.710 | 0.185 | 0.150 | 0.22 | 0.06 | 0.72 |
| lexam_en_idk | 5 | 0.995 | 0.982 | 0.894 | 0.90 | 0.00 | 1.00 |
| mmlu_pro | 10 | 0.945 | 0.407 | 0.195 | 0.80 | 0.42 | 0.92 |
| arc_challenge | 4 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 | 1.00 |
| ag_news | 4 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 | 1.00 |
| boolq | 2 | 1.000 | 1.000 | 1.000 | 1.00 | 0.02 | 1.00 |
| sst5 | 5 | 1.000 | 0.998 | 0.996 | 1.00 | 0.00 | 1.00 |

On GPQA and MMLU-Pro, the non-label mass sat on words that open a written answer: `Let`,
`The`, `To`, `Alright`, `First`, `We`. Under the structured read the mask leaves only labels at
the value.

## Consequences for the runs

- Runs use the structured read: `ConstrainedStrategy` (`run_split.py --strategy constrained
  --provider makora`, run directory `deepseek-v4.1-flash@makora.constrained.state_first`,
  manifest flags `constraint: json_schema`, `providers: makora`). `Capabilities`:
  `structured_choice=True`, `mask_reflected_in_logprobs=True`; `top_logprobs=20`,
  `max_tokens=16`, `reasoning.enabled=false`. The prompt and its `template_id` are the free
  read's; the strategy, the constraint and the provider tell the runs apart.
- A response whose list at the value does not list the generated token at its own logprob (a
  stale list, as on Wafer) is rejected and not cached, so a structured run on such a provider
  stops with an error instead of reading the wrong position.
- MMLU-Pro (10 options) fits and is run, which Luna cannot do. banking77 and CLINC150 are
  skipped.
- Makora's input costs three times Wafer's, and a structured reply bills about 9 output tokens
  instead of 1: $0.03 to $0.10 per 1,000 items.
- No argmax flipped in 100 repeated structured calls on Makora; on Wafer's free read, 2.4% did.
