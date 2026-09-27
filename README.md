# jevemu

An open emulator of TypeSafe's Jev API (`POST /v1/systemone`), plus a paired benchmark of local
and hosted models against the real Jev.

Jev answers structured questions with a probability for every option. A question is a choice
between options, a yes/no question (Jev calls it a "noul"), or a score on a scale. `jevemu`
answers the same `SystemOneRequest` objects with an open model instead. It renders the question
as a prompt, reads the label probabilities from the model's next-token distribution, and can
debias and calibrate them. The model can run locally on vLLM, or be a hosted model that returns
logprobs, read through Structured Outputs (OpenAI GPT-6 Luna, DeepSeek V4.1 Flash via
OpenRouter).

This project is not affiliated with TypeSafe. Comparisons use the public Jev API (`jev-1.13.0`).

> All of the code, tests, documentation and reports in this
> repository were written by LLM coding agents working under human direction: a person chose
> the questions, the models and the budgets, reviewed the results and decided what to keep. Every
> number in the reports is recomputed from recorded runs by the scripts here, but read the code
> and the claims with that in mind.

## Results

We run 9 public benchmarks: multiple choice, intent classification, yes/no and a sentiment
scale. Every system answers the same frozen questions. Each benchmark was split once, before any
model ran:

- `screen` (2,499 questions): a small sample of `select`. All 22 local candidates answered it,
  to pick the finalists. Only the summary report's figures of all candidates use it.
- `select`: half of each benchmark. Every choice (finalists, prompt layout, quantized build,
  vLLM settings) was made on it.
- `holdout` (17,340 questions): the other half. No choice was made on it. Every number below
  is from `holdout`.

Calibration is fitted on `holdout` too, but out of fold: its questions are split into 5 folds,
and each fold is scored with temperatures fitted on the other 4, so no question is scored by a
fit that saw it. The same system scores differently on different splits (the finalists score 1
to 4 points lower on `screen` and `select`, partly because GPQA's `holdout` half happens to be
easier), so compare numbers only within one split.

On the 6 benchmarks every system can answer (7,534 holdout questions), macro accuracy (the mean
of the per-benchmark accuracies) was:

| System | Where it runs | Accuracy | Δ vs Jev (points) | ECE, raw → calibrated | Questions/s |
| --- | --- | --- | --- | --- | --- |
| Jev | TypeSafe API | 81.6% | — | 0.082 → 0.052 | 20.0 (our client's limit) |
| Qwen3.6-27B (AWQ INT4) | vLLM, one RTX 3090 | 76.8% | −4.8 | 0.072 → 0.045 | 10.0 |
| DeepSeek V4.1 Flash | OpenRouter | 75.7% | −5.9 | 0.105 → 0.050 | 7.4 (our client's limit) |
| GPT-6 Luna | OpenAI | 75.6% | −6.0 | 0.168 → 0.053 | 7.5 (our client's limit) |
| Qwen3.6-35B-A3B (AWQ INT4, tuned) | vLLM, one RTX 3090 | 73.4% | −8.1 | 0.078 → 0.044 | 28.1 |
| Gemma 4 26B-A4B (AWQ INT4, QAT) | vLLM, one RTX 3090 | 73.4% | −8.2 | 0.167 → 0.050 | 52.3 |
| Qwen3.6-35B-A3B (GPTQ INT4) | vLLM, one RTX 3090 | 73.0% | −8.6 | 0.091 → 0.050 | 53.2 |

ECE is the expected calibration error: how far stated probabilities are from observed accuracy
(lower is better); "calibrated" is the out-of-fold score described above. Every local model
answers each question in one model call. DeepSeek never answered 3 SST-5 questions (its
provider loops on whitespace); it is scored on the other 7,531.

Much of Jev's lead comes from GPQA-Diamond: its 99 questions count as much in the macro as the
3,800 of AG News, and give 67 to 82% of Jev's lead over Qwen3.6-27B, DeepSeek and Luna. Leaving
GPQA out, Jev leads them by 1.9, 2.2 and 1.3 points (Luna's paired CI reaches −0.1); weighing
every question the same, by 1.5, 1.2 and 1.9 points (every paired CI above 0). Local runs are
not exactly repeatable and the intervals cover only which questions were drawn, so systems
within about 1 point of each other cannot be ranked.

Over all 9 benchmarks (17,340 holdout questions; the hosted models cannot answer all of them),
Jev scores 82.5% at 20.0 questions/s. The local models score 76.6% (Qwen3.6-27B, 4.4/s), 73.5%
(Gemma 4 26B-A4B, 20.4/s), 73.0% (Qwen3.6-35B-A3B AWQ, 19.7/s) and 72.6% (Qwen3.6-35B-A3B GPTQ,
25.1/s). They are slower here because banking77 (77 options) and CLINC150 (150) have long
prompts.

In the questions/s column, the API rates come from our own request limiter and do not measure
what the services can handle. Local speed was measured on one RTX 3090 with vLLM and will differ
on other hardware.

Further reading:

- [Summary report](reports/summary/index.html): interactive HTML. Download it and open it in a
  browser; GitHub shows only the source. It covers:
  - accuracy and Brier score (the mean squared error of the probabilities) against speed, for
    22 local models and the API models;
  - the finalists on holdout;
  - reliability diagrams;
  - how calibration works;
  - where each speed limit comes from, and what the services' own limits would allow.
- [reports/README.md](reports/README.md): the other generated reports (Jev vs the Qwen
  emulators, and the speed/quality trade-off of every local candidate).
- [docs/README.md](docs/README.md): the design document and the experiment notes (probe reports
  for each serving stack, model selection, calibration).
- [docs/TODO.md](docs/TODO.md): open ideas, such as running the benchmark on macOS.

## Usage

The emulator takes the same `SystemOneRequest` objects as Jev and returns a
`SystemOneResponse` (its `model` is `jevemu/<served model>`). Start a vLLM server first (see
below).

The recommended configuration
([docs/research/calibration.md](docs/research/calibration.md#recommended-deployment)) is:

- the `qwen3.6-27b-int4-quanttrio` preset;
- the default `state_first` layout and `mode_distance` confidence;
- `SingleCallStrategy` scoring (`auto_single`: every question in one model call);
- no debiaser;
- the calibrator registry fitted on its `holdout` run.

```python
import asyncio
from pathlib import Path

from jevemu.backends.vllm_http import VLLMHTTPBackend
from jevemu.calibrate import CalibratorRegistry
from jevemu.emulator import Emulator
from jevemu.scoring import SingleCallStrategy
from jevemu.types import SystemOneRequest

REGISTRY = "calibration/qwen3.6-27b-int4-quanttrio/registry.json"


async def main() -> None:
    request = SystemOneRequest.model_validate_json(Path("request.json").read_text())
    async with VLLMHTTPBackend("http://localhost:8000", "QuantTrio/Qwen3.6-27B-AWQ") as backend:
        emulator = Emulator(
            backend,
            strategy=SingleCallStrategy(),
            debiaser=None,  # PriDe/batch did not pay off on holdout
            calibrators=CalibratorRegistry.load(REGISTRY),  # temperature per signature
            round_to=0.01,  # Jev's precision
        )  # probes capabilities on first use
        print((await emulator.system_one(request)).model_dump_json(indent=2))


asyncio.run(main())
```

`Emulator(backend)` alone uses the `auto` strategy (echo scoring where available, for small
models), no debiaser and no calibration.

The registry applies only to the model and template it was fitted on
(`QuantTrio/Qwen3.6-27B-AWQ`, `state_first-5d29289f8b87`). Other backends get the identity
calibrator, which changes nothing.

To use PriDe with fitted priors, pass `debiaser=PriDeDebiaser(priors=LabelPriors.load(path))`
(from `jevemu.debias`). PriDe estimates the model's preference for each answer label and divides
it out. Fit the registry on top of the same priors
(`scripts/calibrate.py registry RUN --priors path`).

Code written for the official `typesafe-sdk` runs on the emulator if you change its import to
`from jevemu.compat.typesafe import TypeSafeClient, Choice, Noul, Score`. The server address
comes from `JEVEMU_VLLM_URL` / `JEVEMU_VLLM_MODEL`. The module docstring lists the differences.

```bash
uv run jevemu ask request.json --diagnostics            # one request from the CLI
uv run python scripts/smoke.py                          # 20 LEXam-en items: Jev vs emulator
```

The real Jev client reads `TYPESAFE_API_KEY` from the environment or a gitignored `.env`. It
caches every response in `~/.cache/jevemu/jev_cache.sqlite` and takes a `max_usd` budget. To
re-record the live Jev fixtures, run `uv run pytest -m live_jev tests/live_jev -s`.

## Running the benchmark

Each built-in benchmark downloads one pinned Hugging Face file (commit and hash checked) and
turns it into Jev questions. Each spec records its license (`jevemu.bench.list_benchmarks()`).
"With IDK" means an added "I don't know" option.

| Benchmark | Source (split) | Type | Items | License |
| --- | --- | --- | --- | --- |
| `gpqa_diamond_idk` | Idavidrein/gpqa (diamond), with IDK | choice (4+1) | 198 | CC-BY-4.0, gated: never publish examples |
| `lexam_en_idk` | LEXam-Benchmark/LEXam (test, English), with IDK | choice (4+1) | 619 | CC-BY-4.0 |
| `mmlu_pro` | TIGER-Lab/MMLU-Pro (test) | choice (3-10) | 12,032 | MIT |
| `arc_challenge` | allenai/ai2_arc (ARC-Challenge test) | choice (3-5) | 1,172 | CC-BY-SA-4.0 |
| `ag_news` | fancyzhx/ag_news (test) | choice (4) | 7,600 | unknown; non-commercial source |
| `banking77` | legacy-datasets/banking77 (test) | choice (77) | 3,080 | CC-BY-4.0 |
| `clinc150` | clinc/clinc_oos (plus test, in-scope only) | choice (150) | 4,500 | CC-BY-3.0 |
| `boolq` | google/boolq (validation) | noul | 3,270 | CC-BY-SA-3.0 |
| `sst5` | SetFit/sst5 (test) | score (5) | 2,210 | unknown |
| `yelp_stars` | Yelp/yelp_review_full (test, 1,000 per star) | score (5) | 5,000 | Yelp terms: non-commercial; dropped |

Yelp is not part of the protocol. Its levels ("1 star" .. "5 stars") are shown to the models
as digits 0-4, and the models read them ambiguously. It is not in `DATASETS`
(`--benchmarks all`). Summaries, comparisons, statistics and calibration studies skip its records
unless it is named (`jevemu.eval.splits.DROPPED_DATASETS`). Results cover the other 9
benchmarks.

The repository does not store benchmark data; it is downloaded at run time. Tests keep 5 rows
per benchmark as fixtures; the GPQA and Yelp fixtures are synthetic. Run records stay in the
gitignored `runs/`, and the reports hold aggregates only. GPQA is gated: do not publish its
questions or per-item outputs.

`jevemu.eval.splits` cuts every benchmark in half, stratified by subject or gold class:

- `select`: for choosing the model and configuration;
- `holdout`: for calibration and the final numbers;
- `screen`: at most 300 `select` items.

Use `split_items("banking77", "select")` to get a split, and `freeze_split(...).to_jsonl(path)` to
write one for paired runs.

```bash
uv run python scripts/run_split.py run --system jev --benchmarks all --split select --out runs/select
uv run python scripts/run_split.py run --system emulator --system-id NAME --strategy auto_single  # vLLM from env
uv run python scripts/run_split.py run --system openai --strategy constrained \
  --prompt-cache-mode explicit --benchmarks all   # skips mmlu_pro, banking77, clinc150
uv run python scripts/run_split.py run --system openrouter --strategy constrained \
  --model deepseek/deepseek-v4.1-flash --provider makora --benchmarks all   # any logprob model
uv run python scripts/run_split.py summarize runs/select/jev          # per-benchmark + macro table
uv run python scripts/run_split.py compare runs/select/NAME runs/select/jev   # paired, a - b
uv run python scripts/run_split.py stats runs/select/jev runs/select/NAME --split select \
  [--gpu-usd-per-hour X]   # tokens, calls, latency, $ per 1k items / per 1k correct
```

`jevemu.bench.runner.run_split` writes one record per item to
`<out>/<system_id>/<benchmark>.<split>.jsonl`, plus a `manifest.json`. It freezes the split under
`<out>/splits/` so every system answers identical items. On resume it skips items already
answered. `jevemu.bench.compare.summarize`/`compare` score run directories. Results and the
protocol: [docs/research/selection.md](docs/research/selection.md).

`--system openai` runs the same `Emulator` over GPT-6 Luna
(`jevemu.backends.openai_chat.OpenAIChatBackend`; `OPENAI_API_KEY` from the environment or
`.env`). What differs from vLLM:

- Prompts end in the user turn, with no `Answer:` prefill (`template_id` `…-noprefill-…`).
- Reasoning is off. With `--strategy constrained` the answer is read through Structured
  Outputs: the reply must be `{"answer": "<label>"}` with the label from a JSON-schema `enum`,
  and the label distribution is read at the token where the value starts. The mask is
  reflected in the logprobs, so only labels compete there. `--strategy first_token` reads the
  first token of a free reply instead. Luna returns at most 5 alternatives and drops those
  below a threshold.
- A label missing from those gets an upper bound: under the structured read the leftover
  probability mass (it can only be the missing labels'); under the free read the smaller of
  that and the smallest returned logprob.
- Benchmarks with more than 5 options per question (MMLU-Pro, banking77, CLINC150) cannot run
  on OpenAI models, because OpenAI returns at most 5 top logprobs. They are skipped.
- Responses are cached in `~/.cache/jevemu/openai_cache.sqlite`, `--max-usd` caps spend, and
  items are priced by tokens (`price_id` in the manifest).

`scripts/probe_chat.py` measures what the API does (prefill, surfaces, Structured Outputs
masking, nondeterminism, prompt caching, tokenizer). Results:
[docs/research/openai_probe_report.md](docs/research/openai_probe_report.md).

`--system openrouter` does the same over any OpenRouter model whose providers return logprobs
(`jevemu.backends.openrouter_chat.OpenRouterChatBackend`, `OPENROUTER_API_KEY`; default
`deepseek/deepseek-v4.1-flash`):

- Requests turn reasoning off, ask for `--top-logprobs` (default 20) and `--max-tokens`
  (default 16), and go only to providers that support logprobs.
- `--provider SLUG[,SLUG]` pins providers with no fallbacks and adds `@SLUG` to the system id.
  This is recommended, because providers differ in quantization. The structured read needs a
  provider whose top logprobs belong to their own position: for DeepSeek V4.1 Flash, Makora
  (Wafer's are often an earlier position's, and such replies are rejected).
- Prices need no table entry. At start, the backend reads the model's public endpoints listing
  and saves the highest rate across the eligible providers and their time-of-day prices into the
  manifest (`token_price`). That rate prices response-cache hits and `stats`. A network call
  costs what OpenRouter billed (`usage.cost`).
- Responses are cached in `~/.cache/jevemu/openrouter_cache.sqlite`.

Measured behavior:
[docs/research/openrouter_probe_report.md](docs/research/openrouter_probe_report.md).

Calibration runs on `holdout`. `scripts/calibrate.py crossfit NAME=RUN_DIR ...` cross-fits
every registered calibrator (5 folds by question id) for Jev and the emulator alike. It reports
the {Jev, emulator} x {raw, calibrated} grid with paired bootstrap intervals.

Debiasing removes answer-position bias: `Emulator(debiaser=...)` (`jevemu.debias`:
`permutation`, `pride`, `contextual`, `batch`).

On `holdout` (9 benchmarks, Yelp dropped), one temperature per question signature brings the
selected emulator's ECE from 0.067 to 0.035 (Jev: 0.072 to 0.048). Results, the deployment
registry, Jev's confidence function and the commands:
[docs/research/calibration.md](docs/research/calibration.md).

## Development

```bash
uv sync                                  # core + dev tools (no CUDA needed)
uv run pre-commit install                # ruff/mypy on commit, scripts/check.sh on push
scripts/check.sh                         # lint, format, mypy, offline tests
```

There is no hosted CI. `scripts/check.sh` is the full check, and the pre-push hook runs it.
Live-server tests are opt-in (`uv run pytest -m gpu`, `-m cpu`, `-m live_jev`).

`third_party/evaluate-idk` is a git submodule, used only for reference and conformance tests.
It is not shipped in the wheel. Fetch it with `git submodule update --init`.

### vLLM server (Docker)

The vLLM server runs from the official Docker image, pinned by digest
(`docker/vllm/compose.yaml`; needs the NVIDIA container runtime). The HTTP backend does not need
the `vllm` pip extra. Each supported model is a preset file, `docker/vllm/presets/<name>.env`,
with the model, pinned revision, context length, GPU share, chat-template defaults and extra
`vllm serve` flags:

| Preset | Model | VRAM | Notes |
| --- | --- | --- | --- |
| `qwen3-0.6b` (default) | `Qwen/Qwen3-0.6B` bf16 | 1.5 GB of weights (reserves 0.85 of the GPU for KV cache) | Fast default for tests and smoke runs |
| `qwen3.8-27b-awq` | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (Qwen3.8-27B, 4-bit) | 24 GB GPU with nothing else on it: 18.4 GB weights, 0.95 of the GPU, ~13k KV tokens | Newest Qwen that fits an RTX 3090; ~2 min startup, ~230 ms per question |
| `qwen3.6-27b-int4-quanttrio` | `QuantTrio/Qwen3.6-27B-AWQ` (Qwen3.6-27B, AWQ 4-bit) | 24 GB GPU alone: 19.05 GiB weights, 0.97 of the GPU, ~11k KV tokens | Recommended emulator ([docs/research/selection.md](docs/research/selection.md)); ~2 min startup, ~4.4 items/s at 16 in flight (9 benchmarks) |

Other presets (`docker/vllm/presets/`) are the selection candidates.

```bash
eval "$(scripts/serve_vllm.sh --preset qwen3.6-27b-int4-quanttrio | grep '^export ')"  # up -d, wait, export
uv run python scripts/probe_vllm.py      # capability probe (cached in ~/.cache/jevemu/probes)
uv run pytest -m gpu tests/gpu          # live checks (skipped without JEVEMU_VLLM_URL)
docker compose -f docker/vllm/compose.yaml down
```

`scripts/serve_vllm.sh` (or `JEVEMU_VLLM_PRESET=<name>`) prints every variable clients read:
URL, image digest, model, resolved revision, `JEVEMU_VLLM_MAX_LOGPROBS`,
`JEVEMU_VLLM_LOGPROBS_MODE` and `JEVEMU_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS`. Preset values override
the environment. Without a preset, the compose file's `JEVEMU_VLLM_*` variables apply (default
`Qwen/Qwen3-0.6B`). Adding a model takes one small preset file.

Launch flags follow the design: `--max-logprobs 576 --logprobs-mode raw_logprobs
--enable-prefix-caching --max-model-len 8192 --seed 0`, plus `--enable-prompt-tokens-details`
to report `cached_tokens`. Every question is answered in one model call, and every answer label
is one token (`jevemu.scoring.single_call`). Questions with up to 32 options use letters.
Questions with more options (banking77 has 77, CLINC150 has 150) use two-capital codes (`AA`,
`AB`, ...), kept only if they are one token with and without a leading space (522 codes on
Qwen, 588 on Gemma). 576 top logprobs let one constrained call read every code of a
255-option question. Notes on the larger models:

- Qwen3.5/3.8 chat templates think unless told not to (Qwen3.8's also adds a reasoning-effort
  instruction to the system prompt), so their presets start the server with
  `--default-chat-template-kwargs '{"enable_thinking":false}'`. vLLM applies this to chat and
  `/tokenize` alike, so every client renders thinking-off prompts. When the exported variable is
  set, `BackendInfo.flags` reports it as `chat-template-kwargs`.
- The `Answer:` prefill goes after the generation prompt. `/tokenize` renders the messages with
  the model's own generation prompt, then `Answer:` is appended as token ids and sent to
  `/v1/completions`, so the model reads `Answer:` exactly where its reply would start. Rendering
  it as a continued assistant message dropped the empty thought channel that Gemma 4 12B and
  26B-A4B open their turn with; their runs were redone after the fix.
- The 27B preset uses a community W4A16 quant (4-bit weights, 16-bit activations), not an
  official checkpoint (official bf16 needs ~52 GB, FP8 ~28 GB). Its logprobs differ from the
  bf16 model's, so report it as its own model. The Qwen3.5-9B ladder (`qwen3.5-9b-*` presets)
  measures how much:
  [docs/research/quantization_report.md](docs/research/quantization_report.md).

Measured behavior of the pinned version, per model:
[docs/research/vllm_probe_report.md](docs/research/vllm_probe_report.md). To re-record the
golden HTTP fixtures, run `uv run python scripts/record_vllm_fixtures.py` (it uses the default
preset).
