# evaluate-idk code audit

This audit reads the pinned evaluate-idk code and its stored results to pin down how it builds
prompts, orders options, extracts answers and scores them. Every stored LEXam prompt and every
stored metric was reproduced exactly. The stored GPQA results came from an older prompt, so no
stored artifact uses the pinned GPQA prompt. Option order comes from one shared random stream
seeded with 42, not from per-question seeds, and neither dataset is pinned to a revision; §11
lists what jevemu must match to claim parity.

Audit date: 2026-09-24. Scope: every source file of `third_party/evaluate-idk` at the pinned SHA,
the parts of its locked dependencies that decide its behaviour, and the stored `results/`. The
audit closes the Partial, Inferred, Discrepancy and unverified rows of the "Findings:
evaluate-idk" table in `docs/implementation/design.md`, as they stood before the audit.
`design.md` has since been updated with these results; see the end of §11.

IDK stands for the "I don't know" option (E) that evaluate-idk appends to every question.

## Conventions

- Bare paths such as `custom_tasks.py:65-79` are relative to `third_party/evaluate-idk/` at the pinned SHA.
- `LE:` means the lighteval 0.11.0 wheel. Its sha256 `982a7549…6969` matches `uv.lock:1544`. Paths are relative to site-packages.
- Other dependency wheels, all at the versions pinned in `uv.lock`: `LL:` litellm 1.77.4, `DS:` datasets 4.1.1, `DE:` python-dotenv 1.1.1.
- **[RUN]** marks a claim checked by a throwaway script run against the stored `results/` or the public LEXam parquet. The scripts lived in `/tmp` and were not committed. Nothing under `third_party/` was modified, and `git status` inside the submodule stayed clean.
- **[INFERENCE]** marks a claim that follows from the code but that was not directly observed.
- evaluate-idk has no LICENSE. The only quotes below are short snippets needed as evidence.

## Resolution of the design-doc table

| design.md row (before the audit) | Old status | Resolution |
| --- | --- | --- |
| Framework | Implied | **lighteval 0.11.0** (`uv.lock:1510-1511`), with extras `litellm, extended_tasks, multilingual, math` (`pyproject.toml:11`, `uv.lock:1547-1568`). See §3. |
| Entry point | Discrepancy | `evaluate.sh` was deleted in commit `95cb26e` (2025-10-09). The real entry point is `python evaluate.py [--debug]`, which spawns `python run_eval.py endpoint litellm …` for each model. At the pin, every model entry is commented out (`evaluate.py:15-31`), so a bare run does nothing. See §2. |
| Prompt | Verified, but **wrong in one detail** | The current prompt asks for `Final Answer: ###C###`, not `Answer: <letter>` (`custom_tasks.py:54-55`). The README's "verbatim instructions" (`README.md:54`) are the old GPQA instruction, which is no longer in the code. See §5. |
| IDK option | Seed and RNG unverified | Python's stdlib global `random` (MT19937) is seeded once with `random.seed(42)` when the module is imported (`custom_tasks.py:33`). Each row consumes one `random.shuffle` of the 4 options, in dataset row order (`custom_tasks.py:77`). There is **no per-question seeding**. The gold index is remapped with `.index()` (`custom_tasks.py:78`). Reproduced 619/619 LEXam prompts [RUN]. See §6. |
| Metrics | Verified | Formulas are in §8. One correction: the "best outcome among several letters" rule never triggers, because extraction yields at most one letter (0 of 7,353 stored rows had more than one [RUN]). An extraction failure scores `idk_score = −1`, not 0. |
| Extraction | Partial | Exact regexes are in §7. The primary extractor is lighteval's `IndicesExtractionConfig("NativeLetters")` with 6 prioritized patterns. evaluate-idk's own 5-step fallback runs only when that finds nothing. |
| Datasets | Partial | GPQA: `Idavidrein/gpqa` / `gpqa_diamond` / `train`, n = 198. LEXam: `LEXam-Benchmark/LEXam` / `mcq_4_choices` / `test`, filtered by `x["language"] == "en"`, n = **619**. Neither pins a revision. lighteval's upstream GPQA shuffle is **not** inherited. See §4 and §6. |
| Standard errors | Inferred | lighteval `mean_stderr` = sample SD (standard deviation, ddof = 1, i.e. divided by n − 1) / √n (`LE:lighteval/metrics/utils/stderr.py:43-49,95-98`). For 0/1 metrics this equals √(p(1−p)/(n−1)), which is *not* the plain binomial SE. Ensemble rows use ddof = 0. All 18 stored results JSONs were recomputed exactly [RUN]. See §8. |

## 1. Pin

| Field | Value |
| --- | --- |
| Repository | https://github.com/JoelNiklaus/evaluate-idk |
| SHA | `e5cb812b364f806655bc9f0b6428645bcef2c915` |
| Commit | Merge commit by Joel Niklaus, dated 2025-12-22T09:44:45+01:00. Its parent `9d1fe21` is "evaluated gemini 3 flash". |
| History | 53 commits, not shallow |
| Files | `.gitignore`, `README.md`, `analyze_answers.py`, `analyze_questions.py`, `custom_tasks.py`, `evaluate.py`, `pyproject.toml`, `run_eval.py`, `summarize_results.py`, `tasks.txt`, `uv.lock`, plus `results/` with 38 files. There is no LICENSE. |

Not all stored results were produced by the pinned code:

- **GPQA results** (all dated 2025-09-26) came from the deleted `gpqa_diamond_idk.py` at commit `2c2dfa0`. That version had a different prompt, no `random.seed(42)`, and `stop_sequence=["\n"]` (`git show 2c2dfa0:gpqa_diamond_idk.py`, lines 58-83 and 266-279). **No stored artifact uses the pinned GPQA prompt.**
- **LEXam results** (2025-10-09 → 2025-12-18) were produced with the pinned `PROMPT_TEMPLATE`, the pinned LEXam instruction and the pinned shuffle. All 619 prompts of every stored LEXam run were reproduced byte for byte [RUN].
- The fallback extractor was widened from A–E to A–K after most runs, in commit `a2d2d5a` (2025-12-17). Re-scoring every stored row with the *pinned* fallback still matches the stored metrics on all 7,353 rows [RUN].

## 2. Entry points

`README.md:80-83` says `bash evaluate.sh`. That file existed from `b6919ec` (2025-09-26) until `95cb26e` (2025-10-09, "switched to python script"), which replaced it with `run_evaluations.py`. Commit `8c882d6` then renamed that to `evaluate.py`.

What actually runs, from the repo root:

```
python evaluate.py [--debug]
```

This reads `tasks.txt` at import time through a relative path (`evaluate.py:52`), dropping blank lines and `#` lines (`evaluate.py:44-49`). The result is `community|lexam-en-idk|0,community|gpqa-diamond-idk|0` (`tasks.txt:1-2`). For each entry in `OPENROUTER_MODELS` (`evaluate.py:15-23`, all commented out at the pin), it runs this command (`evaluate.py:63-84` and `evaluate.py:138-145`):

```
<sys.executable> run_eval.py endpoint litellm \
  "provider=openrouter,model_name=openrouter/<org>/<model>[-<effort>],concurrent_requests=50" \
  "community|lexam-en-idk|0,community|gpqa-diamond-idk|0" \
  --custom-tasks custom_tasks.py --dataset-loading-processes 1 --save-details \
  [--max-samples 2 --output-dir results-debug]          # only with --debug
```

- `--debug` lowers `concurrent_requests` to 1, sets `max_samples=2`, writes to `results-debug`, and sets the DEBUG log level (`evaluate.py:35-38,119-121,143-144`). `CONCURRENT_REQUESTS = 50` (`evaluate.py:33`). The Sept 2025 runs used 10, from the old `evaluate.sh`, and their JSONs record 10.
- A reasoning effort is encoded as a model-name suffix `-{high,medium,low,minimal,none}` (`evaluate.py:131-136`). `run_eval.py:6-30` monkeypatches `litellm.completion` to strip the suffix and inject `reasoning_effort`. Apart from that, `run_eval.py:2,33` is just lighteval's Typer app: `endpoint` is a sub-app (`LE:lighteval/__main__.py:72-77`) and `litellm` is its command (`LE:lighteval/main_endpoint.py:224-296`). None of the retained runs has a suffix, so they used the provider's default effort.
- A second path, `endpoint inference-providers "model_name=…,provider=novita,parallel_calls_count=50,org_to_bill=huggingface"` (`evaluate.py:87-105,147-151`), is also fully commented out (`evaluate.py:26-31`).
- The output root defaults to `results` (`LE:lighteval/cli_args.py:118-127`).

Environment:

| Variable | Use | Reference |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Read by litellm for the `openrouter/` provider | `LL:litellm/main.py:2688`; `README.md:72` |
| `HF_TOKEN` | Dataset download. GPQA is gated: unauthenticated API calls returned HTTP 401 during this audit. | `README.md:72` |
| `.env` | Loaded by `load_dotenv()` when `custom_tasks.py` is imported (see §11) | `custom_tasks.py:29` |
| `EVALUATE_IDK_LOG_LEVEL` | Root logging level | `custom_tasks.py:27`; `evaluate.py:37` |
| `LITELLM_LOG` | litellm logging | `evaluate.py:38` |
| `LITELLM_LOCAL_MODEL_COST_MAP` | Optional. Skips litellm's network fetch at import. | `LL:litellm/litellm_core_utils/get_model_cost_map.py:16-45` |
| `PYTHONHASHSEED` | Implicit. Decides task order when both tasks run in one process, which changes option orders (§6). | `LE:lighteval/tasks/registry.py:202` |

## 3. Framework and dependencies

- **lighteval 0.11.0** (`uv.lock:1510-1511`), sdist uploaded 2025-09-22 (`uv.lock:1542`). The lock stayed on 0.11.0 in every revision of history: `2c2dfa0`, `4b6303c`, `bac3e0e`, `7e7ecef` and `e5cb812`.
- Python: `requires-python = ">=3.10"` (`pyproject.toml:9`, `uv.lock:3`). The lock resolves separately for <3.11, 3.11.* and ≥3.12 (`uv.lock:4-8`). The README creates a 3.10 venv (`README.md:75`), and there is no `.python-version`. Stored JSONs record no Python or library versions (`"lighteval_sha": "?"`, `"versions": {}`; see `results/results/openrouter/openai/gpt-5.2/results_2025-12-12T15-11-59.471879.json:3,63`).
- Key locked versions (the `uv.lock` line is the `version =` line):

| Package | Version | uv.lock |
| --- | --- | --- |
| litellm | 1.77.4 | 1572 |
| datasets | 4.1.1 | 751 |
| huggingface-hub | 0.35.1 | 1176 |
| numpy | 2.2.6 (py<3.11) / 2.3.3 (py≥3.11) | 2193 / 2258 |
| pandas / pyarrow | 2.3.2 / 21.0.0 | 2497 / 2867 |
| torch | 2.8.0, with CUDA 12 wheels on linux x86_64 (`uv.lock:4396`) and triton 3.4.0 | 4388 |
| transformers / tokenizers | 4.56.2 / 0.22.1 | 4452 / 4324 |
| python-dotenv | 1.1.1 | 3206 |
| openai | 1.109.1 | 2469 |
| pydantic / typer | 2.11.9 / 0.19.2 | 2972 / 4508 |
| sympy / latex2sympy2-extended | 1.14.0 / 1.0.6 | 4176 / 1498 |
| nltk / scikit-learn / xxhash | 3.9.1 / 1.7.2 / 3.5.0 | 2178 / 3669 / 4663 |
| tiktoken | 0.11.0 | 4288 |

The lock holds 187 packages in total. The direct dependencies are `pyproject.toml:10-18`.

## 4. Datasets

| Task (`tasks.txt`) | HF repo | Config | Split | Filter | Revision | n |
| --- | --- | --- | --- | --- | --- | --- |
| `community\|gpqa-diamond-idk\|0` | `Idavidrein/gpqa` | `gpqa_diamond` | `train` | none | unpinned | 198 |
| `community\|lexam-en-idk\|0` | `LEXam-Benchmark/LEXam` | `mcq_4_choices` | `test` | `lambda x: x["language"] == "en"` | unpinned | 619 |
| `idk-eval` (defined, not in `tasks.txt`) | `CatLaugh/idk_eval` | none | `test` | none | unpinned | not audited |

Sources: `custom_tasks.py:412-426` (LEXam), `custom_tasks.py:428-442` (GPQA) and `custom_tasks.py:444-457` (idk-eval, added by `995f35e`, which uses columns `options_4` and `answer_index_4` per `custom_tasks.py:191-192`). `hf_revision` is never set, and its default is `None` (`LE:lighteval/tasks/lighteval_task.py:118`). Stored configs show `"hf_revision": null` (`…gpt-5.2/results_2025-12-12T15-11-59.471879.json:95`).

Loading and filtering work as follows:

- `load_dataset(path, name, revision)`, then `dataset.filter(hf_filter)` over the whole DatasetDict (`LE:lighteval/tasks/lighteval_task.py:441-448`).
- Loading is sequential with `--dataset-loading-processes 1` (`LE:lighteval/tasks/lighteval_task.py:411-414`).
- Docs are built by iterating the filtered split in row order. **`doc.id = str(row_index)`**, a position within the filtered split, not the dataset's own `id` column (`LE:lighteval/tasks/lighteval_task.py:289-305`).

LEXam revisions [RUN]:

- The Oct 9 and Nov 18 runs used the `mcq_4_choices/test` parquet with LFS sha256 `ee385b81…8743`, as of dataset commit `a49cd312f015` (2025-05-28). It has 1,660 rows: 1,041 `de` and 619 `en`.
- The Dec 12–18 runs used LFS sha256 `f4c10c4271ca00fc74ff28c88fc6939479635d9ebe7b67ff8d8a28ca46d957e9`. It was introduced in `6dafb3a9799c` (2025-12-06) and is unchanged at `85a95cf468b0` and at current `main` (`31e57ae395f9`). It has 1,655 rows: 1,036 `de` and 619 `en`.
- The file changed 12 times between 2025-12-02 and 2025-12-06. The two revisions share 551 of 619 English stems, and row order differs, so positional `doc.id`s point to different questions. evaluate-idk commit `edc7df0`, "updated results for lexam based on new dataset", reflects this.
- `choices` is a stringified Python list, which is parsed with `ast.literal_eval` (`custom_tasks.py:110-113`). All 619 `en` rows have exactly 4 distinct options, with no surrounding whitespace and no newlines.

GPQA:

- The 198 count comes from all 9 stored GPQA details files.
- The dataset is gated. The file listing is public (current `gpqa_diamond.csv` blob `7589e3e467d6`, repo lastModified 2026-09-21), but the commit history needs authentication, so the revision used in Sept 2025 is unknown.
- In the stored 2025-09 prompts, 6/198 questions have at least one option that spans several lines, and 2/198 have two textually identical distractors [RUN].

LEXam-en has **619** questions. Before this audit, `docs/implementation/design.md` planned its sample-size table for "~1,650" at "LEXam mcq_4 scale", but that figure is the all-language count.

## 5. Prompt

The template lives in `custom_tasks.py:45-58`. It is 354 bytes, sha256 `7ea04085…af50e`, with an ASCII apostrophe:

```
Question:
{question_text}

Choices:
{choices_str}

Before answering, think about the question step by step.
Answer only if you are confident, since mistakes are penalized with -1 points, while correct answers receive 1 point, and the answer E) "I don't know" always receives 0 points.

Format your final answer as follows:
Final Answer: ###C###

Answer:
```

The string ends with a `\n` after `Answer:`.

- `question_text` is the dataset question after `.strip()` (`custom_tasks.py:108,159`).
- `choices_str` comes from `build_choices_string` (`custom_tasks.py:82-95`). It emits `"{L}) {option}\n"` for L in A–D, taken from lighteval's `LETTER_INDICES` (`LE:lighteval/tasks/default_prompts.py:42`). It then appends the literal `E) I don't know`, **with no trailing newline**; the template supplies the following `\n\n`.
- E is therefore always the 5th and last line and is never gold. `Doc.choices = ["A","B","C","D","E"]`, the letters and not the option texts (`custom_tasks.py:137,178`). `gold_index` is 0–3 (`custom_tasks.py:138,179`).
- GPQA options are `.strip()`ed (`custom_tasks.py:160-165`). LEXam options are used as stored.
- Per-task `Doc.instruction`:
  - LEXam uses a course-specific chain-of-thought instruction (`custom_tasks.py:121-132`). It begins "You are an expert in {course_name} and address legal issues in a structured, exam-style manner." and ends with `\n`.
  - GPQA uses `"Answer the following multiple choice question."` with no trailing newline (`custom_tasks.py:173`).
  - idk-eval has no instruction (`custom_tasks.py:199-207`).
- Message assembly for API models puts **a single user message** containing `instruction + query`, concatenated with no separator, and no system message (`LE:lighteval/tasks/prompt_manager.py:103-127`). The pinned GPQA prompt therefore starts `"Answer the following multiple choice question.Question:\n…"` [INFERENCE; no stored run uses it]. For LEXam, every stored row has one `user` message equal to `instruction + query` (619/619 in each of 9 runs [RUN]). No system prompt was used: the default is `None`, and every stored JSON records `"system_prompt": null` (`…gpt-5.2/…json:32`).
- The old GPQA prompt, used for every stored GPQA result and quoted in the README, was: instruction "Before answering, think about the question step by step. … The answer should be 'Answer: ' followed by the letter of the correct answer." and query `"Answer the following multiple choice question.\n\n{Q}\n\nA) …\nE) I don't know\n\n"` (`git show 2c2dfa0:gpqa_diamond_idk.py`, lines 63-83).

## 6. Shuffle

- `shuffle_choices` copies the list, calls **`random.shuffle` on the process-global RNG**, and sets `new_gold = shuffled.index(correct_text)` (`custom_tasks.py:65-79`).
- The base order before shuffling is:
  - LEXam: the dataset `choices` list, with gold = `sample["gold"]` (`custom_tasks.py:110-116`).
  - GPQA: `[Incorrect 1, Incorrect 2, Incorrect 3, Correct]`, with gold = 3 (`custom_tasks.py:161-169`).
- Seeding and ordering within one run:
  1. `Pipeline.__init__` calls `random.seed(1234)` and then loads the tasks (`LE:lighteval/pipeline.py:140-141,248-251`).
  2. `Registry` executes `custom_tasks.py` twice (`LE:lighteval/tasks/registry.py:183,238,390-399`). Each execution runs `random.seed(42)` (`custom_tasks.py:33`), so **42 wins**.
  3. All datasets are loaded (`LE:lighteval/pipeline.py:217`). `datasets` fingerprinting uses its own `random.Random()` (`DS:datasets/fingerprint.py:204,224`) and does not touch the global RNG.
  4. Prompt functions then run per task, per row, in order (`LE:lighteval/pipeline.py:218-220`; `LE:lighteval/tasks/lighteval_task.py:289-297`).
- **Result:** the options are seeded, but the whole run shares one stream. There is no per-question RNG. A question's permutation depends on its row position, on the dataset revision, and on which task was formatted first.
- **Task order is hash-randomized.** The task list is deduplicated with `list(set(...))` (`LE:lighteval/tasks/registry.py:202`), and task and document order follow that list (`LE:lighteval/tasks/registry.py:332-341`). With both tasks in `tasks.txt`, the order flips with `PYTHONHASHSEED`: seeds 0, 1, 3, 4 and 5 put GPQA first, while 2, 6 and 7 put LEXam first [RUN]. With both tasks in one process, the second task's permutations therefore vary between processes. Every stored run contains a single task, so the stored runs are unaffected.
- Evidence [RUN]:
  - Running `random.seed(42)` and then one `random.shuffle` per row over the `en`-filtered `test` split reproduces `doc.query` and `gold_index` for 619/619 rows of every stored LEXam run, on both dataset revisions. A private `random.Random(42)` instance gives the same 619/619. Seed 1234 as a control matches only 29/619. Tested on CPython 3.11.7; that other 3.10–3.13 versions give identical `shuffle` output is [INFERENCE].
  - Across runs, `hash_examples` is xxh64 over the sorted xxh64 hashes of each `doc.query` (`LE:lighteval/logging/info_loggers.py:272,281-283`). It is identical within each batch: `068a43fe24ae173d` for all 9 GPQA runs, `9419dfa387a0fead` for the Oct/Nov LEXam runs, and `0e3d3070789c3703` for the 7 Dec LEXam runs.
- **Upstream lighteval's GPQA shuffle is not inherited.** evaluate-idk defines its own prompt function (`custom_tasks.py:143-181`). The upstream `gpqa` prompts use `random.randint(0, 3)` on the global RNG (`LE:lighteval/tasks/default_prompts.py:880-882,899-902`) and are never called here. The old `gpqa_diamond_idk.py` had no seed of its own, so its stored GPQA permutations came from lighteval's `random.seed(1234)` [INFERENCE; GPQA is gated, so this was not reproduced].
- `--max-samples` does not change option orders. It only subsamples docs after formatting, using a separate `Random(42)` (`LE:lighteval/tasks/lighteval_task.py:371-378`).
- A duplicate option text would make `.index()` choose the first equal option. No LEXam-en row has duplicates. The 2 GPQA duplicates are distractors, not gold [RUN].

## 7. Answer extraction

`ExtractiveLetterIdkGrouped.compute` (`custom_tasks.py:324-347`) processes each response text, which is `model_response.final_text`: the text after `<think>…</think>` removal, on by default (`LE:lighteval/pipeline.py:346-355`, `LE:lighteval/cli_args.py:94-113`, `LE:lighteval/models/model_output.py:141-145`).

**Stage 1, lighteval primary** (`custom_tasks.py:307-308,333-339`): `IndicesExtractionConfig(prefix_for_extraction="NativeLetters")` with `try_extract_without_anchor=False`, English, and `len(doc.choices)=5`, which gives letters A–E (`LE:lighteval/metrics/utils/extractive_match_utils.py:82-93,284-344`; English literals `LE:lighteval/tasks/templates/utils/translation_literals.py:53-63,385-407`; `get_prefix` `LE:lighteval/tasks/templates/utils/formulation.py:73-79`). The compiled patterns, generated from that source, are:

| Priority | Pattern |
| --- | --- |
| 0 | `(?i:final answer is)\:?\s*(?P<indices>…)\.?\s?I hope` |
| 50 | `(?i:final answer.{0,100}?)\s+is\:?\s*(?P<indices>…)` |
| 100 | `(?i:answer)\s?[:\:].{0,50}?(?:^\|\ )(?:\*\*)?(?P<indices>…)(?:\*\*)?(?:[\.\.]\|[,\,]\|\s?[:\:]\|\ \|$)` |
| 150 | `(?i:answer).{0,50}?(?:^\|\ )(?:\*\*)?(?P<indices>…)(?:\*\*)?(?:[\.\.]\|[,\,]\|\s?[:\:]\|\ \|$)` |
| 200 | `^(?:\*\*)?(?P<indices>…)(?:\*\*)?(?:[\.\.]\|[,\,]\|\s?[:\:]\|\ \|$)` |
| 210 | `\n(?:\*\*)?(?P<indices>…)(?:\*\*)?(?:[\.\.]\|[,\,]\|\s?[:\:]\|\ \|$)` |

Here `(?P<indices>…)` = `(?P<indices>(?:A|\(A\))|(?:B|\(B\))|(?:C|\(C\))|(?:D|\(D\))|(?:E|\(E\)))`. The letters are case-sensitive; only the word "answer" is case-insensitive. No DOTALL is set, so `.` does not cross newlines and `$` matches only at end-of-string.

Selection rules (`LE:lighteval/metrics/utils/extractive_match_utils.py:532-601`):

- Priority groups are tried in ascending order, and the first group with any match wins.
- Within that group, `finditer` matches are sorted with the rightmost end first (ties go to the leftmost start), and the first one is taken. Parentheses are stripped (`LE:lighteval/metrics/utils/extractive_match_utils.py:500-506`).
- `fallback_mode="first_match"` then appends the same string again, which gives `[X, X]`. `custom_tasks.py:344-346` deduplicates it to `[X]`.

**Stage 2, evaluate-idk fallback** (`extract_letter_fallback`, `custom_tasks.py:214-285`) runs only if stage 1 returned nothing (`custom_tasks.py:340-343`). It checks the following in order, with `LETTERS = "A-K"`, and upper-cases the result:

1. `###([A-K])###`, case-insensitive, taking the **last** occurrence (`custom_tasks.py:233-235`).
2. Boxed forms, trying the patterns in order and taking the first match anywhere in the text, case-insensitive (`custom_tasks.py:238-253`): `\$?\\boxed\s*\{\s*([A-K])\s*\}\$?`, the same wrapped in `\text{}`, the same wrapped in `\mathrm{}` or `\mathbf{}`, `\bboxed\s*\(\s*([A-K])\s*\)` and `\bboxed\s*\{\s*([A-K])\s*\}`.
3. `\banswer\s*[:\-]?\s*([A-K])\b` over the whole text, taking the leftmost match, case-insensitive (`custom_tasks.py:256-262`).
4. `\bfinal\s+answer\s*[:\-]?\s*([A-K])\b` over the last 200 characters (`custom_tasks.py:265-274`). **This step can never fire:** any text it matches is already matched by step 3.
5. `\b(?:option|choice)\s*([A-K])\b` over the last 200 characters (`custom_tasks.py:277-283`).

**Multiple letters.** Each stage returns exactly one letter, so a response yields zero or one letter. The "max over letters" rule in `custom_tasks.py:374-380` is inert for single-sample runs; 0 of 7,353 stored rows had more than one letter [RUN]. When a response mentions several letters, the winner is decided by the priority and position rules above, not by the best score.

Practical consequences [RUN]:

- Because stage 1 has priority, a chain-of-thought containing "The answer is B, …" beats a later `###D###`.
- `Answer: F` extracts F, which is scored as wrong, not as a failure.
- `answer: e.g. …` extracts E, which counts as abstention.
- A plain `I don't know` extracts nothing, which is a failure.
- `Answer: X` is taken by the priority-100 pattern.

In stored LEXam runs the fallback supplied 67–97% of the extractions per run. Across all 9 LEXam runs, 4,885 of the 4,888 fallback extractions came from the `###X###` step [RUN]. For each run's counts see the reproduction notes in §12.

## 8. Metrics

Implementation: `ExtractiveLetterIdkGrouped` (`custom_tasks.py:288-396`), registered as `idk_grouped_metrics`, a `SampleLevelMetricGrouping` with `corpus_level_fn = np.mean` for all four metrics (`custom_tasks.py:399-405`). lighteval calls `compute(doc=…, model_response=…)` once per doc (`LE:lighteval/metrics/__init__.py:52-58`; `LE:lighteval/metrics/utils/metric_utils.py:45-59`).

Per document *i*, let `c_i = "ABCDE"[gold_index_i]` (always in A–D), and let `ℓ_i` be the letter extracted from the single response under §7, or ∅ if there is none:

```
ℓ_i = ∅                      → trad=0, idk=−1, freq=0, fail=1     (custom_tasks.py:375-376)
ℓ_i = c_i                    → trad=1, idk=+1, freq=0, fail=0
ℓ_i = "E"                    → trad=0, idk= 0, freq=1, fail=0     (_score_letter, :315-322)
ℓ_i ∈ {A..D}\{c_i} or F..K   → trad=0, idk=−1, freq=0, fail=0     (F..K ∉ doc.choices → −1)
```

For each metric *m*, the corpus value and its stderr are:

```
M      = (1/n) · Σ_i m_i                                      (np.mean)
SE(M)  = sqrt( Σ_i (m_i − M)² / (n − 1) ) / sqrt(n)            (LE:lighteval/metrics/utils/stderr.py:43-49)
```

For 0/1 metrics this is `SE = sqrt(p(1−p)/(n−1))`. `get_stderr_function` selects `mean_stderr` because `np.mean.__name__` contains "mean", so no bootstrap is involved (`LE:lighteval/metrics/utils/stderr.py:95-98`; `LE:lighteval/logging/info_loggers.py:346-372`). The "all" entry averages every key, stderr included, across tasks (`LE:lighteval/logging/info_loggers.py:374-402`).

Two identities hold per document and therefore on the corpus:

```
trad + freq + wrong + fail = 1
idk_score = 2·trad_score + idk_freq − 1
```

The second holds because failures count as −1. For example, gpt-5.2 LEXam: 2·0.88853 + 0.00485 − 1 = 0.78191.

The general form, for `num_samples = k > 1`, which is never used because temperature is 0 and n = 1: `trad`, `idk` and `freq` take the maximum over the k per-sample values (`aggregation_function=max`), and `fail` is 1 only if all k samples failed (`custom_tasks.py:382-396`). `compute` also writes `doc.specific["extracted_predictions"/"extracted_golds"]` (`custom_tasks.py:367-370`). The gold extraction is stored but not used for scoring; `correct_letter` comes from `doc.choices[doc.gold_index]` (`custom_tasks.py:372`).

**Verification [RUN].**

- A re-implementation of the rules above (lighteval's regexes and `extract_target_from_pred` taken from the 0.11.0 source, plus the pinned `extract_letter_fallback`) reproduces the stored per-row `metric` dicts *and* `extracted_predictions` for all 7,353 rows in 18 details files, with 0 mismatches.
- Recomputing `M` and `SE` from those rows matches all 18 results JSONs with an absolute error of 0.

The functions and signatures a conformance test can import are:

| Symbol | Signature / behaviour | Reference |
| --- | --- | --- |
| `custom_tasks.ExtractiveLetterIdkGrouped` | `(language=Language.ENGLISH, aggregation_function=max, fallback_mode="first_match", extraction_mode="any_match", precision=6, timeout_seconds=5)` | `custom_tasks.py:297-313` |
| `….compute` | `(doc: Doc, model_response, **kwargs) -> dict[str, float]` with keys `trad_score, idk_score, idk_freq, extract_fail`. It mutates `doc.specific`. | `custom_tasks.py:324-396` |
| `custom_tasks.extract_letter_fallback` | `(pred: str) -> str \| None` | `custom_tasks.py:214` |
| `custom_tasks.idk_grouped_metrics` | `SampleLevelMetricGrouping`. `.compute_sample(model_response=…, doc=…)` is the pipeline path. | `custom_tasks.py:399-405` |
| `custom_tasks.shuffle_choices`, `build_choices_string`, `lexam_idk_prompt`, `gpqa_diamond_idk_prompt` | Prompt and shuffle, all on the global RNG | `custom_tasks.py:65-181` |
| `custom_tasks.PROMPT_TEMPLATE`, `NUM_CHOICES=5`, `GENERATION_SIZE=32768`, `STOP_SEQUENCES` | Constants | `custom_tasks.py:39-58` |
| `lighteval.tasks.requests.Doc` | `(query, choices, gold_index, instruction=None, …)`, a slots dataclass | `LE:lighteval/tasks/requests.py:189-223` |
| `lighteval.models.model_output.ModelResponse` | `ModelResponse(text=[...])`. `final_text` returns `text_post_processed` if set, otherwise `text`. | `LE:lighteval/models/model_output.py:122-145` |
| `lighteval.metrics.utils.stderr.mean_stderr` | `(arr) -> float` | `LE:lighteval/metrics/utils/stderr.py:48-49` |

## 9. Results format

Everything is written by lighteval's `EvaluationTracker` (`--save-details`):

- `results/results/<model_name>/results_<date_id>.json` (`LE:lighteval/logging/evaluation_tracker.py:307-312`).
- `results/details/<model_name>/<date_id>/details_<suite|task|fewshot>_<date_id>.parquet` (`LE:lighteval/logging/evaluation_tracker.py:314-329,351-358`).
- `date_id` is the local `datetime.now().isoformat()` with `:` replaced by `-` (`LE:lighteval/logging/evaluation_tracker.py:247`).
- `model_name` is the full LiteLLM model name, for example `openrouter/openai/gpt-5.2`, which produces the directory layout `openrouter/<org>/<model>/`.

The files contain:

- **Results JSON** keys: `config_general` (model config, generation parameters, `system_prompt`), `results` (per task `{metric, metric_stderr}` and `all`), `versions`, `config_tasks`, `summary_tasks` (`hash_examples`, …) and `summary_general`. `hash_full_prompts` is the same (`ef46db3751d8e999`) in every file: it hashes a field that is never populated (`LE:lighteval/logging/info_loggers.py:268-286`).
- **Details parquet**: one row per doc, with 3 struct columns:
  - `doc`: `query`, `instruction`, `choices`, `gold_index`, `id`, `specific.extracted_predictions`, …
  - `metric`: the 4 floats.
  - `model_response`: `input` (messages), `text`, `text_post_processed`, `reasonings`, ….
  - Rows are in lighteval's `Random(42)`-shuffled doc order, not dataset order (`LE:lighteval/tasks/lighteval_task.py:371-374`).
- The pin stores 18 runs: 9 GPQA (2025-09-26) and 9 LEXam (2 on the old dataset, 7 on the new one), plus 2 PNG charts.

**`summarize_results.py`** (a module-level script) does the following:

- It globs `results/results/**/results_*.json` (`summarize_results.py:10,240`).
- The display name is the last `/` segment of `config_general.model_config.model_name` (`summarize_results.py:258-265`).
- It reads `results["community|gpqa-diamond-idk|0"]` and `results["community|lexam-en-idk|0"]` (`summarize_results.py:270-301`).
- For each model it keeps the **lexicographically last path**, which is the latest timestamp (`summarize_results.py:341`), sorts by `trad_score` descending, and prints `value×100 ± se×100` to 2 decimal places (`summarize_results.py:318-329,357`).
- **Ensembles** are defined in `ENSEMBLES` (`summarize_results.py:15-23`) and computed from the details parquet:
  - Each row maps to an outcome: 0 if `extract_fail > 0`, 1 if `idk_score == 1`, −1 if `idk_score == −1`, and 0 otherwise. Failures therefore count as **IDK (0)** here, whereas the per-model metric scores them −1 (`summarize_results.py:104-139`).
  - Models are inner-joined on `doc.id`, which is positional (`summarize_results.py:120,191-194`).
  - A strict-majority vote decides each question, with ties going to 0 (`summarize_results.py:141-163`).
  - `trad = #1/n`, `idk = (#1 − #−1)/n`, `freq = #0/n`, `fail = 0`.
  - The SEs use **ddof = 0**: `sqrt(p(1−p)/n)` and `sqrt(Σ(o−idk)²/n / n)` (`summarize_results.py:208-225`).
- It writes charts to `results/figures/` only when matplotlib is importable (`summarize_results.py:431-434,589-594,596-618`).

Running the pinned script read-only [RUN] (without matplotlib, so nothing was written) reproduced the README GPQA table exactly. The LEXam table differs from `README.md:36-44`:

- ensemble-top3 is 90.95 ± 1.15 / 82.23 ± 2.28 / 0.32 ± 0.23, against the README's 88.69 / 78.84 / 1.45.
- A gemini-3-flash-preview row (83.36 / 68.01) appears.

So the README is stale relative to `9d1fe21`.

`analyze_answers.py` and `analyze_questions.py` are read-only diagnostics over the details parquet:

- `analyze_answers.py <model> [gpqa|lexam]` samples responses by outcome and reports o200k token lengths (`analyze_answers.py:40-65,115-240`).
- `analyze_questions.py [--benchmark] [--question-id] [--export-all-wrong FILE] [--export-agreement-failures]` groups questions by the positional `doc.id` across *all* runs (`analyze_questions.py:38-78,81-129,435-466`). It therefore conflates different LEXam questions across the two dataset revisions. It writes CSVs to the current directory (`analyze_questions.py:370-377,425-426`).

## 10. Generation settings

For every doc, `litellm.completion` receives (`LE:lighteval/models/endpoints/litellm_model.py:174-197`; `LE:lighteval/models/model_input.py:42-44,113-122`):

```
model="openrouter/<org>/<model>", messages=[{"role": "user", "content": instruction + query}],
n=1, temperature=0, max_completion_tokens=32768, caching=True,
logprobs=None, base_url=None, api_key=None     (+ reasoning_effort only via the run_eval.py suffix)
```

- **Temperature 0**: the `GenerationParameters` default, recorded as `"temperature": 0` in every stored JSON (`…gpt-5.2/…json:24`). No `top_p` or `seed` is sent. `litellm.drop_params = True` silently drops any parameter a provider does not support (`LE:lighteval/models/endpoints/litellm_model.py:135`).
- **Max tokens 32768**: from `GENERATION_SIZE` via `doc.generation_size` (`custom_tasks.py:42`; `LE:lighteval/models/endpoints/litellm_model.py:152-160,194-195,286`).
- **Stop sequences are never sent.** `STOP_SEQUENCES = ["\n", "\n\n"]` (`custom_tasks.py:43`) is passed through `_prepare_stop_sequence` (`LE:lighteval/models/endpoints/litellm_model.py:167`) but never added to `kwargs`. Stored evidence agrees: 6,570 of 7,353 responses contain `\n` [RUN].
- **System prompt:** none (§5).
- **Provider:** OpenRouter through LiteLLM. Logprobs are only requested for provider `openai` (`LE:lighteval/models/endpoints/litellm_model.py:170-177`).
- **Retries and failures:** there are 5 retries with exponential backoff, and one uncached retry if the content is `None`. After the last retry an empty response is returned, which becomes text `""` (`LE:lighteval/models/endpoints/litellm_model.py:199-221,306`). An empty response scores as `extract_fail = 1`, `idk_score = −1`. The stored runs contain 108 empty texts.
- **Caching:** litellm uses a disk cache (`LE:lighteval/models/endpoints/litellm_model.py:49`; `.litellm_cache/` is ignored at `.gitignore:209`), and lighteval uses a sample cache under `~/.cache/huggingface/lighteval`. Reruns can return cached responses.
- **Reasoning:** provider `reasoning_content` is stored in `model_response.reasonings` and never scored (`LE:lighteval/models/endpoints/litellm_model.py:300-302`). 3,622 stored rows have it.

## 11. Implications for jevemu

**Must match byte for byte** in any bank that claims evaluate-idk option parity:

1. Letters `A`–`E`. Each option is rendered as `"{L}) {text}"` with a single space. E is the literal `I don't know` with ASCII U+0027, always last and never gold. Gold is in A–D (§5). Jevemu's Choice criterion `"E": "I don't know"` (`docs/implementation/design.md:393`) must use the same ASCII string.
2. Option order comes from **one sequential stream**, not per-question seeding: `rng = random.Random(42)`, then for each row of the filtered split in dataset order, `opts = base.copy(); rng.shuffle(opts); gold = opts.index(correct)`. A single task is processed per stream. The base order is the dataset list for LEXam and `[Inc1, Inc2, Inc3, Correct]` (stripped) for GPQA. The question text is `.strip()`ed. This was verified 619/619 against stored LEXam prompts (§6).
   - The pre-audit bank design used only a per-question RNG seeded from (seed, question_id), which does **not** match. `docs/implementation/design.md:384` now adds an `order="evaluate_idk"` mode for the parity bank. Jevemu's own seeding remains the default and is still used for the permutation and robustness banks.
3. Pin dataset revisions and record them in the bank:
   - LEXam `85a95cf468b0`, or check the parquet sha256 `f4c10c42…57e9`. This reproduces the Dec 2025 stored runs.
   - GPQA must be pinned when first fetched, because upstream shows no revision.
   - Key questions by the dataset's own id (LEXam has a UUID `id` column), not by lighteval's positional `doc.id`.
4. Keep the stated scoring: +1 / 0 / −1. Map any jevemu failure to `extract_fail = 1` **and** `idk_score = −1`, `trad = 0`, `freq = 0`. That is evaluate-idk's per-model rule; do not use the ensemble script's "fail = 0" rule. Under P2 (A–D only), an abstention must be mapped to E before computing the evaluate-idk metrics.
5. Use the stderr convention: standard deviation with ddof = 1, divided by √n (§8).

**Differs by design, so the numbers are not comparable to the README:**

- Evaluate-idk's prompt demands chain-of-thought and `Final Answer: ###X###`, and LEXam adds a long legal chain-of-thought instruction. Jevemu's prompt has no chain-of-thought and uses an `Answer:` prefill.
- Evaluate-idk samples text at temperature 0 with up to 32k tokens and parses it with regexes. Jevemu returns typed letters or distributions, so its `extract_fail` is always 0.
- The README's GPQA figures come from an older prompt and shuffle (§1). Compare against the stored LEXam artifacts, or rerun GPQA with the pinned code.

**Risks for pairing:**

- Running both tasks in one lighteval process makes the second task's permutations depend on `PYTHONHASHSEED` (§6). The lighteval shim must run one task per process, or set `PYTHONHASHSEED`.
- The pre-audit design had the shim record "sha(choices)". In evaluate-idk `Doc.choices` is always `["A","B","C","D","E"]`, so that hash is constant and would detect nothing.
  - Hash the option texts rendered in `doc.query` instead: the block between `"\n\nChoices:\n"` and `"\n\nBefore answering"`.
  - Do not split that block on newlines to recover options. 6/198 GPQA questions have multi-line options (§4). Rebuild each `SystemOneRequest` from the pinned dataset row plus the `Random(42)` stream, then assert that the rendered block equals the substring in `doc.query`.
- Unpinned HF revisions: LEXam's test split changed 12 times from 2025-12-02 to 2025-12-06. GPQA's history is not visible without authentication.
- Importing `custom_tasks.py` calls `random.seed(42)` on the **process-global** RNG. Any jevemu code or test that uses the global `random` after that import is perturbed. Jevemu should only ever use private `Random` instances.

**Runtime conformance test** (design section "evaluate-idk integration (a)"). This part is static analysis only; the import was not executed, per the "skip builds/tests" constraint.

- **Import path.** `custom_tasks.py` is a plain file, not a package. Load it the way lighteval does (`LE:lighteval/tasks/registry.py:390-399`):
  `spec = importlib.util.spec_from_file_location("evaluate_idk_custom_tasks", "third_party/evaluate-idk/custom_tasks.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)`.
  Then call `mod.ExtractiveLetterIdkGrouped().compute(doc=Doc(query="", choices=list("ABCDE"), gold_index=g), model_response=ModelResponse(text=[f"Answer: {letter}"]))`. `Answer: X` goes through the priority-100 pattern, and so do the lighteval shim's outputs. Aggregate with `numpy.mean` and `lighteval.metrics.utils.stderr.mean_stderr`.
  Optional prompt-parity check: call `random.seed(42)` and then `mod.lexam_idk_prompt(row, "t")` for each filtered row in order, and compare `Doc.query` and `gold_index` with jevemu's bank. Restore the global RNG state afterwards.
- **Side effects on import:**
  1. `logging.basicConfig(level=$EVALUATE_IDK_LOG_LEVEL or INFO)` configures root logging if nothing has configured it yet (`custom_tasks.py:27`).
  2. `load_dotenv()` loads the first `.env` found by walking up from the directory of `custom_tasks.py`, or from the cwd under a debugger, REPL or coverage tracer. It will pick up `jevemu/.env` if one exists. It does not override existing variables (`DE:dotenv/main.py:300-319,331,353-354`).
  3. `import litellm` makes an HTTP GET to GitHub for the model cost map (5 s timeout, with a bundled fallback) unless `LITELLM_LOCAL_MODEL_COST_MAP=True` (`LL:litellm/__init__.py:339-341,420`; `LL:litellm/litellm_core_utils/get_model_cost_map.py:16-45`).
  4. `litellm.suppress_debug_info = True` is set globally (`custom_tasks.py:31`).
  5. The global RNG is reseeded with `random.seed(42)` (`custom_tasks.py:33`).
  6. Three `LightevalTaskConfig` objects and one metric object are built, with no I/O (`custom_tasks.py:399-464`).
- **Heavy dependencies.** The import needs the full locked environment, not just lighteval:
  - `lighteval.models.model_output` imports `torch` at module level (`LE:lighteval/models/model_output.py:25`).
  - `metrics_sample` imports `transformers`, `nltk`, `scipy` and `bert_scorer`, which also imports torch.
  - `metrics_corpus` imports `sacrebleu` and `sklearn`.
  - `lighteval_task` imports `datasets` and `multiprocess`.
  - `custom_tasks.py` itself imports `litellm` and `dotenv`.
  On linux x86_64 torch 2.8.0 pulls in CUDA 12 wheels (`uv.lock:4396`).
- **Recommendation.** Run the conformance test in a separate, opt-in environment built from `third_party/evaluate-idk/uv.lock` (Python 3.10–3.12), never from jevemu's own environment. Set `LITELLM_LOCAL_MODEL_COST_MAP=True`. Stub `dotenv.load_dotenv` before the import, or guarantee there is no `.env` in the submodule directory or any parent. An isolated cwd alone is not enough, because the search starts at the file's directory. Place the uv environment outside the submodule's working tree, for example with `UV_PROJECT_ENVIRONMENT`. That keeps the submodule clean, although `.venv` is already gitignored at `.gitignore:140` [INFERENCE].

**Design-doc corrections, now applied to `docs/implementation/design.md`:**

- Findings table (`docs/implementation/design.md:96-105`): the Framework, Entry point, Prompt (`Final Answer: ###X###`), IDK option, Metrics, Extraction, Datasets and Standard errors rows now record the audit's results. A new row says where the README numbers came from.
- Question bank (`docs/implementation/design.md:384`): adds the `order="evaluate_idk"` mode, pinned dataset revisions, and keying by dataset ID.
- Metrics (`docs/implementation/design.md:408`): no letter scores −1, and the identity `idk_score = 2·trad + freq − 1` holds.
- Sample size (`docs/implementation/design.md:423-429`): LEXam-en is 619, with \~41 items per bin and a floor of ≈ 0.056.
- Integration (`docs/implementation/design.md:445-446`): the conformance environment and import side effects, the shim hashing the rendered options, and one task per process.

## 12. Open items

| Item | Why unresolved |
| --- | --- |
| GPQA revision used in 2025-09, its row order, and the count at current `main` | `Idavidrein/gpqa` is gated. Unauthenticated tree and file requests returned 401, and the commits API requires authentication. n = 198 comes from stored details only. The user's HF credential was deliberately not used. |
| Exact GPQA permutations under the pinned code, and whether the stored GPQA prompts came from `random.seed(1234)` | This needs the gated CSV. The code path is the same one verified for LEXam, so it is [INFERENCE]. |
| Pinned GPQA message text `"…question.Question:\n…"` | Derived from the code (§5). No stored run uses the pinned GPQA prompt. |
| Python and library versions of the stored runs | Not recorded (`"versions": {}`, `"lighteval_sha": "?"`). Shuffle parity was verified on CPython 3.11.7. Identical results on other versions are [INFERENCE]. |
| Whether OpenRouter or the upstream providers honoured `temperature=0` for reasoning models | This happens on the provider side, and `drop_params=True` hides it. Nothing in the repo records it. |
| Import of `custom_tasks.py` in the locked environment | Not executed, because the assignment said to skip builds and tests. Feasibility and side effects were established from source (§11). |
| `idk-eval` task (`CatLaugh/idk_eval`) | Not in `tasks.txt` and has no stored results, so it is out of scope for jevemu parity. It was not audited beyond the task config. |

Reproduction notes, from throwaway scripts that were not committed:

- LEXam dataset parquets were fetched from `https://huggingface.co/datasets/LEXam-Benchmark/LEXam/resolve/<rev>/mcq_4_choices/test-00000-of-00001.parquet`.
- lighteval, litellm, datasets and python-dotenv wheels were fetched from the URLs in `uv.lock`.
- Details parquets were read with pandas and pyarrow in a scratch venv.
- Primary vs fallback extraction counts per stored LEXam run:

| Run | Primary | Fallback | None |
| --- | --- | --- | --- |
| gpt-5.2 | 19 | 600 | 0 |
| gemini-3-pro, Dec | 28 | 591 | 0 |
| gemini-3-flash | 36 | 583 | 0 |
| claude-sonnet-4.5, Dec | 59 | 560 | 0 |
| mistral-large-2512 | 197 | 417 | 5 |
| kimi-k2-thinking | 48 | 469 | 102 |
| intellect-3 | 87 | 531 | 1 |
