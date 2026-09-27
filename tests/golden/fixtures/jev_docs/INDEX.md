# Jev doc fixtures

Examples captured verbatim from https://docs.typesafe.ai (markdown sources `<page>.md`, fetched 2026-09-24).

Capture rules:

- **Fenced JSON** (` ```json ` blocks, plus the heredoc body of the quickstart cURL command): raw text written unchanged apart from dedent/trim. Number spellings such as `0.0`, `1.0` and `3.0` are kept exactly as written.
- **`<TypesafeExample>` requests:** most request examples on the primitives, advanced and fan-out pages are not fenced JSON. They are JS object literals passed to a MDX component, which renders `JSON.stringify({state, questions}, null, 2)` as a JSON code block titled "request". Each fixture is exactly that rendered JSON, so it has **no `model` key**. The component sends `selectedModels` (`['jev-latest']` when present) only to the Playground link and does not render it. JS keys `true:`/`false:` render as JSON string keys `"true"`/`"false"`. None of these literals contain numbers, comments or elisions.
- No fixture needed any fix for comments, trailing commas or `...` elisions. None of the captured examples contained them.
- Confidence check: `formula` = `(K*pmax-1)/(K-1)`, computed on the probabilities as displayed. All `probabilities` maps sum to exactly 1.00 at displayed precision unless noted. In every Choice answer, `choice` is the argmax of `probabilities`. In every Score answer, `score` equals the expected value `Σ level*p`, computed on the displayed probabilities.
- In responses, the key order of Choice `probabilities` differs from the request's `criteria` order (it looks shuffled). Score `probabilities` and `legend` keys are the strings `"0"`..`"K-1"`, in order.

## Fixtures

| File | Source URL | Section | Kind | Pair | Notes |
| --- | --- | --- | --- | --- | --- |
| api__01__request.json | https://docs.typesafe.ai/api | Request body | request | api__01__response.json | string state; noul without criteria; model `jev-latest` |
| api__01__response.json | https://docs.typesafe.ai/api | Response body | response | api__01__request.json | noul 0.95; usage 296/20 |
| api__02__request.json | https://docs.typesafe.ai/api | Question types › Noul | request | api__02__response.json | noul criteria `{"true","false"}` with string values |
| api__02__response.json | https://docs.typesafe.ai/api | Answer types › Noul answer | response | api__02__request.json | same answer as 01 but usage 307/20 (noul criteria add tokens) |
| api__03__request.json | https://docs.typesafe.ai/api | Question types › Choice | request | api__03__response.json | choice, 3 string criteria |
| api__03__response.json | https://docs.typesafe.ai/api | Answer types › Choice answer | response | api__03__request.json | key order choice, probabilities, confidence (confidence last); probs 0.88/0.12/0.0; **confidence 0.81 does not match (K*pmax-1)/(K-1) = 0.82** (off by 0.01; fits an unrounded pmax in [0.875, 0.8775)) |
| api__04__request.json | https://docs.typesafe.ai/api | Question types › Score | request | api__04__response.json | score criteria = array of 3 strings |
| api__04__response.json | https://docs.typesafe.ai/api | Answer types › Score answer | response | api__04__request.json | score 1.05 = E; confidence 0.92 vs formula 0.925 (half-way case, rounds down); key order score, legend, probabilities, confidence |
| introduction_quickstart__01__request.json | https://docs.typesafe.ai/introduction/quickstart | Call it: the API › Sample cURL command | request | none | extracted from the `-d @- <<'EOF'` heredoc of the cURL command and dedented; no response shown |
| introduction_quickstart__02__request.json | https://docs.typesafe.ai/introduction/quickstart | Request body | request | introduction_quickstart__02__response.json | choice + score + noul in one request; choice instructions have no trailing `?` |
| introduction_quickstart__02__response.json | https://docs.typesafe.ai/introduction/quickstart | Response body | response | introduction_quickstart__02__request.json | choice probs 0.85/0.0/0.15: confidence 0.78 vs formula 0.775; score 1.0 with probs 0.0/1.0/0.0, confidence 1.0; **noul written as `1.0`**; usage 392/65 |
| primitives__01__request.json | https://docs.typesafe.ai/primitives | Ask multiple questions together | request | none | TypesafeExample, **no `model`**, no `selectedModels` either; choice + noul + score |
| primitives_choice__01__request.json | https://docs.typesafe.ai/primitives/choice | Request structure | request | primitives_choice__01__response.json | TypesafeExample, **no `model`** (selectedModels jev-latest) |
| primitives_choice__01__response.json | https://docs.typesafe.ai/primitives/choice | Response structure | response | primitives_choice__01__request.json | probs 0.0/1.0/0.0, confidence 1.0; usage 328/34 |
| primitives_choice__02__request.json | https://docs.typesafe.ai/primitives/choice | A more complex example | request | primitives_choice__02__response.json | TypesafeExample, **no `model`**; 5 choice questions; `tone` criteria values are all **null** |
| primitives_choice__02__response.json | https://docs.typesafe.ai/primitives/choice | A more complex example | response | primitives_choice__02__request.json | department 0.61 max, conf 0.42 vs formula 0.415; return_reason K=5 conf 1.0; shipping_issue pmax 0.74, **conf 0.67 vs formula 0.675**; requested_resolution K=4 pmax 0.4, conf 0.2 = formula; tone pmax 0.84, conf 0.76 = formula; usage 589/212 |
| primitives_choice__03__request.json | https://docs.typesafe.ai/primitives/choice | Structured instructions and criteria | request | primitives_choice__03__response.json | TypesafeExample, **no `model`**; **object instructions** `{question, focus}`; **object criteria values** `{what, not_for, examples[]}`; K=2 |
| primitives_choice__03__response.json | https://docs.typesafe.ai/primitives/choice | Structured instructions and criteria | response | primitives_choice__03__request.json | K=2, probs 0.0/1.0, confidence 1.0; usage 407/32 |
| primitives_score__01__request.json | https://docs.typesafe.ai/primitives/score | Request structure | request | primitives_score__01__response.json | TypesafeExample, **no `model`** |
| primitives_score__01__response.json | https://docs.typesafe.ai/primitives/score | Response structure | response | primitives_score__01__request.json | score 1.43 = E; probs 0.0/0.57/0.43; confidence 0.35 vs formula 0.355; usage 332/18 |
| primitives_score__02__request.json | https://docs.typesafe.ai/primitives/score | Splitting a complex judgment into several Score questions | request | primitives_score__02__response.json | TypesafeExample, **no `model`**; 3 scores, one with 4 levels |
| primitives_score__02__response.json | https://docs.typesafe.ai/primitives/score | Splitting a complex judgment into several Score questions | response | primitives_score__02__request.json | severity conf 0.64 = formula; frustration conf 0.58 = formula; report_quality K=4, **score written `3.0`**, confidence 1.0; usage 468/43 |
| primitives_score__03__request.json | https://docs.typesafe.ai/primitives/score | Structured level descriptions | request | primitives_score__03__response.json | TypesafeExample, **no `model`**; **score criteria are objects** `{what, examples[]}` |
| primitives_score__03__response.json | https://docs.typesafe.ai/primitives/score | Structured level descriptions | response | primitives_score__03__request.json | **non-string legend values**: legend echoes the level objects `{what, examples}`, which contradicts api.md's `legend: map<string, string>`; score 1.09 = E; conf 0.87 vs formula 0.865; usage 379/18 |
| primitives_noul__01__request.json | https://docs.typesafe.ai/primitives/noul | Request structure | request | primitives_noul__01__response.json | TypesafeExample, **no `model`**; one noul without criteria, one with string `true`/`false` criteria |
| primitives_noul__01__response.json | https://docs.typesafe.ai/primitives/noul | Response structure | response | primitives_noul__01__request.json | noul 0.99, 0.93; usage 360/39 |
| primitives_noul__02__request.json | https://docs.typesafe.ai/primitives/noul | Structured instructions | request | primitives_noul__02__response.json | TypesafeExample, **no `model`**; **object state** (nested resume with array); **object instructions** `{potential_duplicate{...}, question}` |
| primitives_noul__02__response.json | https://docs.typesafe.ai/primitives/noul | Structured instructions | response | primitives_noul__02__request.json | noul 0.74, 0.09, 0.08; usage 535/58 |
| primitives_advanced__01__request.json | https://docs.typesafe.ai/primitives/advanced | Structured instructions | request | none | TypesafeExample, **no `model`**; object state; object instructions on all 4 questions (nested `field` object); **choice criteria all null**; 2 scores with 5 levels |
| primitives_advanced__02__request.json | https://docs.typesafe.ai/primitives/advanced | JSON rubric for boundary clarification | request | none | TypesafeExample, **no `model`**; object instructions; object choice criteria values `{what, not_for, examples[]}` |
| primitives_advanced__03__request.json | https://docs.typesafe.ai/primitives/advanced | Walking a taxonomy | request | none | TypesafeExample, **no `model`**; choice criteria values are **nested objects of arrays** and one **array** value; option keys contain spaces and `&` |
| primitives_advanced__04__request.json | https://docs.typesafe.ai/primitives/advanced | Structured Score levels | request | none | TypesafeExample, **no `model`**; object instructions `{question, note}`; score levels are objects `{summary, signals[]}` |
| primitives_advanced__05__request.json | https://docs.typesafe.ai/primitives/advanced | Structured Noul criteria | request | none | TypesafeExample, **no `model`**; object state; object instructions `{question, inspect, focus}`; **noul criteria true/false are objects** `{what, examples[]}` |
| patterns_fan-out__01__request.json | https://docs.typesafe.ai/patterns/fan-out | Step 1: speculative fan-out | request | none | TypesafeExample with `display="questions"`: **the page renders only the questions map**. The fixture is `{state, questions}` from the component's `example` prop (the Playground link encodes the same data); **no `model`**, no selectedModels; 5 mixed questions |

Totals: 21 request fixtures (6 with `model`, 15 TypesafeExample without `model`) and 13 response fixtures, 34 in all. No GET /v1/models response fixture: models.md gives only a cURL/SDK call and the field schema, with no example JSON body.

## Skipped (partial fragments / non-HTTP)

- api.md "Question types": bare `"instructions": {...}` object fragment (object with nested `potential_duplicate` object and a `question` field).
- primitives/advanced.md "Structured instructions" (fenced, around L345): bare `"instructions": {question, compare[], focus}` fragment.
- introduction/quickstart.md "Try it: the Playground": bare questions map `{"urgency": {...}}` (Playground input, not an HTTP body).
- patterns/confidence-routing.md "Step 1": TypesafeExample with only `questions` (no state), which is a questions-only fragment. It is a choice `intent` with string criteria check_balance/approve_transfer/other.
- concepts/state.md and primitives.md "Reference specific fields": a JSON *state value* only (ticket/order/refund_policy object), not a request body.
- primitives/score.md `ScoreExplorer` component: 5 answer-only Score objects (no model/usage). Their confidences are notable: formality `probs 0/.14/.86/0/0`, **conf 0.89 vs formula 0.825**; relevance `probs 0/0/.48/.52`, **conf 0.52 vs formula 0.36**; severity/frustration/detail agree with the fenced responses (0.35, 0.61≈formula 0.61, 1.0). Widget data only, so the mismatches may be illustrative, but they show that Score confidence may not use the Choice formula.
- All Python/TypeScript SDK snippets, mermaid diagrams, and plaintext pseudo-examples (e.g. choice.md option lists, score.md "Writing good levels" `score 0.55, confidence 0.33, probabilities 0: 0.45, 1: 0.55, 2: 0.0`).
- models.md, confidence.md, model-jaggedness/jev-1.13.md contain no request/response JSON.

## Schema-relevant facts

### models.md (https://docs.typesafe.ai/models)

- Versioned model ID: `jev-1.13.0` (display name "Jev 1.13"). All example responses report `"model": "jev-1.13.0"`.
- Aliases: `jev-latest` → `jev-1.13.0` ("most recent stable, official release", SDK default, used by all doc examples). `jev-preview` → `jev-1.13.0` ("currently points to the same model as `jev-latest`").
- "The response's `model` field reports the versioned ID that answered". Requests send the alias; responses return the versioned ID.
- Context length: "64k tokens per request; 32k tokens for `state` plus the longest question". "The 64k budget covers the `state` plus all questions combined; the 32k budget applies to the `state` plus the single longest question."
- Input: "Text only. String, JSON object, or array of text values." (concepts/state.md: "State must be a string, JSON object, or array of text values"; api.md types `state` as `string | object | array`.)
- Rate limits: 250,000 tokens/s and 1,200 requests/min. Over the limit → `429 Too Many Requests`. SDKs honor the `retry-after` header.
- Pricing is per input token; output tokens are free.
- `GET /v1/models` → `{"models": [{"name": string, "description": string, "release_date": string}]}`, all required. "It currently lists the aliases. Versioned IDs such as `jev-1.13.0` are accepted by the `model` field whether or not they appear in the list." No example body is given, and the format of `release_date` is not specified.
- Elsewhere: model-jaggedness/jev-1.13.md uses `TypeSafeClient(model="jev-1.13")` in a Python snippet. That is a third spelling, not listed as an ID or alias.

### api.md limits and types (https://docs.typesafe.ai/api)

- Choice: `criteria` is a required `map<string, string | object | array | null>`. "You can have a maximum of 255 options per Choice." (choice.md: "A Choice question accepts up to 255 options".)
- Score: `criteria` is a required `array<string | object | array>`. "A Score should have at least two levels; the API accepts up to 10." (score.md says the same.)
- Noul: `criteria` is an optional object with keys `true`/`false`, each `string | object | array`.
- `instructions`: `string | object | array`, required. **Conflict:** the primitives/advanced.md table says `instructions`, Choice criteria values, Score level entries and Noul `true`/`false` all accept "`string`, `object`, `array`, or `null`". api.md allows null only for Choice criteria values.
- The question id (map key) "is not sent to the underlying model and is not used in inference."
- Response: `model` string, `answers` map, `usage {input_tokens: integer, output_tokens: integer}`, all required.
- Choice answer: `choice` string ("The highest-probability option"), `probabilities` map<string, number> ("floats that sum to 1"), `confidence` number.
- Score answer: `score` number ("probability-weighted answer across the levels; can land between levels"), `legend` `map<string, string>` ("Each level number mapped back to its description"; **contradicted by primitives_score__03__response.json, where legend values are objects**), `probabilities` map<string, number> ("Each level (string key) mapped to its probability (floats that sum to 1)"), `confidence` number.
- Noul answer: `noul` number, "on a scale from 0 (no) to 1 (yes)". No confidence field.
- "Choice and Score answers also carry a `confidence` between 0 to 1, derived from the answer's probability distribution."
- Errors: 401, 422 ("request body failed validation … The body details the offending field"), 429, 529 Overloaded. The format of the JSON error body is not specified.

### confidence.md (https://docs.typesafe.ai/confidence)

- Quote: "The answer's `confidence` property collapses that shape into a single number from 0 to 1, so you can threshold on it without doing the math yourself. (Noul answers don't carry one.)"
- Quote: "`confidence` is a statistic computed from the probability distribution the answer already gives you. TypeSafe computes it for you and returns it on every Choice and Score answer".
- Demo formula (quote): "TypeSafe computes confidence from how the probability is spread across the options. All of it on one option gives 1.0; the more evenly it spreads, the lower the confidence. This demo uses <code>(3 × largest probability − 1) / 2</code> to approximate confidence for three options." The demo code is `Math.max(0, Math.min(1, (count * peak - 1) / (count - 1)))`, i.e. `(K*pmax-1)/(K-1)` clamped to [0,1]. The docs call it an *approximation*.
- "For a Choice, the distribution is `probabilities` across your options. For a Score, it is the distribution across your levels."
- Rounding/precision: no page gives a rounding rule. All example probabilities, confidences, scores and nouls use ≤2 decimal places. Integral values are written as `0.0`/`1.0`/`3.0`. Confidence differs from the formula on displayed probabilities by ≤0.005 (half-way cases round both ways: 0.775→0.78, 0.415→0.42, 0.675→0.67, 0.925→0.92, 0.355→0.35, 0.865→0.87), plus one 0.01 case (api__03). This suggests confidence is computed from unrounded probabilities and then rounded to 2 dp [INFERENCE]. model-jaggedness/jev-1.13.md: "`jev-1.13`'s score levels are weak in numerical calibration".
