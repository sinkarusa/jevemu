# TODO

## Run the benchmark on macOS (not started)

So far every local number comes from vLLM 0.30.0 on one RTX 3090 under Linux. The idea: answer
the same frozen splits on Apple silicon and add them as separate points labelled by machine,
never mixed with the RTX 3090 ones. Notes so far, not yet verified:

- **What an engine must provide:**
  - top-k logprobs of the next token after an `Answer:` assistant prefill, taken before sampling
    (the softmax of the logits);
  - k large enough for the labels (10+ for the 6 shared benchmarks);
  - `tokenize`, for the trie on banking77/CLINC150;
  - thinking off.

  The deployed `auto_noecho` strategy needs neither echo nor constrained decoding.
- **llama.cpp `llama-server`** looks like the easiest first target. Render the chat with
  `/apply-template`, append the prefill, then call `/completion` with `n_predict: 1`, greedy and
  `n_probs: k` (greedy `n_probs` is documented as a plain softmax of the logits).
  `/tokenize`/`/detokenize` exist, and `-np N` slots give concurrency. This matches the vLLM
  backend's `/tokenize` + `/v1/completions` path, so a `LlamaCppHTTPBackend` next to
  `vllm_http.py` should be small. The server does not cap `n_probs`, so always send a fixed k.
- **oMLX** (MLX with continuous batching) accepts `logprobs`/`top_logprobs`, but released
  versions return them empty ([issue #1549](https://github.com/jundot/omlx/issues/1549)).
  Support is in [PR #1591](https://github.com/jundot/omlx/pull/1591), still open when last
  checked (2026-09-22). Worth revisiting once it ships, probably through the no-prefill
  first-token read the API backends support (`--strategy first_token`). `mlx_lm.server` is a
  fallback (top logprobs capped at 10).
- **Models:** the same families as the RTX 3090 finalists (Qwen3.6-27B, Qwen3.6-35B-A3B, a small
  Qwen3-4B), as GGUF `Q4_K_M`/`Q8_0` or MLX 4/8-bit. Different weights, and possibly a different
  embedded chat template, make them different models: score and calibrate them on their own runs.
- **Check first with a small probe:** raw vs post-sampling probabilities, prefill continuation
  with the Qwen3.x templates, label tokenization, nondeterminism under batching and prompt
  caching, agreement with the vLLM runs on the same items, and questions per second against
  concurrency.
- **Reporting:** record the machine (chip, memory, macOS, engine build) with every run. Keep the
  caveat that local speed depends heavily on hardware, engine and concurrency.
