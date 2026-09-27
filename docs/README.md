# Documentation

## Design

- [implementation/design.md](implementation/design.md): architecture, the Jev contract, scoring strategies (including one-call scoring, `auto_single`), calibration design and evaluation protocol.

## Research

Experiments and findings, in the order they were run.

- [research/evaluate_idk_audit.md](research/evaluate_idk_audit.md): code audit of evaluate-idk, a benchmark of whether models say "I don't know" (IDK) when they should. Covers the prompt, shuffle, answer extraction, metrics and datasets that the question bank must match.
- [research/vllm_probe_report.md](research/vllm_probe_report.md): what the pinned vLLM server does with constrained decoding and logprobs, per model.
- [research/quantization_report.md](research/quantization_report.md): Qwen3.5-9B in bf16 vs FP8, INT8 and INT4. Does quantization change accuracy or label probabilities by more than run-to-run noise?
- [research/candidates.md](research/candidates.md): which Qwen checkpoints load and answer on one RTX 3090, with memory and startup measurements.
- [research/selection.md](research/selection.md): choosing the emulator configuration (model, strategy, layout) on the `select` half, with cost and token statistics. Also the API models on the same protocol.
- [research/calibration.md](research/calibration.md): debiasing, cross-fitted calibrators and Jev's confidence function, measured on `holdout`. Includes the recommended deployment.
- [research/openai_probe_report.md](research/openai_probe_report.md): what the OpenAI API (GPT-6 Luna) does with logprobs, reasoning off, prefill, Structured Outputs, caching and nondeterminism.
- [research/openrouter_probe_report.md](research/openrouter_probe_report.md): the same for OpenRouter (DeepSeek V4.1 Flash on Makora; why not Wafer), plus billing.

## Reports

- [../reports/README.md](../reports/README.md): index of the generated result reports, including the interactive [summary report](../reports/summary/index.html).

## Open ideas

- [TODO.md](TODO.md): ideas not started yet, such as running the benchmark on macOS (llama.cpp or MLX).
