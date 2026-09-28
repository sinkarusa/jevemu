# Reports

Generated results with tables and figures, rendered for GitHub. Each report states its inputs, and
every number in it is also in the report's `metrics.json`. Methods and experiment logs are in
[`docs/research/`](../docs/research/).

| Report | What it shows |
| --- | --- |
| [Jev vs the Qwen3.6 emulators](jev_vs_qwen/README.md) | Jev `jev-1.13.0` against two emulators on one RTX 3090: the selected dense configuration (QuantTrio Qwen3.6-27B AWQ, calibrated) and the tuned fast MoE (mixture-of-experts) Qwen3.6-35B-A3B. Quality on the `holdout` half: accuracy, NLL (negative log-likelihood), Brier score, ECE (expected calibration error) and reliability diagrams, overall and per benchmark, raw and calibrated. Tokens, latency and cost on the `select` half. Yelp excluded (9 benchmarks). |
| [Speed/quality trade-off](speed_quality/README.md) | Every emulator candidate (the 16 Qwen presets of the selection, the Gemma 4 26B-A4B MoE, and small models of 2B to 12B) on the `screen` split, all with the same settings. Macro accuracy, ECE and Brier (raw and cross-fitted calibration) against throughput (q/s, questions per second, at 16 in flight). It also shows the Pareto frontier, the best candidate per speed tier, the accuracy cost of each step down, and reliability diagrams for four systems (best small model, best MoE, best dense 27B, Jev). Yelp excluded (9 benchmarks). |
| [Summary report](summary/index.html) (HTML, interactive) | The main results on one page, with interactive Plotly figures. Accuracy and Brier score against questions per second (or tokens per question), for every candidate plus CLM-8B, a dual encoder that answers Jev's wire format (`screen`), and for the finalists (`holdout`: Jev, the dense Qwen3.6-27B emulator, the fast Qwen3.6-35B-A3B emulator in its AWQ and GPTQ builds, the Gemma 4 26B-A4B MoE, GPT-6 Luna, DeepSeek V4.1 Flash). It also shows paired differences against Jev, cost, where each speed limit comes from and the ceiling without it, how calibration works, and reliability diagrams (pooled and per dataset, raw and calibrated). A toggle switches between the 6 benchmarks every system answers and all 9. GitHub shows only the source: download the file and open it in a browser (plotly.js loads from its CDN). |

## Regenerate

```bash
uv run --extra plot python scripts/make_report.py
uv run --extra plot python scripts/make_tradeoff_report.py
uv run python scripts/make_summary_report.py
```

Each script:

- reads the gitignored `runs/` artifacts listed in its docstring (run directories).
  `make_report.py` also reads the holdout calibration study; `make_tradeoff_report.py` and
  `make_summary_report.py` compute their cross-fitted calibration themselves. The summary report also reads
  `speed_quality/metrics.json` for its list of candidates.
- stops with the missing path if an input is absent, or with the mismatch if inputs disagree
  (for example, fast-emulator records without their study). It also stops if any local
  emulator answer took more than one model call (diagnostics `n_backend_calls`): every report
  uses one-call scoring (`auto_single`).
- rewrites its report directory: `README.md`, `metrics.json` and `figures/*.png`, or for the summary report
  `index.html` and `metrics.json`.

The same inputs give byte-identical output. Only aggregates are written: no question text and no
per-item outputs.
