# Candidate models that fit one RTX 3090

This is a load-and-sanity pass over Qwen-family instruct checkpoints for one RTX 3090, not the
model selection itself. All 16 presets load with an 8,192-token context and thinking off, and
answer 20 smoke items sensibly, so all 16 go on to the selection runs. A second pass added small
(2B-12B) models from Qwen and other families for the speed/quality study; there, Ministral 3
cannot be served by the pinned image. A last addition, Gemma 4 26B-A4B (a mixture-of-experts
model with 3.8B active parameters), loads with the Gemma 4 12B flags unchanged.

Measured on 2026-09-25 against the pinned server: vLLM 0.30.0, image
`vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`,
RTX 3090 24 GiB, driver 580.173.02. The question: which Qwen-family instruct checkpoints load on
one 24 GiB GPU with `--max-model-len 8192`, answer without chain-of-thought, and are worth the
model-selection runs on the `select` half of the banks. 20 items cannot rank models (see below).

## Conclusion

On the 20 smoke items, accuracy was 0.30-0.50, against Jev's 0.55 on the same items (chance is
0.20). Nothing that was downloaded was dropped. One preset needed a larger GPU share
(`qwen3.6-27b-int4-quanttrio`, below). No candidate exposed a backend bug.

| Preset | Checkpoint @ revision | Format (quantizer) | Weights GiB | KV cache at 8,192 max-model-len | Startup s | 20-item acc. (sequential / concurrency 16) | Items/s (concurrency 16) | Jev agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.8-27b-awq` (existing) | `cyankiwi/Qwen3.8-27B-AWQ-INT4` @ `6e134bae81` | compressed-tensors W4A16, asym, g32 (cyankiwi) | 18.37 | 1.33 GiB, 13,010 tokens | 129 | 0.35 / 0.35 | 4.45 | 10/20 |
| `qwen3.8-27b-int4-redhat` | `RedHatAI/Qwen3.8-27B-INT4` @ `2f0fa48fea` | compressed-tensors W4A16, sym, g128 (LLM Compressor AWQ + GPTQ) | 16.84 | 2.88 GiB, 28,912 tokens | 138 | 0.45 / 0.40 | 5.08 | 14/20 |
| `qwen3.8-27b-int4-palmfuture` | `palmfuture/Qwen3.8-27B-GPTQ-Int4` @ `d85a556c59` | GPTQ, sym, g32 (GPTQModel 7.0.0) | 17.88 | 1.72 GiB, 17,347 tokens | 135 | 0.30 / 0.30 | 5.08 | 8/20 |
| `qwen3.8-27b-int4-btbtyler09` | `btbtyler09/Qwen3.8-27B-GPTQ-4bit` @ `b64f44dfed` | GPTQ, sym, g32 (GPTQModel 6.0.3) | 17.88 | 1.72 GiB, 17,347 tokens | 135 | 0.35 / 0.35 | 5.08 | 8/20 |
| `qwen3.8-27b-int4-groxaxo` | `groxaxo/Qwen3.8-27B-GPTQ-Pro-4bit-g64-calib128` @ `9722fd1acb` | GPTQ, sym, g64 (GPTQ-Pro, GPTQModel 6.1.0-dev) | 17.20 | 2.42 GiB, 24,094 tokens | 135 | 0.35 / 0.35 | 5.08 | 11/20 |
| `qwen3.6-27b-int4-cyankiwi` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` @ `e5cc0400fb` | compressed-tensors W4A16, asym, g32 (cyankiwi) | 18.32 | 1.39 GiB, 13,974 tokens | 132 | 0.45 / 0.45 | 5.01 | 9/20 |
| `qwen3.6-27b-int4-quanttrio` | `QuantTrio/Qwen3.6-27B-AWQ` @ `9b507bdc9a` | AutoAWQ GEMM, zero point, g128, MLPs + DeltaNet + o_proj only (QuantTrio) | 19.05 | 1.14 GiB, 11,083 tokens (0.97 of the GPU) | 125 | 0.45 / 0.45 | 5.07 | 6/20 |
| `qwen3.6-27b-int4-groxaxo` | `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit` @ `95bfcf97a5` | GPTQ, sym, g128 (GPTQ-Pro, GPTQModel) | 16.83 | 2.81 GiB, 27,949 tokens | 147 | 0.35 / 0.35 | 5.10 | 6/20 |
| `qwen3.5-35b-a3b-int4` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (official) @ `3af5ca2972` | GPTQ, sym, g128, routed experts only (Qwen) | 20.27 | 0.72 GiB, 20,480 tokens | 144 | 0.30 / 0.30 | 25.99 | 7/20 |
| `qwen3.6-35b-a3b-int4-palmfuture` | `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` @ `00a6698351` | GPTQ, sym, g128, routed experts only (GPTQModel 6.0.3) | 20.27 | 0.72 GiB, 20,480 tokens | 132 | 0.50 / 0.45 | 26.59 | 7/20 |
| `qwen3-32b-int4` | `Qwen/Qwen3-32B-AWQ` (official) @ `0499c3ac83` | AutoAWQ GEMM, zero point, g128 (Qwen) | 18.24 | 2.43 GiB, 9,936 tokens | 89 | 0.50 / 0.50 | 5.39 | 10/20 |
| `qwen3-30b-a3b-2507-int4-redhat` | `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` @ `e9c59cdcca` | compressed-tensors W4A16, sym, g128 (llm-compressor GPTQ) | 15.68 | 4.21 GiB, 45,936 tokens (0.90) | 89 | 0.35 / 0.45 | 32.13 | 2/20 |
| `qwen3-14b-int4` | `Qwen/Qwen3-14B-AWQ` (official) @ `31c69efc29` | AutoAWQ GEMM, zero point, g128 (Qwen) | 9.44 | 10.11 GiB, 66,224 tokens (0.90) | 74 | 0.40 / 0.40 | 12.54 | 8/20 |
| `qwen2.5-32b-int4` | `Qwen/Qwen2.5-32B-Instruct-AWQ` (official) @ `5c7cb76a26` | AutoAWQ GEMM, zero point, g128 (Qwen) | 18.14 | 2.52 GiB, 10,336 tokens | 68 | 0.45 / 0.45 | 5.51 | 5/20 |
| `qwen3.5-9b-bf16` (existing) | `Qwen/Qwen3.5-9B` (official) @ `c202236235` | bf16 | 16.80 | 2.28 GiB, 52,503 tokens (0.90) | 110 | 0.35 / 0.35 | 10.70 | 6/20 |
| `qwen3.5-9b-int8` (existing) | `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8` @ `c798ade7cb` | compressed-tensors W8A16, sym, g32 (cyankiwi) | 12.19 | 6.84 GiB, 157,882 tokens (0.90) | 107 | 0.35 / 0.35 | 10.60 | 6/20 |

How to read the table:

- Weights and KV cache are vLLM's own log lines ("Model loading took", "Available KV cache
  memory", "GPU KV cache size"). The GPU share is 0.95 unless noted.
- Startup is seconds from `docker compose up` to a healthy `/health` (3 s poll).
- Accuracy counts the IDK ("I don't know") option as wrong.
- "Jev agreement" is how many of the 20 sequential emulator choices equal Jev's (`jev-1.13.0`,
  from the response cache).
- All checkpoints are Apache-2.0 per their model cards.

Findings:

1. **The 20 items do not rank models.** At these accuracies one standard error is about 0.11.
   The same preset scored 0.40 in the earlier 20-item smoke run and 0.35 here. Three presets
   changed accuracy between the sequential and the concurrency-16 pass on the same server,
   because batch composition moves logprobs (see the quantization report's noise floors). Rank
   on the `select` halves.
2. **Qwen3.5-35B-A3B and Qwen3.6-35B-A3B fit, as 4-bit builds that keep everything but the
   routed experts in bf16.** Their 20.27 GiB of weights leave only 0.72 GiB of KV cache at 0.95.
   But these mixture-of-experts (MoE) models keep KV in only 10 of 40 layers, with 2 KV heads,
   so that still holds 20,480 tokens (2.5 concurrent 8,192-token requests). They are the fastest
   Qwen3.5-architecture candidates: 26-27 items/s at concurrency 16, against about 5 for every
   dense 27B build. Qwen3-30B-A3B-2507 is faster still (32 items/s).
3. **Dense 27B builds: 16.8-19.1 GiB.**
   - The group-64/128 builds (RedHatAI g128, groxaxo g64 and g128) leave 2.4-2.9 GiB of KV cache
     (24-29k tokens). The group-32 builds (cyankiwi, palmfuture, btbtyler09) leave 1.3-1.7 GiB
     (13-17k tokens).
   - `QuantTrio/Qwen3.6-27B-AWQ` keeps attention q/k/v and layer 0 in bf16 (19.05 GiB). At 0.95
     it had 0.62 GiB of KV cache, short of the 0.81 GiB one 8,192-token request needs, so vLLM
     refused to start. Its preset takes 0.97 of the GPU and `--max-num-batched-tokens=512`,
     which gives 1.14 GiB (11,083 tokens).
   - Throughput is flat across the new 27B presets (5.0-5.1 items/s): Marlin costs the same here
     at group 32, 64 and 128. The existing `qwen3.8-27b-awq` measured 4.45. Its only launch
     difference is the missing `--generation-config=vllm` (so the checkpoint's top-k/top-p
     sampling defaults apply), but it was also the first run of the session, so the cause was
     not isolated.
4. **Dense 32B baselines** (Qwen3-32B, Qwen2.5-32B) keep KV in all 64 layers with 8 heads (256
   KiB per token). So 2.4-2.5 GiB holds only about 10k tokens: 1.2 concurrent 8,192-token
   requests.
5. **Echo (S4) is safe on every preset at concurrency 16.** The runner's echo pass (20 items, 16
   in flight, letter keys) completed without error on the MoE builds, on
   `qwen3.6-27b-int4-quanttrio` at 0.97, on the Qwen3 and Qwen2.5 baselines and on the 9B
   presets. The pass was added after the other 27B presets had run, so for those six new 27B
   presets this is [INFERENCE]: the prompt-logprob spike that the pass guards against scales with
   the vocabulary and `--max-num-batched-tokens`, not the model. Those presets use the same
   vocabulary, batch limit and 0.95 share as the 35B presets that passed, and `qwen3.8-27b-awq`
   also ran S4 on all 198 GPQA items in the quantization report.
6. **Thinking is off and nothing is injected.**
   - Every Qwen3.5-architecture and Qwen3 hybrid preset renders
     `<|im_start|>assistant\n<think>\n\n</think>\n\n` after the user turn. The system message is
     unchanged (no Qwen3.8 reasoning-effort line when thinking is off).
   - Qwen3-30B-A3B-Instruct-2507 has no thinking mode and renders a bare assistant turn.
   - Qwen2.5's template adds "You are Qwen, created by Alibaba Cloud. You are a helpful
     assistant." when a conversation has no system message (seen in the user-only render).
     The `question_first` layout (jevemu's default at the time) always sends one; `state_first`,
     the default now, does not.
   - The Qwen3.5-architecture templates trim the prefill's trailing space; the Qwen3/Qwen2.5
     ones do not. Either way, the adapter's rendering keeps it (checked per preset).
7. **`RedHatAI/Qwen3.8-27B-INT4` declares an FP8 KV cache** (`kv_cache_scheme`). vLLM 0.30.0
   applies it when `--kv-cache-dtype` is `auto`, so this preset passes
   `--kv-cache-dtype=bfloat16` to match every other preset's KV cache.
8. **Downloads: 13 checkpoints, 239.3 GiB (256.9 GB)**, slightly over the planned ~250 GB. So
   no AutoRound build was downloaded. Next in line were `Intel/Qwen3.6-27B-int4-AutoRound`
   (17.7 GiB) and `Frozenlock/Qwen3.8-27B-int4-AutoRound` (17.7 GiB). The disk now has about
   69 GB free.

## Preset flags

The new presets differ from each other only where noted here.

- **Qwen3.5-architecture checkpoints** (all 27B and 35B-A3B builds) share the `qwen3.8-27b-awq`
  launch, plus `--generation-config=vllm` and `--dtype=bfloat16` (the base models are bf16;
  several uploads declare float16). That launch is: `--language-model-only` (no vision tower),
  `--max-num-seqs=16`, `--max-num-batched-tokens=1024` (512 for QuantTrio),
  `enable_thinking=false` by server default, and 0.95 of the GPU (0.97 for QuantTrio).
- **Text-only baselines** (Qwen3, Qwen2.5) drop `--language-model-only` and keep the
  checkpoint's dtype (fp16 for the AutoAWQ uploads, bf16 for RedHatAI's). They use 0.95 of the
  GPU for the 32B models and 0.90 for the smaller ones.
- `qwen3.8-27b-int4-redhat` adds `--kv-cache-dtype=bfloat16` (finding 7).
- The MTP (multi-token prediction) heads are never loaded (no speculative decoding).

The table below shows what each build quantizes (language model only, from the safetensors
headers). Embeddings and `lm_head` stay 16-bit in all of them.

Unquantized tensors are stored in bf16, except in `cyankiwi/Qwen3.6-27B-AWQ-INT4` and
`Qwen/Qwen2.5-32B-Instruct-AWQ`, which store them and their quantization scales in fp16.
Qwen2.5 is served in fp16. The cyankiwi Qwen3.6 preset is served in bf16 like the other
Qwen3.5-architecture presets, so vLLM casts those fp16 tensors to bf16 ("Casting
torch.float16 to torch.bfloat16"), which rounds the fp16 scales. On the 9B ladder, fp16 and bf16
activations were indistinguishable for QuantTrio's INT4
(`docs/research/quantization_report.md`). If it matters, a copy of the preset with
`JEVEMU_VLLM_DTYPE=auto` serves this build in fp16.

| Build | 4-bit | 16-bit |
| --- | --- | --- |
| cyankiwi 27B, RedHatAI 27B, palmfuture 27B, btbtyler09 27B, groxaxo 27B (3.8 and 3.6) | attention q/k/v/o, DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`, MLPs | DeltaNet `in_proj_a`/`in_proj_b` (cyankiwi 3.8 also layer 0's `out_proj`) |
| QuantTrio 3.6-27B | attention `o_proj`, DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`, MLPs of layers 1-63 | attention q/k/v, DeltaNet `in_proj_a`/`in_proj_b`, layer 0 |
| Qwen 3.5-35B-A3B, palmfuture 3.6-35B-A3B | routed experts | attention, DeltaNet, shared expert and its gate, router |
| RedHatAI 30B-A3B-2507 | attention, routed experts, router | none |
| Qwen3-32B, Qwen3-14B, Qwen2.5-32B (AutoAWQ) | attention, MLPs | none |

Linear kernels (from the vLLM log): `MarlinLinearKernel` for every dense build
(CompressedTensorsWNA16, AutoGPTQLinearMethod, AutoAWQMarlinLinearMethod). The MoE builds use the
Marlin WNA16 MoE backend (`MarlinExperts`). Attention backend: FlashAttention everywhere.

## Method

**Discovery.** The Hugging Face API listed every Qwen3.8/3.6/3.5 27B and 35B-A3B upload in a
4-bit or 8-bit format vLLM can serve, plus the official quantized Qwen3 and Qwen2.5 baselines.
Calls used: `list_models` by author `Qwen` and by name search, sorted by downloads; `model_info`
with file metadata; safetensors headers.

Sizes below are safetensors totals at the pinned commit. Whether a build fits was judged on its
language-model share: the safetensors headers without the vision tower and the MTP head, which
`--language-model-only` and the non-speculative launch never load. This matched vLLM's loaded
weights within 0.45 GiB for every 27B and 35B-A3B build tried.

Order of preference: official checkpoints first, then reputable quantizers (RedHatAI, Intel,
QuantTrio, cyankiwi), then high-download community builds. Fine-tunes and other runtimes'
formats are out.

Every chat template was hashed against its base model's. All but two are byte-identical. The two
that differ (`groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit`, `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`) were
rendered with jinja2 against the official Qwen3.6 template for system + user, user only, and
system + user + assistant prefill, thinking on and off: identical output in all 12 cases.

**Download budget.** All checkpoints share `~/.cache/huggingface` with the container. The plan
capped new downloads at about 250 GB. Checkpoints were downloaded at pinned commits in priority
order: Qwen3.8-27B builds, Qwen3.6-27B builds, the official 35B-A3B, the baselines, and then
Qwen3.6-35B-A3B once the official 35B-A3B had shown that the layout fits.

**Per candidate.** A throwaway runner did the steps below. Per-preset metadata, server logs and
smoke-run outputs are in the gitignored `runs/candidates/<preset>/`.

1. Stop any server and run `scripts/serve_vllm.sh --preset NAME`. Startup = seconds to a
   healthy `/health`. Each run is a new container, so torch.compile and CUDA graphs are rebuilt
   and the prefix cache is empty.
2. Read weights, KV cache and kernels from the server log.
3. Render `/tokenize` for system + user and for user only, to check that thinking is off and
   nothing else is injected.
4. The emulator (default `auto` strategy, diagnostics on) answers the first 20 LEXam-en bank
   items (seed 0, IDK option, evaluate-idk order; the smoke run's items) with 16 questions and
   16 HTTP requests in flight. Items/s is 20 / wall time of this pass, with a cold prefix cache.
5. The same 20 items again with echo (S4, `mode="sum"`, 16 in flight), to exercise the
   prompt-logprob spike. Only presets measured after this pass was added ran it (finding 5).
6. `scripts/smoke.py` runs the same 20 items one at a time against Jev (answers from the
   response cache, $0) and the emulator. "20-item accuracy" is this smoke run's (sequential)
   emulator accuracy.

**Resolution.** With 20 items, one standard error of an accuracy near 0.4 is 0.11. Restarting
the same server moved `qwen3.8-27b-awq` from 0.40 (the earlier smoke run) to 0.35. The 20-item
numbers only show that a checkpoint answers sensibly (well above the 0.2 chance level on A-D +
IDK). They do not rank candidates.

## Considered, not tried

| Checkpoint | Safetensors GiB | Why not |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B` (bf16) | 51.75 | Does not fit 24 GiB |
| `Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.6-27B-FP8` | 28.75 | Does not fit |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` (official) | 28.16 | Does not fit: only the MLPs are 4-bit; attention and DeltaNet stay bf16 |
| `cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4`, `cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4` | 26.86, 26.37 | Does not fit (more layers kept in bf16) |
| `goldhub/Qwen3.8-27B-BF16-INT4-W4A16-G32-AutoRound` | 26.37 | Does not fit |
| `QuantTrio/Qwen3.6-27B-AWQ-6Bit` | 25.79 | Does not fit |
| `Qwen/Qwen3.5-35B-A3B-FP8`, `Qwen/Qwen3.6-35B-A3B-FP8` | 34.89 | Does not fit |
| `Qwen/Qwen-AgentWorld-35B-A3B` (bf16) | 64.56 | Does not fit |
| `cyankiwi/Qwen-AgentWorld-35B-A3B-AWQ-INT4` (the only 4-bit AgentWorld build), `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | 21.92, 23.25, 22.78 (language model 21.9, 21.9, 21.4) | Does not fit at 8,192 tokens [INFERENCE from the official 35B-A3B's profile: 0.26 GiB non-torch memory, 1.13 GiB activation peak, 0.10 GiB CUDA graphs, so even 0.97 of the GPU leaves at most 21.4 GiB for weights plus KV cache]; download budget |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ`, `QuantTrio/Qwen3.5-35B-A3B-AWQ` | 23.71 (language model 21.3) | Download budget; by the same arithmetic no room is left for the KV cache (QuantTrio's 27B build loaded 0.35 GiB above its language-model share), so it would need a smaller `--max-num-batched-tokens` or a shorter context [INFERENCE] |
| `Qwen/Qwen3.8-Flash-Next` (`-FP8`) | 335.28 (172.78) | Does not fit; license "other" |
| `Qwen/Qwen3-14B` (bf16) | 27.51 | Does not fit |
| `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`, `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16` | 18.17, 18.53 | Not servable by vLLM 0.30.0: its Quark dispatch (`quark.py`, `_get_scheme_cls_from_config`) has FP8, INT8, MXFP4/OCP-MX and NVFP4 schemes but no INT4 weight-only one and raises "No quark compatible scheme was found" (read from the pinned image's source, not run) |
| `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` | 19.36 | No license declared |
| `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | 20.81 | bitsandbytes: no Marlin path in vLLM, slow |
| NVFP4 builds (`unsloth/Qwen3.8-27B-NVFP4`, `RadixArk/Qwen3.8-27B-NVFP4`, `Inferact/Qwen3.8-27B-NVFP4`, `nvidia/Qwen3.6-27B-NVFP4`) | 20.4-24.6 | Blackwell format: on SM 8.6 vLLM would run the FP4 weights through Marlin weight-only (supported from SM 7.5); larger than the INT4 builds; download budget |
| `philbert440/Qwen3.8-27B-W4A16-AWQ` | 18.21 | Download budget; the RedHatAI build has the same layout (compressed-tensors, group 128) from a reputable quantizer; this one is calibrated on thinking-mode traces while the presets run thinking off |
| `avyukth/Qwen3.8-27B-AWQ-INT4`, `pearsonkyle/Qwen3.8-27B-GPTQ-W4A16`, `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and other single-uploader INT4 builds | 17.3-18.2 | Download budget; schemes already covered by tried builds |
| AutoRound builds: `Intel/Qwen3.6-27B-int4-AutoRound`, `Frozenlock/Qwen3.8-27B-int4-AutoRound`, `dbirks/Qwen3.8-27B-W4A16-AutoRound`, `Lorbus/Qwen3.6-27B-int4-AutoRound` | 17.7-18.1 | Download budget (finding 8). vLLM 0.30.0 reads `auto-round` checkpoints through its INC config (`auto_round:auto_gptq` packing), not run here |
| `palmfuture/Qwen3.6-27B-GPTQ-Int4`, `btbtyler09/Qwen3.6-27B-GPTQ-4bit` | 19.54, 20.40 | Download budget; the same quantizers' Qwen3.8 builds are tried |
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (official) | 15.77 | Superseded by Qwen3-30B-A3B-Instruct-2507, the non-thinking update of the same model |
| `cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`, `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` | 16.85, 15.55 | The RedHatAI build of the same model is tried |
| `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` | 18.02 | The official AWQ build of the same model is tried |
| GGUF, MLX, OpenVINO and Quark-MXFP4 uploads | | Other runtimes or formats |
| Uncensored, abliterated, distilled or merged fine-tunes (Heretic, OBLITERATED, Fable, Swift, ...) | | Not the Qwen model |

## Small models for the speed/quality study

Sep 25, 2026. The presets above are 9B-35B Qwen models. For the speed/quality trade-off
([reports/speed_quality/](../../reports/speed_quality/README.md)), the same pinned server was
given small instruct models (2B-12B) from Qwen and from the families with the strongest small
models.

Discovery used the Hugging Face API: `list_models` per author, sorted by creation date (Qwen,
google, microsoft, meta-llama, HuggingFaceTB, mistralai, ibm-granite, LiquidAI, nvidia, allenai
and others); `model_info` with file sizes and gating; `config.json` for the architecture. Every
architecture below is in the pinned image's model registry (SmolLM3 through the Transformers
backend). Revisions are pinned in the presets. Per-preset logs are in the gitignored
`runs/select/sweep/` (`preflight.<preset>.log`, `render.<preset>.log`, `<preset>.screen.*.log`).

**Render check.** For one MMLU-Pro, BoolQ, SST-5 and banking77 screen item each: a live
`/tokenize` of the `state_first` prompt, plus the unconstrained top-10 first tokens after the
prefill. On every kept preset the prompt is one user turn, then the model's generation prompt,
then `Answer:`, and the first token is a label (`" F"`, `" Yes"`/`" No"`, or the space before an
SST-5 digit). Letters, `Yes`/`No` and their space-prefixed forms are single tokens on every
tokenizer. Digits after a space split into `" "` + digit on Phi-4, Gemma 4, SmolLM3 and Llama
3.2; the first-token reader handles this by reading after the gap, as for Qwen3 digits.

This check first ran with the old rendering, which sent `Answer:` as a continued assistant
message. That missed one difference, fixed on 2026-09-27 (finding 5). With thinking off, the
Gemma 4 12B template (and 26B-A4B's, [below](#gemma-4-26b-a4b)) opens the model turn with an
empty thought channel, `<|channel>thought\n<channel|>`, but only on the generation prompt. The
old rendering gave `<|turn>model\nAnswer:` instead of
`<|turn>model\n<|channel>thought\n<channel|>Answer:`. Every other preset here, including Gemma
4 E4B (its pinned template has no empty channel), gives identical token ids either way.

| Preset | Checkpoint @ revision | License | Params | Weights GiB (vLLM) | KV cache (0.90) | Startup s | Template notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.5-4b-bf16` | `Qwen/Qwen3.5-4B` @ `851bf6e806` | Apache-2.0 | 4B (hybrid) | 7.99 | 11.8 GiB, 272,570 tokens | 104 | Thinking off (`enable_thinking=false`): empty think block before `Answer:`; nothing injected |
| `qwen3.5-2b-bf16` | `Qwen/Qwen3.5-2B` @ `15852e8c16` | Apache-2.0 | 2B (hybrid) | 3.63 | 16.4 GiB, 980,061 tokens | 99 | As Qwen3.5-4B |
| `qwen3-4b-2507-bf16` | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17` | Apache-2.0 | 4B | 7.64 | 11.9 GiB, 86,736 tokens | 65 | No thinking mode; nothing injected |
| `qwen3-4b-bf16` | `Qwen/Qwen3-4B` @ `1cfa9a7208` (already cached) | Apache-2.0 | 4B | 7.56 | 12.0 GiB, 87,360 tokens | 70 | Thinking off: empty think block; nothing injected |
| `phi-4-mini-bf16` | `microsoft/Phi-4-mini-instruct` @ `cfbefacb99` | MIT | 3.8B | 7.17 | 11.6 GiB, 94,567 tokens | 61 | No thinking mode; no system turn injected (`<|user|>…<|end|><|assistant|>Answer:`) |
| `gemma-4-e4b-bf16` | `google/gemma-4-E4B-it` @ `ee0ef60236` | Apache-2.0 | 4B effective (8B with per-layer embeddings) | 14.15 | 6.8 GiB, 249,556 tokens | 68 | Thinks only when asked (`enable_thinking` defaults to false); no system turn; needs `--enforce-eager` (below) |
| `gemma-4-12b-int4` | `google/gemma-4-12B-it-qat-w4a16-ct` @ `1d2c2d7f24` | Apache-2.0 | 12B, Google QAT INT4 (compressed-tensors W4A16, Marlin; Triton attention) | 8.28 | 10.7 GiB, 82,407 tokens | 126 | As E4B, but thinking off opens the model turn with an empty thought channel (finding 5); the bf16 checkpoint (22.3 GiB) leaves no KV cache |
| `smollm3-3b-bf16` | `HuggingFaceTB/SmolLM3-3B` @ `a07cc9a04f` | Apache-2.0 | 3B | 5.76 | 14.7 GiB, 214,576 tokens | 88 | Always injects a system turn (knowledge cutoff, today's date, "You are a helpful AI assistant named SmolLM, trained by Hugging Face."); thinking off writes `/no_think` and an empty think block |
| `llama-3.2-3b-bf16` | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f76` | Llama 3.2 Community (gated; the token has access) | 3.2B | 6.04 | 13.5 GiB, 126,624 tokens | 82 | Always injects a system header with "Cutting Knowledge Date" and today's date |

Findings:

1. **Every kept model loads and answers sensibly** (screen results are in the report). Two
   templates put the day's date into the prompt (SmolLM3, Llama 3.2), so their prompts change
   from day to day. The injected text is a generic persona and date, not an instruction that
   changes the task, so it was kept as the vendors' default.
2. **Gemma 4 E4B needs eager mode.** With torch.compile, the profiling run tried to allocate a
   second 5.25 GiB block, the size of the per-layer embedding table (262,144 × 10,752 bf16), and
   ran out of memory. With `--enforce-eager` it loads and serves.
3. **Ministral 3 cannot be served by the pinned image.**
   `mistralai/Ministral-3-3B-Instruct-2512-BF16` (@ `b6d637bef2`, downloaded in Hugging Face
   format) fails at model inspection. vLLM 0.30.0's `pixtral.py` imports
   `PixtralRotaryEmbedding`, which the image's transformers 5.17.0 renamed to
   `PixtralVisionRotaryEmbedding`, so every Mistral3/Pixtral checkpoint fails. The preset
   `ministral-3-3b-bf16` is kept for a future image; it was not screened.
4. **SmolLM3-3B and Llama 3.2 3B were screened last.** The sweep was paused before them to move
   the Hugging Face cache. Both are in the report now, near the bottom (9-benchmark screen
   macro 0.5034 and 0.4941, one call per question).
5. **Gemma 4 12B and 26B-A4B were first run without their empty thought channel.** The adapter
   now renders every prompt with the generation prompt through `/tokenize`, appends `Answer:` as
   token ids and scores on `/v1/completions`
   ([design.md](../implementation/design.md), "Assistant prefill"). A second bug was found at
   the same time: vLLM's completions top-k is a map keyed by text, and Gemma's vocabulary has
   byte tokens that decode to the same text as normal tokens (`<0x41>` is `"A"`), so the map
   could keep the wrong one. Logprobs are now keyed by token id. A live A/B on 50 items of each
   of the 9 benchmarks: 26B-A4B macro accuracy 0.693 → 0.689, NLL 1.548 → 1.447; 12B accuracy
   0.669 → 0.682, NLL 1.289 → 1.439, and the 12B's probability outside the labels disappears.
   Both models' runs were redone with the fix and one-call scoring: the 26B-A4B's `holdout`
   macro went from 0.714 to 0.735 and the 12B's screen macro from 0.672 to 0.708 over the 9
   benchmarks ([selection.md](selection.md#what-one-call-scoring-changed)).

Two more Qwen3.6-35B-A3B builds were downloaded for the fast-MoE tuning (not run yet). Sizes are
from the safetensors headers; "language model" means without the vision tower and MTP head:

| Checkpoint @ revision | Total GiB | Language model GiB | 4-bit | 16-bit |
| --- | --- | --- | --- | --- |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` @ `119886a107` (AutoAWQ GEMM, g128, zero point; declares fp16) | 23.70 | 21.30 | routed experts of layers 1-39 | layer 0's routed experts, attention, DeltaNet, shared expert, router, embeddings, `lm_head` |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` @ `00fcea2d3b` (AWQ GEMM, g32, zero point; bf16) | 23.25 | about 21.9 (candidates table above) | | |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` @ `00a6698351` (the existing preset, for reference) | 22.73 | 20.32 | all routed experts | attention, DeltaNet, shared expert, router, embeddings, `lm_head` |

### Gemma 4 26B-A4B

Sep 26, 2026. Google's mixture-of-experts (MoE) Gemma: 128 experts, 8 routed plus 1 shared dense
MLP per token, 3.8B of 25.2B parameters active, like the 3B-active Qwen MoE presets. The bf16
checkpoint (48 GiB) does not fit, and Google publishes no W4A16 build of this model (only GGUF
and a 16-bit QAT checkpoint). The preset serves cyankiwi's INT4 build of Google's
quantization-aware-trained (QAT) checkpoint. It is a finalist in
[selection.md](selection.md#gemma-4-26b-a4b-as-a-finalist).

| Preset | Checkpoint @ revision | License | Params | Weights GiB (vLLM) | KV cache (0.90) | Startup s | Template notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-4-26b-a4b-int4-cyankiwi` | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` @ `18a3c7285c` (from `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`) | Apache-2.0 | MoE, 3.8B of 25.2B active; compressed-tensors W4A16, sym, g32 (Marlin, `MarlinExperts`; Triton attention) | 15.56 | 3.98 GiB, 43,697 tokens | 104 (98-101 in the sweep) | As E4B; an older revision of Google's template |

Findings:

1. **It fits at 0.90 of the GPU with the `gemma-4-12b-int4` flags unchanged.** All weights are
   on the GPU (no offload). `nvidia-smi` showed 20,716 MiB in use. The KV cache holds 5.33
   requests of 8,192 tokens; during the runs at most 55% of it was in use at 16 in flight.
   Startup includes about 35 s of torch.compile. The download (17.19 GB) took 164 s.
2. **What is 4-bit** (safetensors header): attention q/k/v/o and the routed experts. The dense
   MLP, the router, the tied embeddings/`lm_head` and the vision tower stay 16-bit. The config
   declares float16; the preset serves bf16, so vLLM casts.
3. **Triton attention is forced.** vLLM logs "Gemma4 model has heterogeneous head dimensions
   {sliding 256, full 512}. FA4 not available, forcing TRITON_ATTN backend."
4. **The vision tower is skipped.** It is a 27-layer SigLIP-style encoder (about 1 GiB in fp16,
   not quantized); `--language-model-only` does not load it.
5. **Template.** The upload's `chat_template.jinja` is an older revision of Google's Gemma 4
   template. The newer one (the same for Google's 26B-A4B, 26B-A4B QAT and 12B QAT uploads)
   differs only in tool calling, thinking preservation and multimodal item types, not in the
   chat or prefill path. Like the 12B, it opens the model turn with an empty
   `<|channel>thought\n<channel|>` block, but only on the generation prompt. The adapter renders
   that prompt before `Answer:` (finding 5 [above](#small-models-for-the-speedquality-study));
   runs before 2026-09-27 rendered `Answer:` as a continued assistant message, without the block,
   and were redone. No thinking text is added. One quirk: a system turn ends with an extra space
   before `<turn|>` (Google's current template has the same line). `state_first` sends no
   system message, so it is not exercised.
6. **Labels** as in the rest of the family: only `" 0"`..`" 9"` are two tokens (space + digit).

### Small models considered, not tried

| Checkpoint | Safetensors GiB | Why not |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | 1.63 | Below the 2B floor of the study |
| `Qwen/Qwen3-VL-4B-Instruct` | 8.27 | Vision-language variant; its text sibling `Qwen3-4B-Instruct-2507` is tried |
| `google/gemma-4-E2B-it` (bf16) | 9.54 | E4B is the Gemma 4B-class point |
| `google/gemma-4-E4B-it-qat-w4a16-ct` | 10.72 | The bf16 checkpoint fits and is tried; the INT4 build is barely smaller (per-layer embeddings stay 16-bit) |
| `google/gemma-4-12B-it` (bf16) | 22.28 | Leaves no KV cache on 24 GiB; Google's QAT INT4 build is tried |
| `google/gemma-4-26B-A4B-it` (bf16) | 48.07 | Does not fit; no W4A16 build from Google (only GGUF and a bf16 QAT checkpoint). cyankiwi's INT4 build of the QAT checkpoint is tried ([above](#gemma-4-26b-a4b)) |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 21.67 | Not a small model |
| `google/gemma-3-4b-it` | 8.01 | Superseded by Gemma 4; gated |
| `microsoft/Phi-4-mini-reasoning`, `microsoft/Phi-4-mini-flash-reasoning` | 7.2 | Reasoning models; the emulator reads one token after `Answer:` |
| `microsoft/phi-4` (14B, bf16) | 27.31 | Does not fit |
| `mistralai/Ministral-3-3B-Instruct-2512` (FP8), `Ministral-3-8B-Instruct-2512` (FP8, `-BF16`) | 4.35, 9.70, 16.61 | Same Mistral3 architecture as the 3B BF16 build that cannot load (finding 3) |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 7.40 | NVIDIA Open Model License; a hybrid Mamba reasoning model; lower priority than the families tried |
| `ibm-granite/granite-4.2-3b`, `granite-4.2-8b` | 6.82, 16.38 | Lower priority than the families tried (newest IBM models, Aug 2026) |
| `LiquidAI/LFM2.5-2.6B` | 5.02 | LFM Open License v1.0 (not Apache/MIT) |
| `swiss-ai/Apertus-v1.1-4B-Instruct`, `allenai/Olmo-3-7B-Instruct` | 7.13, 13.59 | Lower priority than the families tried |
| `CohereLabs/tiny-aya-global` | 6.24 | Gated: the token has no access (`GatedRepoError` on `config.json`); CC-BY-NC-4.0 |
| `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` | | A distilled fine-tune of Qwen3.5-9B |
