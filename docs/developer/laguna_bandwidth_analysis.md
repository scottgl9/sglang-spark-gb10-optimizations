# Laguna-S-2.1-NVFP4 Bandwidth Analysis on GB10

Analysis of `poolside/Laguna-S-2.1-NVFP4` (117.6B total / 8.5B active MoE,
served via the `laguna-s-2.1` preset in `sglang.sh`) against GB10's ~273 GB/s
unified-memory bandwidth ceiling. See also
[gb10_optimizations.md](gb10_optimizations.md) for the general SM121 fix
catalog this analysis builds on.

## Checkpoint structure

- 48 layers, 12 global-attention : 36 sliding-window (512-token) in a fixed
  3:1 ratio. Sliding layers use **72** attention heads, global layers use
  **48** — an inversion of the usual "cut heads on the cheap layers" pattern.
- 256 routed experts (10 active/token) + 1 shared expert. `moe_intermediate_size`
  and `shared_expert_intermediate_size` are both 1024.
- Layer 0 is the only dense (non-MoE) FFN layer (`mlp_only_layers: [0]`,
  `intermediate_size: 12288`).
- Quantization: `compressed-tensors`, format `nvfp4-pack-quantized`, on
  **routed-expert weights only** (`gate_proj`/`up_proj`/`down_proj` inside
  `mlp.experts.*`). Everything else — attention (`q/k/v/o/g_proj`, `q_norm`/
  `k_norm`), the shared expert, layer-0's dense FFN, `embed_tokens`, and
  `lm_head` — stays BF16 (checkpoint's own `quantization_config.ignore` list).
  KV cache is FP8.
- Checkpoint is 71.9 GB on disk (`model.safetensors.index.json` `total_size`).
  DFlash draft model (`poolside/Laguna-S-2.1-DFlash-NVFP4`) is a separate
  2.1 GB, 6-layer, all-sliding-window BF16 model that borrows the target's
  `lm_head` rather than having its own.

## Bandwidth breakdown (bytes read per decode step, computed from actual tensor shapes)

| Component | Bytes/token | Scales with context? |
|---|---|---|
| MoE (10 routed + 1 shared expert, per sparse layer × 47) | 3.38 GB | No |
| Attention weights (all 48 layers) | 5.61 GB | No |
| Layer-0 dense FFN | 0.23 GB | No |
| KV cache reads, 12 global-attention layers | 0.10 – 6.44 GB | **Yes** (linear in context) |
| KV cache reads, 36 sliding-window layers | ~0.04 GB (flat) | No (capped at 512-token window) |

| Context | Total bytes/token | Bandwidth ceiling @ 273 GB/s |
|---|---|---|
| 4K | 9.35 GB | 29.2 tok/s |
| 32K | 10.06 GB | 27.1 tok/s |
| 131K | 12.47 GB | 21.9 tok/s |
| 256K | 15.70 GB | 17.4 tok/s |

This tracks the vendor's own reported GB10 number: "decode sits at 13-14 tok/s
... which is the memory-bandwidth ceiling for this model. Speculation
(DFlash) is how you get past it." (poolside README). The gap between the
computed ceiling and the measured number is expected — dequant compute,
softmax, and kernel-launch overhead aren't in this bytes-only model.

### Two non-obvious findings

1. **Attention weights, not MoE, are the largest static bandwidth cost**
   (5.61 GB vs 3.38 GB/token). Driven by the 36 sliding-window layers' 72
   heads — `q_proj`+`o_proj` alone are ~85-90% of attention's bytes, split
   roughly 50/50 between the two.
2. **The 12 global-attention layers are what makes long context expensive**,
   precisely because there are only 12 of them: they read the *entire*
   context every decode step, while the 36 sliding layers stay flat forever.
   Going 4K → 256K context roughly **doubles** total per-token bandwidth,
   entirely from those 12 layers.

### Memory budget cross-check

Per-sequence KV cache at 256K context ≈ 6.5 GB (dominated by the 12 global
layers; sliding layers contribute a fixed ~38 MB regardless of context
length, thanks to the 512-token window cap). This reproduces poolside's
stated "830-870K KV tokens" aggregate almost exactly (≈3.3 concurrent
256K-context sequences × 262144 ≈ 865K). At `--mem-fraction-static 0.85`
(matches poolside's `--gpu-memory-utilization 0.85`), the preset's
`--max-running-requests 4` is right-sized for this hardware, not overly
conservative — despite an unrelated stale log message that used to claim
`32` (copy-pasted from poolside's vLLM `--max-num-seqs`; fixed in `sglang.sh`).

## FP8 post-quantization opportunities for the remaining BF16 layers

Checked every existing FP8 post-quant mechanism in this fork against
Laguna's architecture and model class (`python/sglang/srt/models/laguna.py`):

| Layer | Existing infra? | Bandwidth payoff | Verdict |
|---|---|---|---|
| `lm_head` | Yes — `SGLANG_QUANTIZE_LM_HEAD_FP8`, generic via `CompressedTensorsConfig` | 618 MB → 309 MB/token | **Enabled** (see below) |
| `embed_tokens` | No — `SGLANG_QUANTIZE_EMBED_FP8` is hardcoded to `minimax_m2.py` only | ~0 (embedding lookup reads 1 row/token, not a full matmul) | Not worth pursuing regardless of infra |
| MTP FP8 (`eagle_worker_v2.py`) | No — tied to EAGLE/NEXTN draft heads that share target embed/lm_head | N/A | Doesn't apply; DFlash's draft has its own weights |
| GDN FP8 (`fp8_post_quant.py`) | No — targets Gated-DeltaNet linear-attention layers | N/A | Doesn't apply; Laguna has no GDN/mamba layers |
| Attention `v_proj`/`o_proj` | No — would need new code | ~2.5-2.8 GB/token → ~1.25-1.4 GB/token | Plausible, not implemented (see risk assessment) |
| Attention `q_proj`/`k_proj` | No — would need new code | Largest single lever (~2.4 GB/token) | **Not recommended** without empirical accuracy validation |
| `g_proj` (per-head gate) | No | Negligible (<0.4% of attention bytes) | Not worth the risk regardless |
| Shared expert | No — would need new code | ~887 MB total (47 layers) | Vendor excluded it from their own NVFP4 pass; plausible but unproven here |
| Layer-0 dense FFN | No — would need new code | 226 MB/token | First-layer error sensitivity (errors propagate through 47 more layers); not recommended |

### `lm_head` FP8 — enabled in the `laguna-s-2.1` preset

`SGLANG_QUANTIZE_LM_HEAD_FP8=1` is generic (wired into `CompressedTensorsConfig`,
applies to any compressed-tensors model, not model-specific). Enabling it
uncovered a real bug in DFlash's draft-sampler-folding fast path — see
**Fix 9** in [gb10_optimizations.md](gb10_optimizations.md) — now fixed, so
this is safe to combine with DFlash. Expected accuracy impact: low but
unverified. `lm_head` is a terminal layer (no error compounding into later
layers, unlike attention), FP8 has real precision, and output already goes
through `temperature=0.7`/`top_p=0.95` sampling that masks small logit
perturbations. But no accuracy/quality eval has actually been run for this
specific change — this fork's FP8 post-quant track record (Fix 6/7 in the
general doc) is latency-only benchmarking, never perplexity or eval-score
deltas. Recommend a quick spot-check (diffed completions on a few prompts,
ideally including agentic/coding tasks) before trusting it in production.

### Attention FP8 — assessed, not implemented

Not a blanket yes/no — risk splits sharply by projection:

- **`q_proj`/`k_proj` (risky):** these compute attention scores
  (`QK^T` → softmax). Softmax nonlinearly amplifies small quantization
  errors into shifted attention *routing*, which compounds through every
  remaining layer — unlike `lm_head`'s one-shot terminal error. Laguna's own
  `q_norm`/`k_norm` (QK-norm) is itself a signal that attention here already
  needed extra numerical stabilization.
- **`v_proj`/`o_proj` (safer):** no softmax involvement — `v_proj` is content
  aggregated by an (unquantized) attention pattern, `o_proj` is a plain
  linear re-projection. Same risk class as layers already proven safe in
  this fork (GDN's `out_proj`, Fix 6) — but note that precedent is *linear*
  (GDN/gated-recurrence) attention, not softmax attention, and was measured
  on latency only, never accuracy.
- **`g_proj` (skip):** negligible bytes (<0.4% of attention total), novel/
  untested per-head gating mechanism specific to this model — no established
  precedent either way.

If pursued: implement `v_proj`/`o_proj` FP8 post-quant only (following the
`fp8_post_quant.py` pattern), leave `q_proj`/`k_proj` in BF16, and validate
accuracy empirically before trusting — this fork has no quality-metric data
for any FP8 post-quant work done so far, only latency benchmarks.

## Changes made to `sglang.sh` (`cmd_laguna_s_21`) this session

1. `--preferred-sampling-params '{"temperature":0.7,"top_p":0.95}'` — SGLang's
   equivalent of vLLM's `--override-generation-config`, matches the model
   card's recommended sampling settings.
2. `SGLANG_QUANTIZE_LM_HEAD_FP8=1` — enabled now that the DFlash interaction
   bug (Fix 9) is fixed.
3. Fixed a stale log message claiming `max-running-requests 32` (copy-pasted
   from poolside's vLLM recipe) when the actual flag is `4`.
