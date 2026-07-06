# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch LLM research harness that pits novel sub-quadratic architectures
against a reference **GPT-2 Small** under an *equal-budget* protocol (same wall-clock
seconds or same step count per model), then compares validation loss/perplexity. All
models train on wikitext-103. Comments and prints are in French.

Current architectures under research: **Memora** (GLA + local attention hybrid) and
**Cortex** (BitNet b1.58 ternary weights + GDN + SWA hybrid — see `models/cortex.md`
for the full staged implementation plan).

## Commands

```bash
source .venv/bin/activate        # uv-managed venv (pyproject.toml); or: uv run <cmd>

# Architecture self-checks (no GPU, no data needed) — run these first when touching models
python -m models.memora          # shape checks + GLA chunked==recurrent equivalence + stability + param budget
python -m models.cortex          # BitLinear shape + backward + ternary-weight assert
python -m models.triton_kernels  # ternary matmul correctness + benchmark (requires CUDA)
python -m training.bitlinear_deploy  # ternary packing round-trip check (CPU-safe)

# Full Memora training (save/resume, WSD schedule, wandb)
python training/train.py --steps 10000                          # ~655M tokens, wandb on
python training/train.py --steps 20000 --resume                 # reprend depuis checkpoints/latest.pt
python training/train.py --steps 5000 --no-wandb --no-compile   # debug/CPU
python training/train.py --batch-size 4 --grad-accum 8          # batch eff = batch_size × grad_accum × context_len tok

# Comparative training (currently hardcoded: Cortex vs GPT-2, même budget chacun)
python training/compare.py                                       # 300s/model, wandb on
python training/compare.py --max-steps 200                       # steps fixes (override --duration)
python training/compare.py --duration 600 --no-wandb

# Text generation (GPT-2 only wired in CLI)
python training/generate.py --pretrained --prompt "Once upon"   # loads HF gpt2 weights
```

Deps declared in `pyproject.toml`: `torch`, `tiktoken`, `datasets`, `wandb` (+ `triton` for
GPU kernels). First run tokenizes wikitext-103 and caches tensors to `.cache/wikitext103_{split}.pt`;
subsequent runs load from cache. `training/dataset.py` is the shared module for `WikiTextDataset`
and `evaluate`.

## Architecture

All model implementations live in `models/`. The package is `models/` with an empty
`__init__.py`; import as `from models.model_interface import ...`.

**The contract is `models/model_interface.LanguageModel`** (ABC + `nn.Module`). Any model
implements `forward(idx) -> logits` and `from_pretrained(name)`; the base class
provides `loss`, `generate`, and `num_params` (which excludes positional embeddings
so param counts compare fairly across architectures). `BaseModelConfig` defaults are
exactly GPT-2 Small (50257/1024/12/12/768). To add a model, subclass `LanguageModel`,
subclass `BaseModelConfig`, and wire it into `train.py:main` and `compare.py:main`.

**`models/gpt2.py`** — faithful GPT-2 Small: pre-norm, learned pos-embeddings, GELU-tanh,
weight tying, biases everywhere. `from_pretrained` maps HF param names and transposes
the Conv1D weights to `nn.Linear` layout.

**`models/memora.py`** — hybrid sub-quadratic at ~126M params:
- Most layers are `LocalAttention` (sliding-window causal + GQA + RoPE + QK-Norm).
- Layers in `recurrent_layers` are `GatedLinearAttention` (GLA) — linear attention
  with a fixed-size recurrent state for unbounded context. **Critical invariant:**
  the chunked training path (`_chunked`) and the recurrent reference (`_recurrent`)
  must stay numerically equivalent — asserted in `models/memora.py.__main__`. GLA
  stability comes from never materializing absolute `e^{-Gc}` (only bounded
  differences `Gc_i - Gc_j ≤ 0`); preserve that if you edit `_chunked`.
- SwiGLU MLP, RMSNorm, RoPE (no learned pos-embeddings), z-loss, no biases on
  content projections. GLA layers deliberately skip RoPE (position lives in the state).

**`models/cortex.py`** — the next research architecture (Sparse-Cortex), brain-inspired:
- **Ternary weights (BitNet b1.58):** all internal `nn.Linear` replaced by `BitLinear`
  ({−1, 0, +1}, straight-through estimator on a full-precision latent weight).
  Embeddings, RMSNorm, and LM head stay bf16 — never ternarize those.
- **Hybrid token mixer:** ~3 GatedDeltaNet (GDN) layers per 1 SlidingWindowAttention
  (SWA) layer. GDN = fixed-size recurrent state (linear cost, bounded VRAM); SWA =
  exact local recall the compressed state loses. Do not remove SWA layers.
- **ReLU² FFN:** squared ReLU induces ~90–95% activation sparsity emergently.
- **Mixture-of-Recursions (phase 3+):** single shared block rebouclé R times per token,
  routed by difficulty. Requires a router homeostasis loss to prevent depth collapse.
- See `models/cortex.md` for the full staged implementation plan (one lever at a time,
  measured against GPT-2 baseline before adding the next).

**`models/memora_gla.py`** — GLA-dominant Memora variant (~124–127M params): 10 GLA layers,
2 global attention, 2 local attention. Uses `flex_attention` compiled once at import; the
Triton GLA kernel is optional (falls back to pure-torch `_chunked`). Not currently wired
into `compare.py`; run directly via `train.py` with a config change.

**`models/triton_kernels.py`** — Triton ternary matmul for `BitLinearInference`: unpacks
2-bit weights inline (no dense weight allocation), int8-quantizes activations, accumulates
in fp32, then dequantizes. Block sizes tuned for RTX 5070 Ti (K ∈ {768, 3072}). Called by
`training/bitlinear_deploy.BitLinearInference.forward` on CUDA; CPU/MPS falls back silently
to dense unpack + `F.linear`.

**`training/bitlinear_deploy.py`** — ternary deployment path: converts a trained Cortex
(fp32 latent weights) to inference mode by packing `BitLinear` → `BitLinearInference`
(2-bit packed weights, no fp32 master). `convert_to_inference(model)` does the in-place
swap; `report_memory(model)` prints resident bytes by category. ~16× weight compression vs
fp32, but PyTorch matmul still depacks transiently — the Triton kernel avoids that.

**`training/train.py`** — full Memora training: WSD schedule (warmup→stable→cosine decay),
atomic checkpointing (`checkpoints/latest.pt` + `best.pt`), AdamW fused kernel,
grad norm logging. Resume with `--resume`. Imports `WikiTextDataset`/`evaluate` from `training/dataset.py`.

**Equal-budget comparison (`training/compare.py`)** — currently hardcoded to **Cortex vs GPT-2**.
`train_model` runs each for the *same* `duration_seconds` OR `max_steps` (exactly one, asserted),
AdamW + manual cosine schedule with warmup. wandb runs grouped by timestamp so curves overlay.
Results dumped to `results/{timestamp}_{model}.json`.

## Gotchas

- **vocab_size mismatch.** `MemoraConfig` defaults to 49152, but the dataset uses
  tiktoken `gpt2` (50257 tokens). `train.py` overrides Memora to 50257 — token ids
  ≥ 49152 would otherwise overflow `tok_embd` and trigger a CUDA device-side assert.
  Keep both models on the same vocab for a fair comparison.
- `Memora.from_pretrained` raises `NotImplementedError` by design (novel arch, no HF
  checkpoint) — train from scratch.
- `training/generate.py` is hardcoded to GPT-2; Memora/Cortex aren't wired into its CLI.
- Editing GLA? Run `python models/memora.py` — the equivalence assert is the regression guard.
- **Cortex BitLinear training rules:** natif ternaire from scratch (never PTQ), LR ~2×
  bf16 baseline, activations in int8 (never ternary), no biases in BitLinear layers.
- **MoE ≠ VRAM savings** at small scale — all experts must reside in memory. Reserve MoE
  for scale; ternary + MoR is the VRAM play at 124M.
