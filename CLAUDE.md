# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch LLM research harness that pits a novel sub-quadratic architecture
(**Memora**) against a reference **GPT-2 Small** under an *equal-budget* protocol
(same wall-clock seconds or same step count per model), then compares validation
loss/perplexity. Both train on wikitext-103. Comments and prints are in French.

## Commands

```bash
source venv/bin/activate

# Architecture self-checks (no GPU, no data needed) — run these first when touching models
python memora.py      # shape checks + GLA chunked==recurrent equivalence + stability + param budget
python gpt2.py        # (no __main__ self-test; import-only)

# Full Memora training (save/resume, WSD schedule, wandb)
python train.py --steps 10000                          # ~655M tokens, wandb on
python train.py --steps 20000 --resume                 # reprend depuis checkpoints/latest.pt
python train.py --steps 5000 --no-wandb --no-compile   # debug/CPU
python train.py --batch-size 4 --grad-accum 8          # batch eff = batch_size × grad_accum × context_len tok

# Comparative training (Memora vs GPT-2, même budget chacun)
python compare.py                                      # 300s/model, wandb on
python compare.py --max-steps 200                      # steps fixes (override --duration)
python compare.py --duration 600 --no-wandb

# Text generation (GPT-2 only wired in CLI)
python generate.py --pretrained --prompt "Once upon"   # loads HF gpt2 weights
```

There is no requirements.txt. Deps (in `venv`): `torch`, `tiktoken`, `datasets`,
`transformers`, `wandb`. First run tokenizes wikitext-103 and caches tensors to
`.cache/wikitext103_{split}.pt`; subsequent runs load from cache.
`dataset.py` is the shared module for `WikiTextDataset` and `evaluate`.

## Architecture

**The contract is `model_interface.LanguageModel`** (ABC + `nn.Module`). Any model
implements `forward(idx) -> logits` and `from_pretrained(name)`; the base class
provides `loss`, `generate`, and `num_params` (which excludes positional embeddings
so param counts compare fairly across architectures). `BaseModelConfig` defaults are
exactly GPT-2 Small (50257/1024/12/12/768). To add a model, subclass `LanguageModel`,
subclass `BaseModelConfig`, and wire it into `train.py:main` (around line 383 — the
`model_a`/`model_b` block).

**`gpt2.py`** — faithful GPT-2 Small: pre-norm, learned pos-embeddings, GELU-tanh,
weight tying, biases everywhere. `from_pretrained` maps HF param names and transposes
the Conv1D weights to `nn.Linear` layout.

**`memora.py`** — the research architecture, hybrid sub-quadratic at ~126M params:
- Most layers are `LocalAttention` (sliding-window causal + GQA + RoPE + QK-Norm).
- Layers in `recurrent_layers` are `GatedLinearAttention` (GLA) — linear attention
  with a fixed-size recurrent state for unbounded context. **Critical invariant:**
  the chunked training path (`_chunked`) and the recurrent reference (`_recurrent`)
  must stay numerically equivalent — this is asserted in `memora.py.__main__`. GLA
  stability comes from never materializing absolute `e^{-Gc}` (only bounded
  differences `Gc_i - Gc_j ≤ 0`); preserve that if you edit `_chunked`.
- SwiGLU MLP, RMSNorm, RoPE (no learned pos-embeddings), z-loss, no biases on
  content projections. GLA layers deliberately skip RoPE (position lives in the state).

**`train.py`** — full Memora training: WSD schedule (warmup→stable→cosine decay),
atomic checkpointing (`checkpoints/latest.pt` + `best.pt`), AdamW fused kernel,
grad norm logging. Resume with `--resume`. Imports `WikiTextDataset`/`evaluate` from `dataset.py`.

**Equal-budget comparison (`compare.py`)** pits Memora vs GPT-2: `train_model` runs each
model for the *same* `duration_seconds` OR `max_steps` (exactly one, asserted),
AdamW + manual cosine schedule with warmup. wandb runs grouped by timestamp so curves
overlay. Results dumped to `results/{timestamp}_{model}.json`. To add a model, wire it
into `compare.py:main` (the `model_a`/`model_b` block).

## Gotchas

- **vocab_size mismatch.** `MemoraConfig` defaults to 49152, but the dataset uses
  tiktoken `gpt2` (50257 tokens). `train.py` overrides Memora to 50257 — token ids
  ≥ 49152 would otherwise overflow `tok_embd` and trigger a CUDA device-side assert.
  Keep both models on the same vocab for a fair comparison.
- `Memora.from_pretrained` raises `NotImplementedError` by design (novel arch, no HF
  checkpoint) — train from scratch.
- `generate.py` is hardcoded to GPT-2; Memora isn't wired into its CLI.
- Editing GLA? Run `python memora.py` — the equivalence assert is the regression guard.
