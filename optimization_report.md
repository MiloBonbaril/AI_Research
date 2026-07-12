# Optimization report — DeepSeek(mini) training speed & VRAM

**Scope:** `models/deepseek.py` + `training/compare.py`.
**Symptom:** DeepSeek ≈ 15 000 tok/s @ ~90 % VRAM vs GPT-2 ≈ 65 000 tok/s @ ~50 % VRAM, identical settings (`--max-steps 300 --batch-size 3 --log-interval 20 --context-len 1024`, grad-accum 16 by default, autocast bf16, RTX 5070 Ti 16 GB, torch 2.12.1+cu130).

Evidence labels: **[VERIFIED]** = reproduced on this machine during this analysis. **[ANALYSIS]** = derived from documented PyTorch behavior + consistency with your observed numbers. **[OBSERVED]** = seen on this machine but not necessarily true during your runs.

Classification asked for: **FREE** = no downside, model math unchanged. **TOUCHES MODEL** = changes the computation/regularization (says how). **RISK TO QUALITY** = could plausibly hurt validation loss.

---

## TL;DR — ranked by expected impact

| # | Fix | File | Expected gain | Classification |
|---|-----|------|---------------|----------------|
| 1 | Window-attention branch falls back to the *math* SDPA backend → use bool mask + expanded KV heads (or flex_attention) | `deepseek.py` | Largest single win: speed + ~2–3 GB VRAM | **FREE** (mathematically identical) |
| 2 | `torch.compile` is currently a **no-op** for every model → compile the executed path | `compare.py` | Large for DeepSeek (small-kernel storm), mild for GPT-2 | **FREE** |
| 3 | Hoist/cache masks & indices rebuilt in every layer, every step | `deepseek.py` | Small–moderate | **FREE** |
| 4 | Remove per-micro-batch `loss.item()` GPU sync (16 syncs/step) | `compare.py` | Small–moderate | **FREE** |
| 5 | `fused=True` AdamW + `non_blocking=True` H2D copies | `compare.py` | Small | **FREE** |
| 6 | After VRAM is freed: raise `--batch-size`, lower `--grad-accum` (same effective batch) | usage | Moderate throughput | **FREE** (same expected gradient) |
| 7 | Fuse the 5–6 input projections per attention layer into one GEMM | `deepseek.py` | Moderate in eager, small once compiled | **FREE**\* (different init RNG draw) |
| 8 | Sinkhorn iterations 20 → 5 | `deepseek.py` | Moderate | **TOUCHES MODEL** — likely benign, verify |
| 9 | flex_attention sliding window (true sparsity, codebase precedent) | `deepseek.py` | Further speed beyond #1 | **TOUCHES MODEL** (drops attention-dropout) |
| 10 | Chunked cross-entropy (shared lever, both models) | interface | ~1 GB VRAM each | **FREE**, but more code |

Items 1+2 alone should close most of the gap. DeepSeek will **not** reach GPT-2's 65 k tok/s even fully optimized — it does inherently more work per token (two attention branches per layer, 4-stream mHC residual, Sinkhorn). A realistic target after 1–7 is roughly **35–50 k tok/s** and VRAM well under GPT-2's headroom-adjusted level; measure, don't trust my estimate.

---

## 1. The window branch runs on the math SDPA backend — the main VRAM & speed killer

**Where:** `models/deepseek.py:307` (`CompressedAttention.forward`, window branch).

```python
y_win = F.scaled_dot_product_attention(
    q_win, k_w, v_w, attn_mask=win_mask, enable_gqa=True, ...)
```

**Why it's slow [ANALYSIS]:**
- A non-None `attn_mask` **disqualifies FlashAttention** (documented, stable across torch versions). GPT-2 passes `is_causal=True` with no mask (`models/gpt2.py:82`) → Flash → that's the whole story of 65 k vs 15 k, more than any architectural difference.
- `enable_gqa=True` additionally disqualifies the memory-efficient backend (GQA is supported by flash/cuDNN/math only).
- Net result: the only backend left is **math**, which materializes the full `(B, H, T, T)` attention matrix per layer — ≈ 75 MB bf16 for scores + 75 MB softmax output saved for backward + a `(B,H,T,T)` dropout mask ≈ 38 MB, **per layer**. ×12 layers ≈ **2.3–3 GB** of pure waste, plus the quadratic compute. This is consistent with your observed 90 % vs 50 % VRAM; that consistency is the corroborating evidence (I could not probe the backend live — see "GPU state" note at the end).

**Fix (minimal diff, mathematically identical → FREE):** make the mask boolean and expand KV heads instead of `enable_gqa` — this makes the call eligible for the memory-efficient backend, which supports arbitrary masks *and* dropout, never materializes `(B,H,T,T)`, and is within ~10–20 % of Flash:

```python
# masque booléen (pas de -inf flottant) → backend memory-efficient éligible
allowed = (i >= j) & (i - j < self.sliding_window)          # (T,T) bool
g = H // self.n_kv_win
y_win = F.scaled_dot_product_attention(
    q_win,
    k_w.repeat_interleave(g, dim=1),
    v_w.repeat_interleave(g, dim=1),
    attn_mask=allowed,
    dropout_p=self.dropout if self.training else 0.0,
)
```

The `repeat_interleave` costs a small copy (3→12 heads of `(B,·,T,64)` ≈ 4.5 MB) — negligible against what it unlocks. Attention output is bit-for-bit the same *function*; only kernel-level float reordering differs (same as any backend switch).

**Verify after the change** (one-off, GPU free):

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):   # lève une erreur si non éligible
    model.loss(x, y)
```

## 2. `torch.compile` currently compiles nothing — for any model **[VERIFIED]**

**Where:** `training/compare.py:115-117` + `training/compare.py:174`.

`train_model` does `model = torch.compile(model)` but then only ever calls `model.loss(x, y)`. `torch.compile(module)` returns an `OptimizedModule` whose **`forward` is compiled but whose other attributes delegate to the original module** — so `model.loss` is the original bound method, and the `self.forward(idx)` inside it (`models/model_interface.py:86`) runs the **uncompiled** forward.

Reproduced on this machine: calling `.loss()` on a compiled module traces **0 dynamo frames**. So your benchmark numbers for *both* models are eager-mode numbers, and `--no-compile` changes nothing.

This hurts DeepSeek far more than GPT-2: DeepSeek's forward is a storm of tiny ops (40 sequential `logsumexp` per HyperConnections × 24 modules ≈ **960 micro-kernels/forward** for Sinkhorn alone, plus dozens of small einsums), which is exactly what Inductor fuses well. GPT-2 is a few big GEMMs + Flash; eager barely penalizes it.

**Fix (FREE):**

```python
if use_compile and device == "cuda":
    model.loss = torch.compile(model.loss)   # compile le chemin réellement exécuté
```

(`evaluate()` calls `model.loss` too, so eval benefits for free. The existing docstring note about the caller keeping the trained parameters still holds — compile shares the `Parameter` tensors.)

**Caveats, not downsides:**
- First-step compile latency (tens of seconds to a couple of minutes for DeepSeek's unrolled Sinkhorn). At `--max-steps 300` that pollutes a wall-clock benchmark — either exclude step 0 from tok/s, or bench with more steps. With `--duration` budgets it directly eats budget.
- After changing the model per item 1, run once with `TORCH_LOGS=graph_breaks` to confirm the forward compiles as one graph.

## 3. Masks and indices rebuilt in every layer, every step

**Where:** `models/deepseek.py:303-306` (`i`, `j`, `allowed`, `win_mask`), `models/deepseek.py:320` (`causal_block`), `models/deepseek.py:283` (`end_idx`).

All of these depend only on `T` (fixed at 1024 all run) and the layer's block size (two distinct values). They are recomputed **12× per forward, 4 800× per 300-step run** (× 16 micro-batches). The `(T,T)` mask alone is a 1M-element construction ×12 per micro-batch.

**Fix (FREE):** build them once per forward in `DeepSeek.forward` (or cache per-`T` in a dict / registered buffer) and pass them down alongside `cos, sin`, one set per block size. Zero change to results. Mostly subsumed by item 2 once compiled (Inductor hoists some of it), but it's a trivial edit that also helps the uncompiled path.

## 4. `loss.item()` inside the micro-batch loop — 16 GPU syncs per step

**Where:** `training/compare.py:176` (`accum_loss += loss.item()`).

Each `.item()` blocks the CPU until the GPU drains, preventing the host from enqueueing the next micro-batch's kernels. With `grad_accum=16` that's 16 pipeline stalls per optimizer step, for a number only *read* every `log_interval` steps.

**Fix (FREE):** accumulate on-device, sync once per step:

```python
accum_loss = torch.zeros((), device=device)
...
    accum_loss += loss.detach()
...
accum_loss = accum_loss.item()   # un seul sync, après optimizer.step()
```

(If you want zero syncs off the logging path, only `.item()` when `step % log_interval == 0` — the wandb per-step `train/loss` then needs the same treatment.)

## 5. Small free wins in the training loop

- **`training/compare.py:118`** — `torch.optim.AdamW(..., fused=True)` on CUDA (already done in `train.py` per project docs). One fused kernel instead of a foreach chain over 124 M fp32 params. **FREE**, ~ms-level per step.
- **`training/compare.py:172`** — `x.to(device, non_blocking=True)` (same for `y`). You already pay for `pin_memory=True`; without `non_blocking` the copy is synchronous and the pinning buys nothing. **FREE**, small.
- **`training/compare.py:317`** — `--grad-checkpoint` is parsed and **never used** (dead flag). Either wire it (it's a real VRAM↔speed lever: ~30 % slower for a large activation-memory cut, letting batch size grow) or delete it so nobody trusts it. Right now it silently does nothing.

## 6. Refill the freed VRAM with batch size

Once items 1–2 land, DeepSeek's VRAM should drop by several GB. Raise `--batch-size` and lower `--grad-accum` keeping `batch_size × grad_accum` constant (e.g. 6×8 instead of 3×16): same tokens/step, same expected gradient, fewer/larger kernels → higher tok/s. **FREE** for the comparison protocol (`tokens_seen` axis unchanged); only the dropout/data-order RNG pattern shifts, same as any batch-size choice. Applies to GPT-2's runs too — keep both models on the same effective batch for fairness.

## 7. Fuse per-layer input projections into one GEMM

**Where:** `models/deepseek.py:233-258` — `q_proj`, `k_win`, `v_win`, `c_proj`, `gate` (+ `iq_proj` on CSA layers) are six separate `nn.Linear` over the *same* `x`.

GPT-2 fuses QKV into a single `c_attn` (`models/gpt2.py:60`); DeepSeek launches ~6 skinny GEMMs per attention layer instead of one wide one. Concatenate into one `nn.Linear(n_embd, sum_of_outputs)` and `split` the result.

**Classification: FREE\*** — the function class and per-slice init distribution are identical; only the RNG draw of the initial weights changes (a different seed does more). Lower priority once item 2 lands (compile reduces launch overhead, though it does not merge GEMMs).

## 8. Sinkhorn: 20 iterations is likely 4× more than needed

**Where:** `models/deepseek.py:77` (`sinkhorn_iters: int = 20`), `models/deepseek.py:183-189`.

Each HyperConnections forward runs 20 sequential normalization rounds (40 `logsumexp` + subtracts) on a tiny `(B,T,4,4)` tensor, ×24 modules — a long serial dependency chain that neither eager nor compile can parallelize away, plus ~40 saved intermediates per module for backward. Your own self-check comment (`models/deepseek.py:473-475`) notes the model's logits start near-identity and converge much faster than the randomized test case.

**Fix:** drop to 5 iterations. **Classification: TOUCHES MODEL** — `B` becomes slightly less exactly doubly-stochastic, which is the mHC stability guarantee, so don't do it blind. Cheap verification: during a short run, log `max |row_sum − 1|` and `max |col_sum − 1|` of `Bm` at 5 iters; if it stays ≲1e-2 (the tolerance your own self-check uses) through training, the change is safe. Keep 20 in the self-check itself.

## 9. flex_attention for the window branch (beyond item 1)

Item 1's memory-efficient backend still *computes* all masked-out positions (window 128 of 1024 → ~7/8 of the score matrix is wasted FLOPs). `flex_attention` with a sliding-window `BlockMask` skips them — and the codebase already uses exactly this pattern in `models/memora_gla.py`.

**Classification: TOUCHES MODEL** — `flex_attention` has **no dropout**, and this branch currently applies attention-dropout 0.1 in training (`compare.py` registry passes `dropout=0.1`). Dropping it is a mild regularization change: plausibly neutral at 300–10 000 steps on wikitext-103, but it *is* a training-behavior change and could shift val loss either way. Do it after item 1 is measured, as a separate A/B. (The HC-level output dropout at `models/deepseek.py:203` remains either way.)

## 10. Shared lever: cross-entropy materializes ~1 GB for both models

`LanguageModel.loss` (`models/model_interface.py:86-90`) computes full `(B·T, 50257)` logits; under autocast, `cross_entropy` promotes to fp32 → ≈ 0.6 GB fp32 logits + saved softmax state, per micro-batch, for **both** models. A chunked CE (loop over sequence chunks, or a fused kernel like cut-cross-entropy) reclaims ~1 GB. **FREE** numerically (same loss up to reduction order) but it's real added code in the shared interface and helps both models equally — so it changes nothing about the *comparison*. Do it only if you need the VRAM for batch size.

## Explicitly NOT recommended (would change the research object)

- **Reducing `hc_streams` 4 → 2** or collapsing the mHC residual: that 4× stream memory/traffic *is* the mHC experiment. **RISK TO QUALITY** and invalidates the architecture comparison.
- **Removing one of the two attention branches per layer** (window or compressed): same reason — the dual-branch fusion is the DeepSeek-V4 mechanism under test. This is the main irreducible reason DeepSeek stays slower than GPT-2 per token.
- **Full-bf16 weights** (`model.to(bfloat16)` instead of autocast): saves optimizer/master-weight VRAM but changes optimizer dynamics at 126 M scale. **RISK TO QUALITY**; autocast bf16 (current setup) is the right call.

## Measurement hygiene

- **[OBSERVED]** While writing this report, the GPU was at 100 % with ~15.8/16.3 GB used: a 12.4 GB python run, **plus Mount & Blade II (1.6 GB VRAM) and a browser's GPU process**. If anything similar was running during your 15 k/65 k measurements, both numbers are depressed and the VRAM headroom picture is distorted. Re-benchmark with the desktop quiet.
- Benchmark order for attribution: baseline → item 1 alone → +item 2 → +items 3–5. One change at a time, 300 steps each, log tok/s excluding step 0 (compile warmup) and `torch.cuda.max_memory_allocated()`.
- The live SDPA-backend probe and a per-op profile couldn't run here (GPU full — CUDA context wouldn't even initialize). The two-line `sdpa_kernel` check in item 1 confirms the backend story in seconds once the GPU is free.
