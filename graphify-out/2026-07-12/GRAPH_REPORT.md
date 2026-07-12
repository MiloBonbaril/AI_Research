# Graph Report - .  (2026-07-11)

## Corpus Check
- 54 files · ~69,762 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 403 nodes · 904 edges · 14 communities (13 shown, 1 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 154 edges (avg confidence: 0.54)
- Token cost: 258,608 input · 0 output

## Community Hubs (Navigation)
- Memora & Oneira Models
- Cortex, Interface & Deploy Pipeline
- Training & Comparison Runs
- Research Concepts & Papers
- Memora-GLA Variant
- DeepSeek-Inspired Architecture
- Effy Architecture
- GPT-2 Baseline
- Triton Ternary Kernel
- Loss Comparison Chart
- Project Root Package

## God Nodes (most connected - your core abstractions)
1. `LanguageModel` - 66 edges
2. `BaseModelConfig` - 47 edges
3. `CLAUDE.md — RSI-Research Project Guide` - 31 edges
4. `Memora` - 27 edges
5. `cortex.md — Sparse-Cortex-LM Architecture Specification` - 27 edges
6. `GPT2` - 24 edges
7. `MemoraConfig` - 20 edges
8. `GPT2Config` - 19 edges
9. `GatedLinearAttention` - 18 edges
10. `TrainResult` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Gated Linear Attention (GLA)` --implements--> `GatedLinearAttention`  [INFERRED]
  CLAUDE.md → models/memora.py
- `Memora (hybrid sub-quadratic architecture)` --implements--> `Memora`  [INFERRED]
  CLAUDE.md → models/memora.py
- `Equal-budget comparison protocol` --rationale_for--> `train_model()`  [EXTRACTED]
  CLAUDE.md → training/compare.py
- `cortex.md — Sparse-Cortex-LM Architecture Specification` --semantically_similar_to--> `WSD schedule (Warmup-Stable-Decay)`  [INFERRED] [semantically similar]
  models/cortex.md → CLAUDE.md
- `FlashAttention-2` --semantically_similar_to--> `Sliding Window Attention (SWA)`  [INFERRED] [semantically similar]
  README.md → models/cortex.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Staged one-lever-at-a-time implementation plan (Cortex)** — concept_bitlinear, concept_gated_deltanet, concept_sliding_window_attention, concept_relu2, concept_mixture_of_recursions, concept_mixture_of_experts [EXTRACTED 1.00]
- **Adaptive-computation / routing mechanism family (MoD, MoR, MoE)** — concept_mixture_of_depths, concept_mixture_of_recursions, concept_mixture_of_experts [INFERRED 0.85]
- **Architectures compared under the equal-budget protocol** — concept_memora, concept_cortex, models_gpt2_gpt2, rationale_equal_budget_protocol [EXTRACTED 1.00]
- **Equal-budget comparison run: GPT-2 Small vs Memora GPU memory profile logged via compare.py/wandb** — results_wnb_gpu_memory_allocation_during_training_chart, models_gpt2_gpt2, models_memora_memora, training_compare_train_model [INFERRED 0.85]
- **Equal-budget train-loss comparison (GPT-2 Small vs Memora over tokens_seen)** — results_wnb_train_loss_per_token_seen, results_wnb_train_loss_per_token_seen_gpt2_small, results_wnb_train_loss_per_token_seen_memora, results_wnb_train_loss_per_token_seen_train_loss_metric [INFERRED 0.85]
- **Equal-budget throughput benchmark: GPT-2 Small vs Memora** — results_wnb_train_token_per_sec, models_gpt2_gpt2, models_memora_memora [INFERRED 0.75]

## Communities (14 total, 1 thin omitted)

### Community 0 - "Memora & Oneira Models"
Cohesion: 0.06
Nodes (38): GatedLinearAttention, Linear, MemoraConfig, apply_rope(), Block, build_rope_cache(), GatedLinearAttention, LocalAttention (+30 more)

### Community 1 - "Cortex, Interface & Deploy Pipeline"
Cohesion: 0.06
Nodes (45): ABC, BitLinear, CausalSelfAttention, CortexConfig, FeedForward, Tensor, Cortex - inspiré du cerveau humain.    Points clés du design :   - Poids ternair, FFN à deux couches avec activation ReLU² (Squared ReLU).      Expansion 4× stand (+37 more)

### Community 2 - "Training & Comparison Runs"
Cohesion: 0.06
Nodes (38): DataLoader, Dataset, GPT2, Module, GPT-2 Small (124M paramètres).      Architecture :         Token Embedding + Pos, Initialisation des poids selon le papier GPT-2., Memora, Module (+30 more)

### Community 3 - "Research Concepts & Papers"
Cohesion: 0.08
Nodes (46): ARC (ARC-easy / ARC-challenge), GLUE, HellaSwag, MMLU, PiQA, SuperGLUE, Winogrande, CLAUDE.md — RSI-Research Project Guide (+38 more)

### Community 4 - "Memora-GLA Variant"
Cohesion: 0.08
Nodes (23): apply_rope(), Block, build_rope_cache(), GatedLinearAttention, LocalAttention, MemoraConfig, MemoraGLA, Module (+15 more)

### Community 5 - "DeepSeek-Inspired Architecture"
Cohesion: 0.10
Nodes (21): apply_rope(), Block, build_rope_cache(), CompressedAttention, DeepSeek, DeepSeekConfig, HyperConnections, Module (+13 more)

### Community 6 - "Effy Architecture"
Cohesion: 0.10
Nodes (17): CausalSelfAttention, Effy, EffyConfig, FeedForward, LinearAttention, Module, Tensor, Forme chunkée : coût O(T·chunk), mémoire O(T) (pas de tenseur T×T).          q,k (+9 more)

### Community 7 - "GPT-2 Baseline"
Cohesion: 0.14
Nodes (11): CausalSelfAttention, FeedForward, GPT2Config, Tensor, Un bloc Transformer pre-norm.      Structure :         x → LayerNorm → Attention, Passe avant.          Args:             idx: (B, T) indices de tokens, avec T ≤, Charge les poids pré-entraînés depuis HuggingFace.          Usage :, Hyperparamètres GPT-2 Small (124M).      Hérite de BaseModelConfig. Les valeurs (+3 more)

### Community 8 - "Triton Ternary Kernel"
Cohesion: 0.14
Nodes (14): constexpr, dtype, Tensor, Noyau Triton pour le matmul ternaire de BitLinear.  Le chemin PyTorch (BitLinear, Matmul ternaire : y = dequant(quant(x) @ W_q.T).      x          : (..., in_feat, absmax fp32 par ligne, clampé à 1e-5 — un programme par ligne., ref_fn(), _rowwise_absmax_kernel() (+6 more)

### Community 9 - "Loss Comparison Chart"
Cohesion: 0.43
Nodes (7): Train Loss vs Tokens Seen (GPT-2 Small vs Memora), GPT-2 Small, Memora, Memora reaches lower train loss than GPT-2 Small at equal tokens-seen, tokens_seen (x-axis), train/loss metric, Weights & Biases (wandb) run export

## Ambiguous Edges - Review These
- `CausalSelfAttention` → `Cortex / Sparse-Cortex (ternary + GDN + SWA hybrid architecture)`  [AMBIGUOUS]
  models/cortex.md · relation: conceptually_related_to
- `CLAUDE.md — RSI-Research Project Guide` → `README.md — RSI-Research: Scaling with Sub-Quadratic Attention`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `Train Loss vs Tokens Seen (GPT-2 Small vs Memora)` → `Memora reaches lower train loss than GPT-2 Small at equal tokens-seen`  [AMBIGUOUS]
  results/wnb/train_loss_per_token_seen.png · relation: shares_data_with

## Knowledge Gaps
- **13 isolated node(s):** `rsi-research`, `Straight-Through Estimator (STE)`, `PiQA`, `ARC (ARC-easy / ARC-challenge)`, `HellaSwag` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `CausalSelfAttention` and `Cortex / Sparse-Cortex (ternary + GDN + SWA hybrid architecture)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `CLAUDE.md — RSI-Research Project Guide` and `README.md — RSI-Research: Scaling with Sub-Quadratic Attention`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Train Loss vs Tokens Seen (GPT-2 Small vs Memora)` and `Memora reaches lower train loss than GPT-2 Small at equal tokens-seen`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `LanguageModel` connect `Cortex, Interface & Deploy Pipeline` to `Memora & Oneira Models`, `Training & Comparison Runs`, `Research Concepts & Papers`, `Memora-GLA Variant`, `DeepSeek-Inspired Architecture`, `Effy Architecture`, `GPT-2 Baseline`?**
  _High betweenness centrality (0.391) - this node is a cross-community bridge._
- **Why does `BaseModelConfig` connect `Memora & Oneira Models` to `Cortex, Interface & Deploy Pipeline`, `Training & Comparison Runs`, `Research Concepts & Papers`, `Memora-GLA Variant`, `DeepSeek-Inspired Architecture`, `Effy Architecture`, `GPT-2 Baseline`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Why does `CLAUDE.md — RSI-Research Project Guide` connect `Research Concepts & Papers` to `Memora & Oneira Models`, `Cortex, Interface & Deploy Pipeline`, `Training & Comparison Runs`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `LanguageModel` (e.g. with `BitLinear` and `CausalSelfAttention`) actually correct?**
  _`LanguageModel` has 44 INFERRED edges - model-reasoned connections that need verification._