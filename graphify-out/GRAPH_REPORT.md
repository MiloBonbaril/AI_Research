# Graph Report - RSI-research  (2026-07-12)

## Corpus Check
- 64 files · ~77,870 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 567 nodes · 1033 edges · 63 communities (21 shown, 42 thin omitted)
- Extraction: 85% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 149 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `92d75b6e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- compilerOptions
- package.json
- Sparse-Cortex-LM — Spécification d'architecture
- RSI-Research: Scaling with Sub-Quadratic Attention
- Optimization report — DeepSeek(mini) training speed & VRAM
- CLAUDE.md
- CLAUDE.md
- Architecture Atlas
- index.ts
- ARC (ARC-easy / ARC-challenge)
- GLUE
- HellaSwag
- MMLU
- PiQA
- SuperGLUE
- Winogrande
- AdamW optimizer
- BitLinear (native ternary weight linear layer)
- Cortex / Sparse-Cortex (ternary + GDN + SWA hybrid architecture)
- FlashAttention-2
- Gated DeltaNet (GDN)
- Gated Linear Attention (GLA)
- Mamba (State Space Model)
- Memora (hybrid sub-quadratic architecture)
- Mixture-of-Depths (MoD)
- Mixture-of-Experts (MoE)
- Mixture-of-Recursions (MoR)
- Muon optimizer
- Squared ReLU (ReLU^2) activation
- RoPE (Rotary Position Embedding)
- Sliding Window Attention (SWA)
- Straight-Through Estimator (STE)
- WSD schedule (Warmup-Stable-Decay)
- cortex.md — Sparse-Cortex-LM Architecture Specification
- Attention Is All You Need (arXiv:1706.03762)
- BitNet b1.58 — The Era of 1-bit LLMs (Ma et al., arXiv:2402.17764)
- BitNet b1.58 2B4T technical report (arXiv:2504.12285)
- CATS (arXiv:2404.08763)
- Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang et al., arXiv:2412.06464)
- Gated Linear Attention paper (arXiv:2302.10205)
- The Lazy Neuron Phenomenon (Li et al., 2022)
- Mamba: State Space Models for Data-Efficient Language Modeling (arXiv:2312.00752)
- Mixture-of-Depths (Raposo et al., arXiv:2404.02258)
- Mixture-of-Recursions (arXiv:2507.10524)
- RoPE: Rotary Position Embedding (arXiv:2104.09864)
- TEAL (arXiv:2408.14690)
- Equal-budget comparison protocol
- GDN-H1 interleaving pattern (3 GDN : 1 SWA)
- VRAM vs Compute independence principle

## God Nodes (most connected - your core abstractions)
1. `LanguageModel` - 66 edges
2. `BaseModelConfig` - 47 edges
3. `Memora` - 26 edges
4. `GPT2` - 24 edges
5. `MemoraConfig` - 22 edges
6. `GPT2Config` - 21 edges
7. `compilerOptions` - 21 edges
8. `TrainResult` - 18 edges
9. `Oneira` - 17 edges
10. `main()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `GPU Memory Allocation Chart: Memora vs GPT-2 Small (wandb system/gpu.0.memoryAllocated)` --references--> `Memora`  [EXTRACTED]
  results/wnb/gpu_memory_allocation_during_training.png → models/memora.py
- `train/tok_per_sec — GPT-2 Small vs Memora throughput` --references--> `Memora`  [EXTRACTED]
  results/wnb/train_token_per_sec.png → models/memora.py
- `GPU Memory Allocation Chart: Memora vs GPT-2 Small (wandb system/gpu.0.memoryAllocated)` --references--> `train_model()`  [INFERRED]
  results/wnb/gpu_memory_allocation_during_training.png → training/compare.py
- `train_model()` --references--> `train/tok_per_sec — GPT-2 Small vs Memora throughput`  [INFERRED]
  training/compare.py → results/wnb/train_token_per_sec.png
- `TrainResult` --uses--> `CortexConfig`  [INFERRED]
  training/compare.py → models/cortex.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Staged one-lever-at-a-time implementation plan (Cortex)** — concept_bitlinear, concept_gated_deltanet, concept_sliding_window_attention, concept_relu2, concept_mixture_of_recursions, concept_mixture_of_experts [EXTRACTED 1.00]
- **Adaptive-computation / routing mechanism family (MoD, MoR, MoE)** — concept_mixture_of_depths, concept_mixture_of_recursions, concept_mixture_of_experts [INFERRED 0.85]
- **Architectures compared under the equal-budget protocol** — concept_memora, concept_cortex, models_gpt2_gpt2, rationale_equal_budget_protocol [EXTRACTED 1.00]
- **Equal-budget comparison run: GPT-2 Small vs Memora GPU memory profile logged via compare.py/wandb** — results_wnb_gpu_memory_allocation_during_training_chart, models_gpt2_gpt2, models_memora_memora, training_compare_train_model [INFERRED 0.85]
- **Equal-budget train-loss comparison (GPT-2 Small vs Memora over tokens_seen)** — results_wnb_train_loss_per_token_seen, results_wnb_train_loss_per_token_seen_gpt2_small, results_wnb_train_loss_per_token_seen_memora, results_wnb_train_loss_per_token_seen_train_loss_metric [INFERRED 0.85]
- **Equal-budget throughput benchmark: GPT-2 Small vs Memora** — results_wnb_train_token_per_sec, models_gpt2_gpt2, models_memora_memora [INFERRED 0.75]

## Communities (63 total, 42 thin omitted)

### Community 0 - "Memora & Oneira Models"
Cohesion: 0.05
Nodes (44): GatedLinearAttention, Linear, MemoraConfig, apply_rope(), Block, build_rope_cache(), GatedLinearAttention, LocalAttention (+36 more)

### Community 1 - "Cortex, Interface & Deploy Pipeline"
Cohesion: 0.06
Nodes (34): ABC, CausalSelfAttention, CortexConfig, FeedForward, Tensor, Cortex - inspiré du cerveau humain.    Points clés du design :   - Poids ternair, FFN à deux couches avec activation ReLU² (Squared ReLU).      Expansion 4× stand, Un bloc Transformer pre-norm.      Structure :         x → LayerNorm → Attention (+26 more)

### Community 2 - "Training & Comparison Runs"
Cohesion: 0.07
Nodes (35): DataLoader, Dataset, GPT2, Module, GPT-2 Small (124M paramètres).      Architecture :         Token Embedding + Pos, Initialisation des poids selon le papier GPT-2., Passe avant.          Args:             idx: (B, T) indices de tokens, avec T ≤, Charge les poids pré-entraînés depuis HuggingFace.          Usage : (+27 more)

### Community 3 - "Research Concepts & Papers"
Cohesion: 0.11
Nodes (21): Cortex, Module, Cortex (124M paramètres).      Architecture :         Token Embedding + Position, Initialisation des poids Cortex., Passe avant.          Args:             idx: (B, T) indices de tokens, avec T ≤, Effy, Module, Effy Small (~124M paramètres).      Architecture : identique à GPT-2 Small — 3 b (+13 more)

### Community 4 - "Memora-GLA Variant"
Cohesion: 0.08
Nodes (23): apply_rope(), Block, build_rope_cache(), GatedLinearAttention, LocalAttention, MemoraConfig, MemoraGLA, Module (+15 more)

### Community 5 - "DeepSeek-Inspired Architecture"
Cohesion: 0.09
Nodes (22): apply_rope(), _blk_causal(), Block, build_rope_cache(), CompressedAttention, DeepSeek, DeepSeekConfig, HyperConnections (+14 more)

### Community 6 - "Effy Architecture"
Cohesion: 0.13
Nodes (13): CausalSelfAttention, EffyConfig, FeedForward, LinearAttention, Tensor, Forme chunkée : coût O(T·chunk), mémoire O(T) (pas de tenseur T×T).          q,k, FFN à deux couches avec GELU — identique à gpt2.FeedForward., Un bloc Transformer pre-norm — structure identique à gpt2.TransformerBlock. (+5 more)

### Community 7 - "GPT-2 Baseline"
Cohesion: 0.11
Nodes (21): canvas, data, hexColor(), renderInfo(), renderLegend(), scene, selectModel(), ArchitectureScene (+13 more)

### Community 8 - "Triton Ternary Kernel"
Cohesion: 0.09
Nodes (33): constexpr, dtype, BitLinear, Remplaçant de nn.Linear avec poids ternaires et activations int8.      Forward :, Tensor, Noyau Triton pour le matmul ternaire de BitLinear.  Le chemin PyTorch (BitLinear, Matmul ternaire : y = dequant(quant(x) @ W_q.T).      x          : (..., in_feat, absmax fp32 par ligne, clampé à 1e-5 — un programme par ligne. (+25 more)

### Community 9 - "Loss Comparison Chart"
Cohesion: 0.43
Nodes (7): Train Loss vs Tokens Seen (GPT-2 Small vs Memora), GPT-2 Small, Memora, Memora reaches lower train loss than GPT-2 Small at equal tokens-seen, tokens_seen (x-axis), train/loss metric, Weights & Biases (wandb) run export

### Community 14 - "compilerOptions"
Cohesion: 0.08
Nodes (25): bun, DOM, DOM.Iterable, ESNext, compilerOptions, allowImportingTsExtensions, allowJs, jsx (+17 more)

### Community 15 - "package.json"
Cohesion: 0.10
Nodes (19): three, @types/bun, @types/three, typescript, dependencies, three, devDependencies, @types/bun (+11 more)

### Community 16 - "Sparse-Cortex-LM — Spécification d'architecture"
Cohesion: 0.12
Nodes (16): 0. Ce que ce document est (et n'est pas), 10. Budget VRAM (poids seuls, indicatif), 11. Ordre de grandeur biologique (rappel de cadrage), 12. Plan d'implémentation étagé (à suivre dans l'ordre), 13. Pièges connus (checklist), 14. Références (papiers clés), 1. Principe de conception : quel levier agit sur quoi, 2. Vue d'ensemble (+8 more)

### Community 17 - "RSI-Research: Scaling with Sub-Quadratic Attention"
Cohesion: 0.12
Nodes (15): 1. GPT-2 (Baseline), 2. Memora (Hybride Sub-Quadratique), 🏗️ Architectures, Entraînement comparatif, 🧪 Expérimentations futures, 🏃‍♂️ Exécution, 🛠️ Installation, Installation des dépendances (+7 more)

### Community 18 - "Optimization report — DeepSeek(mini) training speed & VRAM"
Cohesion: 0.13
Nodes (14): 10. Shared lever: cross-entropy materializes ~1 GB for both models, 1. The window branch runs on the math SDPA backend — the main VRAM & speed killer, 2. `torch.compile` currently compiles nothing — for any model **[VERIFIED]**, 3. Masks and indices rebuilt in every layer, every step, 4. `loss.item()` inside the micro-batch loop — 16 GPU syncs per step, 5. Small free wins in the training loop, 6. Refill the freed VRAM with batch size, 7. Fuse per-layer input projections into one GEMM (+6 more)

### Community 19 - "CLAUDE.md"
Cohesion: 0.29
Nodes (5): Architecture, Commands, Gotchas, graphify, What this is

### Community 20 - "CLAUDE.md"
Cohesion: 0.50
Nodes (3): APIs, Frontend, Testing

### Community 21 - "Architecture Atlas"
Cohesion: 0.50
Nodes (3): Architecture Atlas, Data, Run

## Ambiguous Edges - Review These
- `Train Loss vs Tokens Seen (GPT-2 Small vs Memora)` → `Memora reaches lower train loss than GPT-2 Small at equal tokens-seen`  [AMBIGUOUS]
  results/wnb/train_loss_per_token_seen.png · relation: shares_data_with

## Knowledge Gaps
- **127 isolated node(s):** `rsi-research`, `server`, `name`, `module`, `type` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Train Loss vs Tokens Seen (GPT-2 Small vs Memora)` and `Memora reaches lower train loss than GPT-2 Small at equal tokens-seen`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `LanguageModel` connect `Cortex, Interface & Deploy Pipeline` to `Memora & Oneira Models`, `Training & Comparison Runs`, `Research Concepts & Papers`, `Memora-GLA Variant`, `DeepSeek-Inspired Architecture`, `Effy Architecture`, `Triton Ternary Kernel`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `BaseModelConfig` connect `Cortex, Interface & Deploy Pipeline` to `Memora & Oneira Models`, `Training & Comparison Runs`, `Research Concepts & Papers`, `Memora-GLA Variant`, `DeepSeek-Inspired Architecture`, `Effy Architecture`, `Triton Ternary Kernel`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `BitLinear` connect `Triton Ternary Kernel` to `Cortex, Interface & Deploy Pipeline`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `LanguageModel` (e.g. with `BitLinear` and `CausalSelfAttention`) actually correct?**
  _`LanguageModel` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `BaseModelConfig` (e.g. with `BitLinear` and `CausalSelfAttention`) actually correct?**
  _`BaseModelConfig` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `Memora` (e.g. with `GPT2` and `BaseModelConfig`) actually correct?**
  _`Memora` has 10 INFERRED edges - model-reasoned connections that need verification._