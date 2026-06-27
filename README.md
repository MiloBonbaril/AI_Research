# RSI-Research: Scaling with Sub-Quadratic Attention

**RSI-Research** est un projet de recherche expérimental visant à construire un LLM de nouvelle génération. Notre objectif principal est de dépasser l'architecture transformer traditionnelle en intégrant des mécanismes d'attention sub-quadratiques tout en maintenant un budget de paramètres comparable aux modèles de référence comme GPT-2.

Le projet est structuré autour de deux expériences parallèles menées dans `train.py`:
1.  **Architecture baseline (GPT-2)** : Entraînement du modèle standard pour établir une référence de performance.
2.  **Architecture expérimentale (Memora)** : Entraînement d'un modèle hybride incorporant de l'attention locale et des couches récurrentes (GLA) pour évaluer l'efficacité des mécanismes sub-quadratiques.

---

## 🛠️ Installation

### Prérequis

-   Python 3.10+
-   PyTorch 2.0+ avec support CUDA (recommandé pour les comparaisons de performance)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Exécution

### Entraînement comparatif

Pour lancer l'entraînement et comparer les deux modèles, utilisez la commande suivante:

```bash
python train.py --duration 3600
```

Cette commande lancera:
1.  L'entraînement de `gpt2.GPT2` pendant 1 heure.
2.  L'entraînement de `memora.Memora` pendant 1 heure.
3.  L'évaluation des deux modèles sur wikitext-103.
4.  L'affichage des résultats comparatifs (loss, perplexity, performance).

#### Options de configuration

Vous pouvez ajuster les paramètres d'entraînement via les arguments de ligne de commande:

-   `--duration <seconds>` : Durée de l'entraînement pour chaque modèle (par défaut : 3600s).
-   `--batch-size <size>` : Taille du batch (par défaut : 8).
-   `--grad-accum <steps>` : Accumulation de gradient (par défaut : 8).
-   `--use-compile` : Active la compilation PyTorch 2.0 (`torch.compile`) pour accélérer l'entraînement.

---

## 🏗️ Architectures

### 1. GPT-2 (Baseline)

Implémenté dans `gpt2.py`, ce modèle sert de référence pour évaluer les gains potentiels de l'architecture expérimentale.

### 2. Memora (Hybride Sub-Quadratique)

Implémenté dans `memora.py`, ce modèle combine:
-   **Attention locale** : Fenêtre glissante pour réduire la complexité quadratique.
-   **Gated Linear Attention (GLA)** : Couches récurrentes pour une mémoire efficace à coût constant.
-   **Configuration optimisée** : Têtes de query réduites (`n_kv_heads=3`), dimensions optimisées et techniques modernes (RoPE, RMSNorm, SwiGLU) pour un budget de paramètres de ~124-127M.

---

## 🧪 Expérimentations futures

Ce projet constitue une base pour explorer davantage d'architectures hybrides. Voici quelques pistes de recherche:

### Optimisation de Memora
-   Ajuster `n_kv_heads` et `head_dim` pour explorer le trade-off coût/performance.
-   Expérimenter avec `global_layers` pour trouver le bon équilibre entre attention locale et globale.
-   Tester différents `recurrent_layers` et `sliding_window`.

### Intégration d'autres mécanismes sub-quadratiques
-   Ajouter des couches Mamba (SSM) pour améliorer la capacité de mémoire.
-   Implémenter FlashAttention-2 pour accélérer les couches d'attention standard.

### Évaluation complète
-   Évaluer les modèles sur des benchmarks downstream (GLUE, SuperGLUE, MMLU).
-   Mesurer la capacité de contexte long des modèles Memora expérimentaux.

---

## 📚 Références

-   [Transformer Architecture](https://arxiv.org/abs/1706.03762) - Attention Is All You Need
-   [Gated Linear Attention](https://arxiv.org/abs/2302.10205) - Gated Linear Attention (GLA)
-   [RoPE](https://arxiv.org/abs/2104.09864) - RoPE: Rotary Position Embedding
-   [Mamba](https://arxiv.org/abs/2312.00752) - State Space Models for Data-Efficient Language Modeling
