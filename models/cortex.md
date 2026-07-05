# Sparse-Cortex-LM — Spécification d'architecture

*Un LLM efficace en VRAM, inspiré des principes du cortex : poids ternaires, attention
hybride locale + récurrente, sparsité d'activation, et profondeur de calcul adaptative
par token.*

Document de conception destiné à la réimplémentation depuis une pipeline de pré-entraînement
existante (référence : GPT-2 small, ~124 M paramètres). Nom de code : **Sparse-Cortex** (à renommer).

---

## 0. Ce que ce document est (et n'est pas)

C'est un **plan de construction étagé**, pas une recette monolithique. Chaque levier est
validé isolément dans la littérature 2024–2026 ; leur empilement complet ne l'est pas. Le
plan (§12) vous fait ajouter **un levier à la fois**, chacun mesuré contre votre baseline
GPT-2 small. N'implémentez pas tout d'un coup.

---

## 1. Principe de conception : quel levier agit sur quoi

La confusion fatale à éviter : **VRAM** (stockage) et **calcul** (FLOPs/énergie/vitesse)
sont deux axes indépendants. Un levier qui écrase l'un ne touche pas forcément l'autre.

| Levier | VRAM (stockage) | Calcul / énergie | Analogie corticale |
|---|---|---|---|
| **Poids ternaires (BitNet b1.58)** | ✅✅✅ (levier principal) | ✅✅ (mult → add) | synapse excitatrice / inhibitrice / absente |
| **Attention hybride (GDN + SWA)** | ✅ (état borné, pas de KV-cache qui explose) | ✅✅ (linéaire vs quadratique) | mémoire locale récente + trace récurrente compressée |
| **Sparsité d'activation (ReLU²)** | ✗ | ✅✅ | décharge clairsemée dans la colonne |
| **Profondeur adaptative (MoR)** | ✅✅ (weight-tying) | ✅✅ (calcul selon difficulté) | neuromodulation / effort variable |
| **MoE** *(phase tardive)* | ✗ (tous les experts en mémoire) | ✅✅✅ | recrutement sélectif de colonnes |

**Conséquence directe pour votre objectif VRAM :** le duo gagnant est **ternaire (le plus gros
levier) + weight-tying via MoR**. Le MoE, lui, ne réduit *pas* la VRAM ; réservez-le au passage
à l'échelle pour le gain de *calcul*.

---

## 2. Vue d'ensemble

```
                        tokens
                          │
                  [ Embedding (bf16) ]         ← NON ternarisé
                          │
        ┌─────────────────┴─────────────────┐
        │   Bloc récurrent partagé (MoR)     │  ← rebouclé 1..R fois/token
        │  ┌──────────────────────────────┐  │
        │  │  RMSNorm                      │  │
        │  │  Mélangeur hybride :          │  │
        │  │    • couches Gated DeltaNet   │  │  ← mémoire longue compressée
        │  │    • + Sliding Window Attn    │  │  ← recall local exact
        │  │  RMSNorm                      │  │
        │  │  FFN ternaire, activation ReLU²│  │  ← sparsité d'activation
        │  └──────────────────────────────┘  │
        │   Routeur de profondeur (par token) │  ← neuromodulation
        └─────────────────┬─────────────────┘
                          │
                  [ RMSNorm final ]
                  [ LM Head (bf16) ]           ← NON ternarisé
                          │
                        logits
```

Tous les `nn.Linear` **à l'intérieur** du bloc (projections d'attention, FFN) sont remplacés
par `BitLinear` (ternaire). Embeddings, normes et tête de sortie restent en bf16.

---

## 3. Composant A — BitLinear (poids ternaires natifs)

Remplaçant direct de `nn.Linear`. Poids contraints à **{−1, 0, +1}** (≈ 1,58 bit/poids),
activations en **int8**. Entraîné *nativement* ternaire (jamais de quantization post-entraînement).

**Forward :**
1. Quantifier le poids par la moyenne des valeurs absolues (*absmean*) :
   - `γ = mean(|W|)`
   - `W_q = clamp(round(W / (γ + ε)), −1, +1)` → valeurs dans {−1, 0, +1}
2. Quantifier l'activation par `absmax` par token en int8 :
   - `x_q = round(clamp(x · 127 / max(|x|), −128, 127))`
3. Produit : `y = (W_q · x_q) · (γ · scale_x / 127)`.
   Le produit `W_q · x_q` n'est que des additions/soustractions (pas de multiplication).

**Backward — Straight-Through Estimator (STE) :**
On conserve un **poids latent pleine précision** `W` (le « fantôme » synaptique). Le forward
utilise `W_q`, mais le gradient est propagé *comme si* la quantification était l'identité :
`∂L/∂W ≈ ∂L/∂W_q`. Le fantôme continu apprend ; la transmission reste discrète.

**Règles d'or (pièges classiques) :**
- **Ne ternarisez PAS** : embeddings, RMSNorm, LM head. Gardez-les en bf16.
- Pas de biais dans les couches linéaires ni les normes (convention BitNet).
- Activations en **int8**, jamais ternaires — sinon le modèle s'effondre.
- Entraînement ternaire *from scratch* obligatoire.

**Analogie :** la *force* d'une synapse est discrète et stable (le poids ternaire) ; le *train
de spikes* qui la traverse porte une information plus riche (l'activation int8).

---

## 4. Composant B — Mélangeur de tokens hybride

Objectif : remplacer l'attention softmax globale (quadratique, KV-cache qui explose) par une
combinaison **mémoire récurrente compressée + fenêtre locale exacte**.

- **Gated DeltaNet (GDN)** : attention linéaire à *delta rule* + *gating*. État de taille fixe
  (pas de KV-cache croissant → VRAM bornée), coût linéaire en longueur. Le gating permet
  l'effacement rapide, la delta rule les mises à jour ciblées.
- **Sliding Window Attention (SWA)** : attention softmax exacte mais restreinte à une fenêtre
  locale (p. ex. 512 tokens). Récupère le *recall* local précis que l'état compressé perd.

**Agencement recommandé (pattern « GDN-H1 ») :** interleaving ~**3 couches GDN : 1 couche SWA**.
Le GDN compresse l'historique long ; le SWA garantit le recall local exact. C'est le meilleur
compromis actuel pour les petits modèles.

*Note :* l'état de taille fixe du GDN ne peut pas tout stocker — c'est *pourquoi* les couches SWA
sont là. Ne supprimez pas le SWA en pensant « gagner » ; vous perdriez le recall.

*Bloc GDN, détail des chemins :* q/k via proj. linéaire + short-conv (noyau 3–4) + SiLU + L2-norm ;
v via proj. + short-conv + SiLU ; portes decay/erase/write via branches linéaires séparées ;
sortie RMS-normalisée × porte SiLU, puis reprojetée.

---

## 5. Composant C — FFN à sparsité d'activation

FFN standard `W_down · act(W_up · x)`, mais avec une activation qui **induit nativement** la
sparsité (le *lazy neuron phenomenon* : ~90–95 % des neurones cachés sortent 0 par token).

- **Activation : Squared ReLU (ReLU²)** — `f(z) = max(0, z)²`. Sparsité émergente sans
  prédicteur ni bidouille. (Alternative : dReLU / *double ReLU* type TurboSparse.)
- À l'inférence, seules les colonnes de `W_down` correspondant aux neurones non nuls sont
  calculées → gain de calcul. En entraînement, laissez la sparsité émerger ; ajoutez au besoin
  une petite régularisation L1 sur les activations cachées pour la pousser.

**Sparsité double :** (colonne = expert via MoE, plus tard) × (neurones dans la colonne via ReLU²).
Clairsemé × clairsemé, comme une colonne corticale où seules quelques cellules déchargent.

---

## 6. Composant D — Mixture-of-Recursions (profondeur adaptative) *(phase 3)*

La « neuromodulation » : moduler le calcul **par token selon sa difficulté**, appris pendant
le pré-entraînement (pas une chain-of-thought post-hoc).

- Un **unique bloc partagé** (weight-tied) est **rebouclé** jusqu'à `R` fois (p. ex. R = 3).
- Un **routeur** attribue à chaque token une profondeur de récursion : token facile → 1 tour,
  token difficile → 3 tours.
- **Double bénéfice :** le partage de poids **réduit le nombre de paramètres → VRAM** ; le
  routage **réduit les FLOPs** (on ne reboucle que les tokens durs).
- **Cache KV par récursion** : ne stocker les paires K/V que pour les tokens encore actifs à
  chaque profondeur.

**Précurseur plus simple (phase 2, si MoR trop complexe d'emblée) :** *Mixture-of-Depths* — un
routeur par bloc laisse passer les top-k tokens dans le bloc, les autres court-circuitent via
le résiduel. Capacité fixe → graphe statique, plus simple à implémenter.

**Homéostasie (obligatoire) :** ajoutez une perte auxiliaire sur le routeur pour empêcher
l'effondrement (le routeur qui choisit toujours la profondeur min ou max). Probabilités de
profondeur en sigmoïde + perte de régularisation du routeur.

---

## 7. Composant E — Mixture-of-Experts *(phase tardive, passage à l'échelle uniquement)*

**Ne pas utiliser à 100 M.** Le MoE ne réduit pas la VRAM (tous les experts doivent résider en
mémoire pour être routables) et sous-performe à petite échelle. Il n'apporte son gain de
*calcul* qu'une fois le modèle gros (cible ~40 B).

Quand vous y viendrez :
- Remplacer le FFN par `N` experts, top-`k` activés par token (p. ex. N = 8, k = 2).
- **Homéostasie / load-balancing (obligatoire) :** perte auxiliaire (coeff ~0,01) forçant
  l'équilibre des charges entre experts. Sans elle : quelques experts monopolisent, les autres
  s'atrophient — un cortex « épileptique » en excitation runaway.

---

## 8. Configuration de référence — Phase 1 (~124 M, comparable GPT-2 small)

Modèle **dense, ternaire, attention hybride** — pas encore de MoR ni de MoE. C'est le socle
à comparer directement à votre baseline.

| Hyperparamètre | Valeur |
|---|---|
| Vocabulaire | ~50 257 (tokenizer GPT-2) ou le vôtre |
| `d_model` | 768 |
| Couches (physiques) | 12 |
| Têtes | 12 |
| Dimension FFN | 3072 (4×), activation **ReLU²** |
| Mélangeur | 3 GDN : 1 SWA (9 GDN + 3 SWA sur 12 couches) |
| Fenêtre SWA | 512 |
| Contexte d'entraînement | 1024 (puis 2048) |
| Normalisation | RMSNorm (bf16) |
| Position | RoPE dans les couches SWA ; GDN gère l'ordre nativement |
| Poids linéaires internes | **BitLinear ternaire** |
| Embedding / head | **bf16** (non ternarisés) |

Paramètres ≈ ceux de GPT-2 small, mais les poids linéaires internes pèsent ~1,58 bit.

---

## 9. Recette d'entraînement (quantization-aware, from scratch)

- **Natif ternaire dès l'initialisation.** Jamais de PTQ. Le réseau doit apprendre à vivre
  ternaire dès la naissance.
- **Poids latents pleine précision + STE** (cf. §3).
- **Learning rate plus élevé qu'en bf16** (BitNet utilise typiquement ~2× le LR bf16), avec un
  **schedule en deux temps** : weight decay actif en début, réduit/annulé ensuite.
- **Warmup** linéaire classique, puis cosine decay.
- **Optimiseur :** AdamW ; envisager **Muon** qui stabilise la convergence et réduit la
  perplexité à petite échelle (observé sur des configs BabyLM avec attention linéaire + SWA).
- **Pertes auxiliaires** à activer selon la phase :
  - MoR : perte de régularisation du routeur (anti-effondrement de profondeur).
  - MoE (tardif) : perte de load-balancing (coeff ~0,01).
  - FFN : régularisation L1 optionnelle sur activations cachées pour renforcer la sparsité.
- **Métrique de comparaison :** perplexité validation + accuracy few-shot (PiQA, ARC-e/c,
  HellaSwag, Winogrande) vs baseline GPT-2 small, à budget FLOPs *égal*.

---

## 10. Budget VRAM (poids seuls, indicatif)

| Config | Params | Poids linéaires @1,58 bit | Embeddings (bf16) | Ordre de grandeur total |
|---|---|---|---|---|
| Phase 1 (124 M) | ~124 M | ~24 Mo | ~77 Mo (50k×768) | dominé par l'embedding — tient trivialement |
| Cible (40 B) | ~40 B | **~7,9 Go** | quelques centaines de Mo | **tient dans 16 Go** avec marge activations/état |

À 124 M, le ternaire est presque « gratuit » et l'embedding domine — la démonstration de la
**victoire VRAM se voit à l'échelle**. À 40 B ternaire, ~8 Go de poids → cœur de votre RTX 5070 Ti.
(Pour aller au-delà, offload des experts dormants vers la RAM système du 5950X, style ktransformers —
mais latence accrue : phase ultérieure.)

---

## 11. Ordre de grandeur biologique (rappel de cadrage)

Les « neurones » ne sont pas les paramètres — les **synapses** le sont. Le noyau langagier du
cortex compte de l'ordre de 10⁸–10⁹ neurones, mais 10¹¹–10¹² synapses. Un modèle de 124 M–1 B
*paramètres* est donc dans l'ordre de grandeur des *neurones* langagiers, pas des synapses. La
révolution ne vient pas du *nombre* mais du **nombre de paramètres actifs par token** — ce que
tous les leviers ci-dessus s'emploient à écraser.

---

## 12. Plan d'implémentation étagé (à suivre dans l'ordre)

> **Règle :** un levier à la fois, chacun validé contre la baseline avant d'ajouter le suivant.

- **Étape 0 — Baseline.** Votre GPT-2 small actuel. Fige la référence (perplexité, few-shot).
- **Étape 1 — Ternaire seul.** Remplacer les `nn.Linear` internes par `BitLinear`, garder
  l'attention softmax classique. Objectif : converger et rester proche de la baseline. C'est le
  test le plus risqué numériquement — isolez-le.
- **Étape 2 — Attention hybride.** Remplacer softmax global par GDN + SWA (3:1). Vérifier
  perplexité et recall long-contexte.
- **Étape 3 — Sparsité d'activation.** Passer le FFN en ReLU² (ou dReLU). Vérifier le taux de
  sparsité effectif et l'absence de régression.
- **Étape 4 — Profondeur adaptative.** Ajouter MoD (simple) puis, si stable, MoR (weight-tying).
  Surveiller l'effondrement du routeur (homéostasie).
- **Étape 5 — Passage à l'échelle + MoE.** Seulement une fois le socle solide : grossir, puis
  introduire le MoE avec load-balancing pour le gain de calcul.

À chaque étape : si ça casse, vous savez *exactement* quel organe est en cause.

---

## 13. Pièges connus (checklist)

- [ ] Embeddings / normes / head **non** ternarisés (bf16).
- [ ] Activations en **int8**, pas ternaires.
- [ ] Entraînement **natif** ternaire, jamais PTQ.
- [ ] LR ~2× bf16, schedule weight-decay en deux temps.
- [ ] MoE ≠ gain VRAM ; réservé à l'échelle.
- [ ] Ne pas retirer les couches SWA (recall local).
- [ ] Pertes d'homéostasie pour tout routeur (MoR, MoE).
- [ ] Un levier à la fois.

---

## 14. Références (papiers clés)

- **BitNet b1.58** — *The Era of 1-bit LLMs* (Ma et al., arXiv:2402.17764) et **BitNet b1.58 2B4T**
  (rapport technique, arXiv:2504.12285) — poids ternaires natifs, BitLinear.
- **Gated DeltaNet** — *Gated Delta Networks: Improving Mamba2 with Delta Rule* (Yang et al.,
  arXiv:2412.06464) — attention linéaire hybride GDN + SWA (variante H1).
- **Mixture-of-Depths** — Raposo et al., arXiv:2404.02258 — calcul adaptatif par token.
- **Mixture-of-Recursions** — arXiv:2507.10524 — profondeur récursive adaptative + weight-tying.
- **Lazy neuron / sparsité** — *The Lazy Neuron Phenomenon* (Li et al., 2022) ; **TurboSparse/dReLU**,
  **CATS** (arXiv:2404.08763), **TEAL** (arXiv:2408.14690) pour la sparsité d'activation.

---

*Étape suivante logique : implémenter l'Étape 1 (BitLinear) et la brancher sur votre pipeline
existante pour valider la convergence ternaire avant tout le reste.*