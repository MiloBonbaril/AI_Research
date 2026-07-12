"""
DeepSeek(mini) — inspiré de DeepSeek-V4 (arXiv:2606.19348) à budget GPT-2 Small (≤126M).

Mécanismes repris de l'architecture réelle (papier + mHC, arXiv:2512.24880), redimensionnés
pour 126M de paramètres (le papier vise 1.6T/284B avec MoE + contexte 1M tokens ; ici on
n'implémente que ce qui a un sens à notre échelle — voir "Écarts vs le papier" plus bas) :

  - **mHC (Manifold-Constrained Hyper-Connections)** : le flux résiduel est étendu à
    `hc_streams` (n) flux parallèles par token. Chaque sous-couche (attention OU MLP) lit une
    combinaison pondérée des n flux (poids `A`, sigmoïde), applique la sous-couche, puis
    réinjecte la sortie (poids `C`, sigmoïde) tout en propageant les flux via une matrice n×n
    `B` — CONTRAINTE au polytope de Birkhoff (doublement stochastique, via Sinkhorn-Knopp,
    20 itérations). Cette contrainte préserve la propriété de mapping identité peu importe la
    profondeur d'empilement (une matrice doublement stochastique ne peut ni exploser ni
    effondrer le signal), contrairement aux Hyper-Connections "vanilla" (Zhu et al. 2024) dont
    les matrices de mixing apprises librement sont instables en profondeur.
    Formule (papier) : X_{l+1} = B_l·X_l + C_l·F_l(A_l·X_l).

  - **CSA (Compressed Sparse Attention)** : le contenu K/V est compressé par blocs (pooling
    appris, pondération softmax intra-bloc) puis un "lightning indexer" sélectionne les
    top-k blocs compressés les plus pertinents par requête (MQA sur les blocs sélectionnés).

  - **HCA (Heavily Compressed Attention)** : même mécanisme de compression mais avec un bloc
    BEAUCOUP plus large (`hca_block` ≫ `csa_block`) et SANS sélection top-k — attention dense
    sur tous les blocs compressés (déjà peu nombreux). Utilisée sur la majorité des couches
    (moins chère), CSA sur une minorité (meilleur recall) — même logique de ratio que
    Cortex (3 GDN : 1 SWA, cf. models/cortex.md).

  - **Branche fenêtre locale** : chaque couche d'attention combine sa branche compressée
    (CSA ou HCA) avec une branche attention locale exacte (RoPE + GQA), fusionnées par une
    porte apprise par tête — le papier décrit cette branche comme un "supplément" alongside
    les entrées compressées, pas une couche séparée (contrairement à Cortex/Memora qui
    alternent des couches de TYPES différents).

Écarts délibérés vs le papier (budget 126M, pas de service d'inférence à 1M tokens) :
  - Pas de MoE (le papier utilise DeepSeekMoE) : à 126M le MoE ne réduit pas la VRAM et
    sous-performe (cf. models/cortex.md §1/§7) — FFN dense SwiGLU à la place.
  - Pas de MLA : le papier lui-même n'en utilise PAS pour V4 (attention standard + compression).
  - Pas de "grouped output projection" (détail d'optimisation paramétrique) : o_proj standard.
  - RoPE appliqué en entier sur head_dim (pas de partial-RoPE 64-dim + trick position -i sur
    les entrées compressées) : les entrées compressées n'ont PAS de RoPE (leur position est
    déjà absorbée par le pooling), exactement comme les couches GLA de Memora sautent RoPE.
  - Pas d'optimiseur Muon (choix d'entraînement, hors périmètre du fichier modèle).

Suit le contrat models.model_interface.LanguageModel, comme gpt2.GPT2/cortex.Cortex/memora.Memora.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_interface import BaseModelConfig, LanguageModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepSeekConfig(BaseModelConfig):
    """Hyperparamètres DeepSeek(mini). Hérite des champs de BaseModelConfig."""
    vocab_size: int = 50257
    context_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    head_dim: int = 64          # n_embd // n_head

    # --- mHC ---
    hc_streams: int = 4          # n : nombre de flux résiduels parallèles
    sinkhorn_iters: int = 20     # itérations Sinkhorn-Knopp → polytope de Birkhoff

    # --- attention hybride compressée ---
    csa_layers: tuple = (3, 7, 11)   # couches CSA (sélection fine) ; le reste = HCA
    d_compress: int = 128            # dim du contenu compressé (avant projection K/V)
    csa_block: int = 16              # taille de bloc CSA (m)
    hca_block: int = 64              # taille de bloc HCA (m' ≫ m)
    csa_topk: int = 8                # top-k blocs compressés sélectionnés (CSA)
    index_dim: int = 16              # dim par tête de l'indexeur "lightning"
    index_heads: int = 2
    n_kv_heads_window: int = 3       # GQA pour la branche fenêtre locale
    sliding_window: int = 128

    # --- position / normalisation ---
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    use_qk_norm: bool = True

    # --- FFN (dense, pas de MoE — cf. docstring) ---
    d_ff: int = 2432

    tie_embeddings: bool = True
    bias: bool = False


# ---------------------------------------------------------------------------
# Briques de base
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """cos/sin de forme (seq_len, head_dim) pour RoPE (convention half-split)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    """MLP SwiGLU : silu(W_gate x) * (W_up x) → W_down. FFN dense (pas de MoE, cf. docstring)."""

    def __init__(self, c: DeepSeekConfig):
        super().__init__()
        self.w_gate = nn.Linear(c.n_embd, c.d_ff, bias=c.bias)
        self.w_up   = nn.Linear(c.n_embd, c.d_ff, bias=c.bias)
        self.w_down = nn.Linear(c.d_ff, c.n_embd, bias=c.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# mHC — Manifold-Constrained Hyper-Connections
# ---------------------------------------------------------------------------

class HyperConnections(nn.Module):
    """Enveloppe une sous-couche (attention OU mlp) dans un flux résiduel à n voies.

    X: (B,T,n,d). Lecture pondérée (A) → sous-couche → écriture (C) + propagation
    inter-flux via une matrice n×n B contrainte doublement-stochastique (Sinkhorn-Knopp).
    Composantes statiques (par couche) + dynamiques (dépendantes du token), même idiome
    que le forget-gate de GatedLinearAttention dans models/memora.py (a_low/a_high + a_bias).
    """

    def __init__(self, c: DeepSeekConfig):
        super().__init__()
        n = c.hc_streams
        self.n = n
        self.norm = RMSNorm(c.n_embd, c.norm_eps)
        self.dropout = c.dropout

        # statique : A/C initialisés "ouverts" (sigmoid(4)≈0.98 → proche d'un résiduel
        # classique), B initialisé proche de l'identité (diagonale dominante en logits).
        self.static_A = nn.Parameter(torch.full((n,), 4.0))
        self.static_C = nn.Parameter(torch.full((n,), 4.0))
        self.static_B = nn.Parameter(torch.eye(n) * 4.0)

        # dynamique : résumé inter-flux (mean sur n) → petite projection → deltas.
        # Poids à zéro à l'init (cf. DeepSeek.__init__) : démarre pur statique, la
        # dépendance au token s'apprend ensuite (même trick que a_bias dans Memora).
        self.dyn_proj = nn.Linear(c.n_embd, n + n + n * n, bias=False)

        self.iters = c.sinkhorn_iters

    @staticmethod
    def _sinkhorn(log_alpha: torch.Tensor, iters: int) -> torch.Tensor:
        """Projette log_alpha (…,n,n) sur le polytope de Birkhoff (doublement stochastique)."""
        for _ in range(iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return log_alpha.exp()

    def forward(self, X: torch.Tensor, sublayer, *args) -> torch.Tensor:
        B, T, n, d = X.shape
        pooled = X.mean(dim=2)                                    # (B,T,d) résumé inter-flux
        raw = self.dyn_proj(pooled)
        dA, dC, dB = raw.split([n, n, n * n], dim=-1)
        A = torch.sigmoid(self.static_A + dA)                     # (B,T,n) lecture
        Cw = torch.sigmoid(self.static_C + dC)                    # (B,T,n) écriture
        Braw = self.static_B + dB.view(B, T, n, n)                 # (B,T,n,n)
        Bm = self._sinkhorn(Braw, self.iters)                      # doublement stochastique

        h_in = torch.einsum("btn,btnd->btd", A, X)
        h_out = sublayer(self.norm(h_in), *args)
        h_out = F.dropout(h_out, p=self.dropout, training=self.training)

        X_prop  = torch.einsum("btij,btjd->btid", Bm, X)
        X_write = torch.einsum("btn,btd->btnd", Cw, h_out)
        return X_prop + X_write


# ---------------------------------------------------------------------------
# CSA / HCA — attention à KV compressée + branche fenêtre locale
# ---------------------------------------------------------------------------

class CompressedAttention(nn.Module):
    """Branche compressée (CSA si use_topk, HCA sinon) + branche fenêtre locale, fusionnées
    par une porte apprise par tête. Attention causale à deux niveaux :
      1. masque de causalité "bloc" : un bloc compressé n'est utilisable que par les requêtes
         situées APRÈS son dernier token source (sinon fuite du futur à travers le pooling).
      2. (CSA) sélection top-k parmi les blocs déjà causalement valides.
    """

    def __init__(self, c: DeepSeekConfig, block: int, use_topk: bool, topk: int = 0):
        super().__init__()
        assert c.n_head % c.n_kv_heads_window == 0, "n_head doit être multiple de n_kv_heads_window"
        self.n_head, self.hd = c.n_head, c.head_dim
        self.n_kv_win = c.n_kv_heads_window
        self.block = block
        self.use_topk = use_topk
        self.topk = topk
        self.dropout = c.dropout
        self.sliding_window = c.sliding_window

        # Projections d'entrée fusionnées en UN GEMM sur x (même idiome que c_attn de GPT-2) :
        # q | k_win | v_win | contenu compressé | gate [| iq indexeur] — mêmes matrices
        # qu'avant (mêmes tailles, même init par tranche), un seul launch de kernel.
        # ponytail: un seul flux de compression par bloc (le papier utilise deux flux
        # chevauchants C^a/C^b) — plus simple, quantité de blocs déjà généreuse à cette échelle.
        self.splits = [self.n_head * self.hd,      # q — partagée par les deux branches
                       self.n_kv_win * self.hd,    # k fenêtre locale (GQA)
                       self.n_kv_win * self.hd,    # v fenêtre locale
                       c.d_compress,               # contenu → pooling appris par bloc
                       2 * self.n_head]            # porte de fusion par tête
        if use_topk:
            self.index_dim = c.index_dim
            self.index_heads = c.index_heads
            self.splits.append(c.index_dim * c.index_heads)   # requêtes de l'indexeur
            self.ik_proj = nn.Linear(c.d_compress, c.index_dim * c.index_heads, bias=c.bias)
            # ponytail: poids d'agrégation des têtes de l'indexeur statiques (appris mais pas
            # dépendants du token) — le papier les génère par token (w_{t,h}) ; un vecteur
            # global suffit pour apprendre "quelle tête d'index pondérer davantage" à 126M.
            self.idx_w = nn.Parameter(torch.ones(c.index_heads))
        self.in_proj = nn.Linear(c.n_embd, sum(self.splits), bias=c.bias)

        self.pool_score = nn.Linear(c.d_compress, 1, bias=c.bias)
        self.k_comp = nn.Linear(c.d_compress, self.hd, bias=c.bias)
        self.v_comp = nn.Linear(c.d_compress, self.hd, bias=c.bias)
        self.sink = nn.Parameter(torch.zeros(self.n_head))   # attention sink par tête

        self.o_proj = nn.Linear(self.n_head * self.hd, c.n_embd, bias=c.bias)

        self.use_qk_norm = c.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.hd, c.norm_eps)
            self.k_norm_win = RMSNorm(self.hd, c.norm_eps)
            self.k_norm_comp = RMSNorm(self.hd, c.norm_eps)

    def _compress(self, c: torch.Tensor) -> torch.Tensor:
        """Pooling appris (softmax intra-bloc) : (B,T,d_c) → (B,nb,d_c)."""
        B, T, _ = c.shape
        m = self.block
        pad = (m - T % m) % m
        if pad:
            c = F.pad(c, (0, 0, 0, pad))
        nb = (T + pad) // m
        c = c.view(B, nb, m, -1)
        w = self.pool_score(c).softmax(dim=2)                 # (B,nb,m,1)
        return (c * w).sum(dim=2)                             # (B,nb,d_c)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                win_allowed: torch.Tensor, causal_block: torch.Tensor) -> torch.Tensor:
        # win_allowed (T,T) bool et causal_block (T,nb) bool ne dépendent que de T →
        # construits UNE fois par forward dans DeepSeek.forward, pas dans chaque couche.
        B, T, _ = x.shape
        H, hd = self.n_head, self.hd

        parts = self.in_proj(x).split(self.splits, dim=-1)       # un seul GEMM d'entrée
        q_raw, kw_raw, vw_raw, c_lat, gate_raw = parts[:5]

        q = q_raw.view(B, T, H, hd).transpose(1, 2)              # (B,H,T,hd)
        if self.use_qk_norm:
            q = self.q_norm(q)
        q_win = apply_rope(q, cos, sin)                          # branche fenêtre : avec RoPE
        # q (sans RoPE) sert à la branche compressée : les entrées compressées n'ont pas de
        # position propre (absorbée par le pooling), donc pas de rotation à leur appliquer.

        # --- branche fenêtre locale ---
        k_w = kw_raw.view(B, T, self.n_kv_win, hd).transpose(1, 2)
        v_w = vw_raw.view(B, T, self.n_kv_win, hd).transpose(1, 2)
        if self.use_qk_norm:
            k_w = self.k_norm_win(k_w)
        k_w = apply_rope(k_w, cos, sin)
        # Masque BOOLÉEN + têtes KV expansées (pas de enable_gqa) : rend l'appel éligible
        # au backend memory-efficient de SDPA. L'ancien masque float -inf + enable_gqa
        # forçait le backend math ((B,H,T,T) matérialisée) — mêmes maths, sans ce coût.
        rep = H // self.n_kv_win
        y_win = F.scaled_dot_product_attention(
            q_win, k_w.repeat_interleave(rep, dim=1), v_w.repeat_interleave(rep, dim=1),
            attn_mask=win_allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                        # (B,H,T,hd)

        # --- branche compressée (CSA ou HCA) ---
        comp = self._compress(c_lat)                             # (B,nb,d_c)
        nb = comp.size(1)
        k_c = self.k_comp(comp)                                  # (B,nb,hd)
        v_c = self.v_comp(comp)                                  # (B,nb,hd)
        if self.use_qk_norm:
            k_c = self.k_norm_comp(k_c)

        if self.use_topk:
            iq = parts[5].view(B, T, self.index_heads, self.index_dim)
            ik = self.ik_proj(comp).view(B, nb, self.index_heads, self.index_dim)
            dots = torch.einsum("bthd,bshd->bhts", iq, ik)                 # (B,Hidx,T,nb)
            idx_score = torch.einsum("bhts,h->bts", F.relu(dots), self.idx_w)
            idx_score = idx_score * (self.index_dim ** -0.5)               # (B,T,nb)
            idx_score = idx_score.masked_fill(~causal_block.unsqueeze(0), float("-inf"))
            k_sel = min(self.topk, nb)
            _, top_i = idx_score.detach().topk(k_sel, dim=-1)
            sel = torch.zeros_like(idx_score, dtype=torch.bool).scatter_(-1, top_i, True)
            comp_mask = sel & causal_block.unsqueeze(0)           # (B,T,nb) — ceinture+bretelles
            # le top-k lui-même (indices) est non-différentiable ; en biaisant les logits
            # d'attention PAR le score de l'indexeur (plutôt qu'un simple 0/-inf), le gradient
            # remonte vers iq_proj/ik_proj/idx_w — sinon l'indexeur ne reçoit jamais de signal
            # et la sélection reste figée à l'initialisation (cf. gating différentiable des MoE).
            attn_bias = torch.where(comp_mask, idx_score, torch.full_like(idx_score, float("-inf")))
        else:
            comp_mask = causal_block.unsqueeze(0)                 # (1,T,nb) → broadcast batch
            attn_bias = torch.where(comp_mask, 0.0, float("-inf"))

        attn_bias = attn_bias.to(q.dtype)                                    # (B|1,T,nb)

        logits = torch.einsum("bhtd,bsd->bhts", q, k_c) * (hd ** -0.5)       # (B,H,T,nb)
        logits = logits + attn_bias.unsqueeze(1)
        sink = self.sink.view(1, H, 1, 1).expand(B, H, T, 1)
        combined = torch.cat([logits, sink], dim=-1)                        # (B,H,T,nb+1)
        attn = combined.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_kv = attn[..., :nb]
        y_comp = torch.einsum("bhts,bsd->bhtd", attn_kv, v_c)                # (B,H,T,hd)

        # --- fusion par porte apprise (par tête) ---
        g = torch.sigmoid(gate_raw).view(B, T, H, 2).permute(0, 2, 1, 3)      # (B,H,T,2)
        y = g[..., 0:1] * y_comp + g[..., 1:2] * y_win

        y = y.transpose(1, 2).contiguous().view(B, T, H * hd)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Bloc
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, c: DeepSeekConfig, is_csa: bool):
        super().__init__()
        block = c.csa_block if is_csa else c.hca_block
        self.attn = CompressedAttention(c, block=block, use_topk=is_csa, topk=c.csa_topk)
        self.mlp = SwiGLU(c)
        self.hc_attn = HyperConnections(c)
        self.hc_mlp = HyperConnections(c)

    def forward(self, X: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                win_allowed: torch.Tensor, blk_causal: dict) -> torch.Tensor:
        X = self.hc_attn(X, self.attn, cos, sin, win_allowed, blk_causal[self.attn.block])
        X = self.hc_mlp(X, self.mlp)
        return X


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------

class DeepSeek(LanguageModel):
    """DeepSeek(mini) — mHC + hybride CSA/HCA, budget GPT-2 Small (≤126M paramètres)."""

    def __init__(self, config: DeepSeekConfig | None = None):
        super().__init__()
        self.config = config or DeepSeekConfig()
        c = self.config
        assert c.n_embd == c.n_head * c.head_dim, "n_embd doit == n_head * head_dim"
        assert all(0 <= i < c.n_layer for i in c.csa_layers), "csa_layers hors bornes"

        self.hc_streams = c.hc_streams
        self.tok_embd = nn.Embedding(c.vocab_size, c.n_embd)

        csa_set = set(c.csa_layers)
        self.blocks = nn.ModuleList([Block(c, is_csa=(i in csa_set)) for i in range(c.n_layer)])

        self.norm_f = RMSNorm(c.n_embd, c.norm_eps)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_embd.weight

        cos, sin = build_rope_cache(c.context_len, c.head_dim, c.rope_theta, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaling résiduel de sortie (1/√(2·n_layer)) + dyn_proj mHC à zéro (cf. HyperConnections)
        scale = 0.02 / math.sqrt(2 * c.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)
            elif name.endswith("dyn_proj.weight"):
                nn.init.zeros_(p)

        print(f"DeepSeek(mini) initialisé — {self.num_params()/1e6:.1f}M paramètres "
              f"({c.n_layer} couches, CSA={c.csa_layers}, flux mHC={c.hc_streams})")

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device, dtype):
        if T > self.rope_cos.size(0):
            cos, sin = build_rope_cache(T, self.config.head_dim, self.config.rope_theta,
                                        device, torch.float32)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:T].to(dtype), self.rope_sin[:T].to(dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.config.context_len, (
            f"Séquence trop longue ({T} > {self.config.context_len})"
        )
        cos, sin = self._rope(T, idx.device, self.tok_embd.weight.dtype)

        # Masques ne dépendant que de T — construits UNE fois par forward et passés aux
        # couches (ils étaient identiques dans les 12 couches).
        i = torch.arange(T, device=idx.device)[:, None]
        j = torch.arange(T, device=idx.device)[None, :]
        win_allowed = (i >= j) & (i - j < self.config.sliding_window)     # (T,T) bool
        blk_causal = {}
        for m in {self.config.csa_block, self.config.hca_block}:
            Tp = T + (m - T % m) % m
            # dernier index de token SOURCE (séquence réelle, non paddée) de chaque bloc.
            # Un bloc dont la fin dépasse T (bloc final tronqué par le padding) reste ainsi
            # inatteignable par toute requête réelle → aucune fuite via le padding.
            end_idx = torch.arange(m, Tp + 1, m, device=idx.device) - 1   # (nb,)
            blk_causal[m] = end_idx[None, :] <= i                         # (T,nb) bool

        x = self.tok_embd(idx)                                            # (B,T,d)
        # flux mHC initial : x réparti également entre les n flux (leur somme = x),
        # cohérent avec un résiduel classique avant tout apprentissage.
        X = (x / self.hc_streams).unsqueeze(2).expand(B, T, self.hc_streams, -1).contiguous()

        for block in self.blocks:
            X = block(X, cos, sin, win_allowed, blk_causal)

        x = X.sum(dim=2)                                                  # collapse des flux
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_name: str = "deepseek") -> "DeepSeek":
        raise NotImplementedError(
            "DeepSeek(mini) est une architecture originale — entraîner from scratch."
        )


# ---------------------------------------------------------------------------
# Self-check : Sinkhorn, mHC, CSA/HCA, causalité, budget params
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Sinkhorn-Knopp → matrice doublement stochastique (polytope de Birkhoff)
    # (100 itérations ici pour valider la convergence de l'algo ; le modèle réel n'a besoin
    # que de 20 itérations car ses logits d'entrée sont beaucoup moins dispersés — proches
    # de l'identité à l'init, cf. HyperConnections.static_B).
    logits = torch.randn(2, 5, 4, 4) * 3
    Bm = HyperConnections._sinkhorn(logits, iters=100)
    row_sums = Bm.sum(-1)
    col_sums = Bm.sum(-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), "lignes non stochastiques"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2), "colonnes non stochastiques"
    assert (Bm >= 0).all(), "entrées négatives (hors polytope)"
    print("Sinkhorn-Knopp OK — matrice doublement stochastique (lignes/colonnes ≈ 1)")

    # 2. mHC : forme + finitude (sublayer identité factice)
    cfg_hc = DeepSeekConfig(n_embd=32, hc_streams=4, sinkhorn_iters=10)
    hc = HyperConnections(cfg_hc)
    X = torch.randn(2, 6, 4, 32)
    out = hc(X, lambda h: h * 2.0)
    assert out.shape == X.shape, f"forme mHC inattendue : {out.shape}"
    assert torch.isfinite(out).all(), "mHC produit des NaN/Inf"
    print(f"HyperConnections OK — forme {tuple(out.shape)} conservée, sortie finie")

    # 3. CompressedAttention : formes HCA (dense) et CSA (top-k)
    cfg_a = DeepSeekConfig(n_embd=64, n_head=4, head_dim=16, d_compress=16,
                           hca_block=8, csa_block=4, csa_topk=3, index_dim=8, index_heads=2,
                           n_kv_heads_window=2, sliding_window=4, context_len=64)
    cos, sin = build_rope_cache(20, cfg_a.head_dim, cfg_a.rope_theta, "cpu", torch.float32)
    x = torch.randn(2, 20, cfg_a.n_embd)

    # masques normalement construits par DeepSeek.forward — reconstruits ici à la main
    T = 20
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    win_allowed = (i >= j) & (i - j < cfg_a.sliding_window)

    def _blk_causal(m: int) -> torch.Tensor:
        Tp = T + (m - T % m) % m
        return (torch.arange(m, Tp + 1, m) - 1)[None, :] <= i

    hca = CompressedAttention(cfg_a, block=cfg_a.hca_block, use_topk=False)
    y_hca = hca(x, cos, sin, win_allowed, _blk_causal(cfg_a.hca_block))
    assert y_hca.shape == x.shape, f"forme HCA inattendue : {y_hca.shape}"

    csa = CompressedAttention(cfg_a, block=cfg_a.csa_block, use_topk=True, topk=cfg_a.csa_topk)
    y_csa = csa(x, cos, sin, win_allowed, _blk_causal(cfg_a.csa_block))
    assert y_csa.shape == x.shape, f"forme CSA inattendue : {y_csa.shape}"
    print(f"CompressedAttention OK — HCA {tuple(y_hca.shape)}, CSA {tuple(y_csa.shape)}")

    # 4. Causalité : modifier un token futur ne doit PAS changer les logits passés
    #    (le point sensible : le pooling/compression ne doit jamais faire fuir le futur).
    cfg_m = DeepSeekConfig(vocab_size=128, n_embd=64, n_head=4, head_dim=16, n_layer=4,
                           d_ff=128, hc_streams=4, d_compress=16, csa_block=4, hca_block=8,
                           csa_topk=3, index_dim=8, index_heads=2, n_kv_heads_window=2,
                           sliding_window=4, csa_layers=(1, 3), context_len=64)
    model = DeepSeek(cfg_m)
    model.eval()
    idx = torch.randint(0, cfg_m.vocab_size, (2, 24))
    with torch.no_grad():
        logits_a = model(idx)
        idx2 = idx.clone()
        idx2[:, 15:] = torch.randint(0, cfg_m.vocab_size, idx2[:, 15:].shape)  # perturbe le futur
        logits_b = model(idx2)
    err = (logits_a[:, :15] - logits_b[:, :15]).abs().max().item()
    assert err < 1e-4, f"fuite de causalité détectée (err={err})"
    print(f"Causalité OK — perturber le futur ne change pas le passé (err max = {err:.2e})")

    # 5. forward + loss + backward + generate (contrat LanguageModel)
    model.train()
    loss = model.loss(idx[:, :-1], idx[:, 1:])
    assert torch.isfinite(loss), loss
    loss.backward()
    params = dict(model.named_parameters())
    for pname in ("tok_embd.weight", "blocks.1.hc_attn.static_B", "blocks.3.attn.k_comp.weight"):
        g = params[pname].grad
        assert g is not None and torch.isfinite(g).all(), f"gradient manquant/non-fini : {pname}"
    # l'indexeur "lightning" (CSA, couche 3) doit recevoir un gradient NON NUL — le top-k est
    # non-différentiable ; sans biaiser les logits par idx_score, iq/ik_proj/idx_w restent
    # figés à l'init (sélection aléatoire mais fixe pour tout l'entraînement). Régression gardée ici.
    for pname in ("blocks.3.attn.ik_proj.weight", "blocks.3.attn.idx_w"):
        g = params[pname].grad
        assert g is not None and g.abs().sum() > 0, f"indexeur CSA mort (gradient nul) : {pname}"
    # iq vit désormais dans in_proj (projections fusionnées) : ses lignes sont le dernier split
    attn3 = model.blocks[3].attn
    iq_grad = attn3.in_proj.weight.grad[-attn3.splits[-1]:]
    assert iq_grad.abs().sum() > 0, "indexeur CSA mort (gradient nul) : lignes iq de in_proj"
    print(f"forward/loss/backward OK — loss={loss.item():.3f}, indexeur CSA vivant")
    model.eval()
    out = model.generate(idx[:, :5], max_new_tokens=10)
    assert out.shape == (2, 15), out.shape
    print(f"generate OK {tuple(out.shape)}")

    # 6. Budget params à la config réelle — contrainte dure ≤126M
    full = DeepSeek(DeepSeekConfig())
    n_params = full.num_params()
    assert n_params <= 126_000_000, f"budget dépassé : {n_params/1e6:.1f}M > 126M"
    print(f"Budget params OK — {n_params/1e6:.1f}M ≤ 126M")
