"""
Effy — hybride attention pleine / attention linéaire causale "d'économie".

Reprend l'architecture de models/gpt2.py à l'identique (pre-norm, GELU-tanh,
weight tying, embeddings positionnels appris, biais partout, FFN 4×) à une
seule différence : la plupart des blocs gardent l'attention PLEINE O(T²) de
GPT-2, et un bloc sur 4 bascule sur une attention LINÉAIRE causale (noyau
φ(q)·φ(k), coût O(T), état de taille fixe) pour amortir le coût quadratique —
ratio 3 couches pleines pour 1 couche linéaire (même principe que le mélange
GDN/SWA de Cortex, cf. models/cortex.md, mais inversé : ici la majorité reste
l'attention exacte).

Les deux types de couches partagent exactement les mêmes formes de
projections (c_attn: n_embd → 3·n_embd, c_proj: n_embd → n_embd) : seul le
calcul d'attention diffère, donc le nombre de paramètres d'Effy est identique
à GPT-2 (mêmes embeddings, même MLP, même nombre de couches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_interface import BaseModelConfig, LanguageModel


# ---------------------------------------------------------------------------
# Configuration Effy Small
# ---------------------------------------------------------------------------

@dataclass
class EffyConfig(BaseModelConfig):
    """
    Hyperparamètres Effy Small (~124M).

    Hérite de BaseModelConfig (mêmes valeurs par défaut que GPT-2 Small :
    50257/1024/12/12/768). `linear_layers` fixe les indices de blocs qui
    basculent en attention linéaire ; tous les autres blocs gardent
    l'attention pleine. Défaut = 3 couches linéaires sur 12 → ratio 3 pleines
    pour 1 linéaire, réparties tous les 4 blocs.
    """
    linear_layers: tuple = (3, 7, 11)


# ---------------------------------------------------------------------------
# Blocs élémentaires
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention pleine — identique à gpt2.CausalSelfAttention.

    Type par défaut (3 blocs sur 4) : recall exact que le noyau linéaire de
    LinearAttention ne peut pas offrir, sauf sur les `linear_layers`.
    """

    def __init__(self, config: EffyConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class LinearAttention(nn.Module):
    """
    Multi-head causal LINEAR attention (Katharopoulos et al., 2020).

    Mêmes projections combinées Q/K/V que CausalSelfAttention, mais noyau
    φ(q)·φ(k) (φ = elu+1, positif) au lieu de softmax(QK^T)/√d — coût O(T) au
    lieu de O(T²), état de taille fixe (head_dim × head_dim) reporté de chunk
    en chunk. Forme chunkée parallèle : même principe que
    GatedLinearAttention._chunked dans models/memora.py, mais sans gate de
    décroissance (état simplement accumulé, jamais décayé).
    """

    def __init__(self, config: EffyConfig, chunk: int = 64):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.chunk = chunk

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q, k = self._phi(q), self._phi(k)

        y = self._chunked(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def _chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forme chunkée : coût O(T·chunk), mémoire O(T) (pas de tenseur T×T).

        q,k,v: (B,H,T,d), q/k déjà passés par φ. État (S,Z) reporté de chunk
        en chunk — pas de décroissance, contrairement à GLA.
        """
        B, H, T, d = q.shape
        C = self.chunk
        # fp32 pour l'accumulation de S/Z (état cumulé sur tout T) : sous autocast
        # bf16 l'entrée arrive en bf16, mais une somme de ~T produits externes y perdrait
        # les bits de poids faible une fois S grand (cf. memora.py:GatedLinearAttention._chunked)
        in_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        pad = (C - T % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Tp = T + pad
        nC = Tp // C
        q = q.view(B, H, nC, C, d)
        k = k.view(B, H, nC, C, d)
        v = v.view(B, H, nC, C, d)

        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

        o = torch.empty(B, H, nC, C, d, device=q.device, dtype=q.dtype)
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        Z = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)
        for c in range(nC):
            qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]

            A = torch.einsum("bhid,bhjd->bhij", qc, kc).masked_fill(~causal, 0.0)
            o_intra = torch.einsum("bhij,bhjd->bhid", A, vc)
            z_intra = A.sum(dim=-1)

            o_inter = torch.einsum("bhid,bhde->bhie", qc, S)
            z_inter = torch.einsum("bhid,bhd->bhi", qc, Z)

            denom = (z_intra + z_inter).clamp(min=1e-6).unsqueeze(-1)
            o[:, :, c] = (o_intra + o_inter) / denom

            S = S + torch.einsum("bhjd,bhje->bhde", kc, vc)
            Z = Z + kc.sum(dim=2)

        return o.view(B, H, Tp, d)[:, :, :T].to(in_dtype)


class FeedForward(nn.Module):
    """FFN à deux couches avec GELU — identique à gpt2.FeedForward."""

    def __init__(self, config: EffyConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer pre-norm — structure identique à gpt2.TransformerBlock.

    `full_attn=True` (défaut, 3 couches sur 4) → CausalSelfAttention (pleine).
    `full_attn=False` (`linear_layers`) → LinearAttention.
    """

    def __init__(self, config: EffyConfig, full_attn: bool):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) if full_attn else LinearAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------

class Effy(LanguageModel):
    """
    Effy Small (~124M paramètres).

    Architecture : identique à GPT-2 Small — 3 blocs sur 4 gardent l'attention
    pleine softmax exacte ; les blocs `linear_layers` basculent sur une
    attention linéaire causale (état de taille fixe) pour amortir le coût
    quadratique.
    """

    def __init__(self, config: EffyConfig | None = None):
        super().__init__()
        self.config = config or EffyConfig()
        c = self.config

        # -- Embeddings -----------------------------------------------------
        self.tok_embd = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_embd = nn.Embedding(c.context_len, c.n_embd)

        self.drop = nn.Dropout(c.dropout)

        # -- Transformer ----------------------------------------------------
        linear = set(c.linear_layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(c, full_attn=(i not in linear)) for i in range(c.n_layer)
        ])

        # -- Sortie ---------------------------------------------------------
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

        # Weight tying : la tête de sortie partage les poids des embeddings
        self.lm_head.weight = self.tok_embd.weight

        # Initialisation à la GPT-2
        self.apply(self._init_weights)
        # Scaling spécial des projections de sortie des couches résiduelles
        # (facteur 1/√(2·n_layer) sur c_proj)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

        print(f"Effy Small initialisé — {self.num_params()/1e6:.1f}M paramètres "
              f"({c.n_layer} couches, linéaires={sorted(linear)})")

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initialisation des poids selon le papier GPT-2."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Passe avant.

        Args:
            idx: (B, T) indices de tokens, avec T ≤ context_len.

        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = idx.size()
        assert T <= self.config.context_len, (
            f"Séquence trop longue ({T} > {self.config.context_len})"
        )

        pos = torch.arange(T, device=idx.device)

        x = self.tok_embd(idx) + self.pos_embd(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Effy":
        # Architecture hybride nouvelle (attention pleine + linéaire) : aucun
        # poids pré-entraîné HuggingFace ne correspond à ce mélange de couches.
        raise NotImplementedError(
            "Effy est une architecture hybride nouvelle, sans checkpoint HuggingFace. "
            "Entraîner depuis zéro via train.py."
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. LinearAttention chunkée == forme naïve O(T²) (référence du noyau causal linéaire)
    cfg = EffyConfig(vocab_size=64, n_embd=32, n_head=4, n_layer=4, context_len=64,
                      linear_layers=(1,))
    la = LinearAttention(cfg, chunk=8).eval()
    x = torch.randn(2, 23, cfg.n_embd)  # T non multiple de chunk → teste le padding
    with torch.no_grad():
        qkv = la.c_attn(x)
        q, k, v = qkv.split(cfg.n_embd, dim=2)
        head_dim = cfg.n_embd // cfg.n_head
        q = q.view(2, 23, cfg.n_head, head_dim).transpose(1, 2)
        k = k.view(2, 23, cfg.n_head, head_dim).transpose(1, 2)
        v = v.view(2, 23, cfg.n_head, head_dim).transpose(1, 2)
        q, k = la._phi(q), la._phi(k)

        o_chunked = la._chunked(q, k, v)

        T = q.shape[2]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        A = torch.einsum("bhid,bhjd->bhij", q, k).masked_fill(~causal, 0.0)
        o_naive = torch.einsum("bhij,bhjd->bhid", A, v) / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    err = (o_chunked - o_naive).abs().max().item()
    assert err < 1e-4, f"LinearAttention chunkée != naïve (err={err})"
    print(f"LinearAttention chunkée == naïve (err max = {err:.2e})")

    # 2. forward + shapes + loss finie, config hybride 3 pleines / 1 linéaire (6/8, 2/8)
    cfg2 = EffyConfig(vocab_size=64, n_embd=32, n_head=4, n_layer=8, context_len=64,
                       linear_layers=(3, 7))
    model = Effy(cfg2)
    idx = torch.randint(0, cfg2.vocab_size, (2, 20))
    logits = model(idx)
    assert logits.shape == (2, 20, cfg2.vocab_size), logits.shape
    model.train()
    loss = model.loss(idx[:, :-1], idx[:, 1:])
    assert torch.isfinite(loss), loss
    print(f"forward OK {tuple(logits.shape)} | loss={loss.item():.3f}")

    # 3. génération (vérifie la longueur de sortie)
    model.eval()
    out = model.generate(idx[:, :5], max_new_tokens=10)
    assert out.shape == (2, 15), out.shape
    print(f"generate OK {tuple(out.shape)}")

    # 4. budget params à la config réelle — comparé à GPT-2 Small (même formes de
    # projections partout → doit être identique, pas juste "quasiment")
    from models.gpt2 import GPT2, GPT2Config
    full = Effy(EffyConfig())
    ref = GPT2(GPT2Config())
    print(f"Effy  : {full.num_params()/1e6:.1f}M params")
    print(f"GPT-2 : {ref.num_params()/1e6:.1f}M params")
    assert full.num_params() == ref.num_params(), (
        f"Effy ({full.num_params()}) devrait avoir exactement le même nombre de "
        f"params que GPT-2 ({ref.num_params()})"
    )
