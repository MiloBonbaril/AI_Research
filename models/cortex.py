"""
Cortex - inspiré du cerveau humain.



Points clés du design :
  - Poids ternaires (BitNet b1.58)
  - Attention hybride (GDN + SWA)
  - Sparsité d'activation (ReLU²)
  - Profondeur adaptative (MoR)
  - Mixture of Experts (MoE) (phase tardive)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_interface import BaseModelConfig, LanguageModel


# ---------------------------------------------------------------------------
# BitLinear — poids ternaires {-1, 0, +1} + activations int8 (BitNet b1.58)
# ---------------------------------------------------------------------------

class BitLinear(nn.Linear):
    """
    Remplaçant de nn.Linear avec poids ternaires et activations int8.

    Forward :
      W_q = clamp(round(W / (mean(|W|) + ε)), -1, +1)   ← ternaire
      x_q = round(clamp(x · 127 / max(|x|), -128, 127)) ← int8 par token
      y   = (W_q · x_q) · (γ · scale_x / 127)

    Backward : Straight-Through Estimator — le gradient traverse la
    quantification comme si elle était l'identité.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight

        # --- Quantification des poids (absmean → ternaire) ---
        gamma = W.abs().mean()
        W_q = (W / (gamma + 1e-8)).clamp(-1, 1).round()
        W_q = W + (W_q - W).detach()  # STE : gradient = identité

        # --- Quantification des activations (absmax par token → int8) ---
        scale_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_q = (x * 127.0 / scale_x).clamp(-128, 127).round()
        x_q = x + (x_q - x).detach()  # STE

        # --- Produit + déquantification ---
        y = F.linear(x_q, W_q, self.bias)
        return y * (gamma * scale_x / 127.0)


# ---------------------------------------------------------------------------
# Configuration Cortex - Small
# ---------------------------------------------------------------------------

class CortexConfig(BaseModelConfig):
    """
    Hyperparamètres Cortex (124M).

    Hérite de BaseModelConfig.
    """
    vocab_size: int = 50257
    context_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # 0 par défaut (pas de dropout en inférence)


# ---------------------------------------------------------------------------
# Blocs élémentaires
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Une seule projection linéaire produit Q, K, V simultanément,
    puis on utilise scaled_dot_product_attention de PyTorch (FlashAttention
    quand disponible).
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = BitLinear(config.n_embd, 3 * config.n_embd)
        self.c_proj = BitLinear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, séquence, embedding dim
        head_dim = C // self.n_head

        # Projection Q, K, V
        qkv = self.c_attn(x)                         # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # 3 × (B, T, C)

        # Reshape pour multi-head : (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention causale (masque triangulaire automatique, FlashAttention si dispo)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, n_head, T, head_dim)

        # Recombiner les têtes
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class FeedForward(nn.Module):
    """
    FFN à deux couches avec GELU.

    Expansion 4× standard : n_embd → 4·n_embd → n_embd.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.c_fc   = BitLinear(config.n_embd, 4 * config.n_embd)
        self.c_proj = BitLinear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ponytail: GELU approx tanh — identique à l'implémentation OpenAI originale
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer pre-norm.

    Structure :
        x → LayerNorm → Attention → + résiduel
          → LayerNorm → FFN       → + résiduel
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
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

class Cortex(LanguageModel):
    """
    Cortex (124M paramètres).

    Architecture :
        Token Embedding + Position Embedding
        → 12 × TransformerBlock (pre-norm)
        → LayerNorm finale
        → Tête linéaire (poids partagés avec Token Embedding)
    """

    def __init__(self, config: CortexConfig | None = None):
        super().__init__()
        self.config = config or CortexConfig()
        c = self.config

        # -- Embeddings -----------------------------------------------------
        self.tok_embd = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_embd = nn.Embedding(c.context_len, c.n_embd)

        self.drop = nn.Dropout(c.dropout)

        # -- Transformer ----------------------------------------------------
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layer)])

        # -- Sortie ---------------------------------------------------------
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

        # Weight tying : la tête de sortie partage les poids des embeddings
        self.lm_head.weight = self.tok_embd.weight

        # Initialisation à la cortex
        self.apply(self._init_weights)
        # Scaling spécial des projections de sortie des couches résiduelles
        # (facteur 1/√(2·n_layer) sur c_proj)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

        print(f"Cortex Small initialisé — {self.num_params()/1e6:.1f}M paramètres")

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initialisation des poids Cortex."""
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

        # Positions : 0, 1, 2, ..., T-1
        pos = torch.arange(T, device=idx.device)

        # Embeddings = token + position
        x = self.tok_embd(idx) + self.pos_embd(pos)  # (B, T, C)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Projection vers le vocabulaire
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    @classmethod
    def from_pretrained(cls, model_name: str = "cortex") -> "Cortex":
        raise NotImplementedError("Cortex est une architecture originale — entraîner from scratch.")


if __name__ == "__main__":
    # Vérification BitLinear : forme, backward, poids ternaires
    bl = BitLinear(64, 32)
    x = torch.randn(2, 8, 64)
    y = bl(x)
    assert y.shape == (2, 8, 32), f"forme inattendue : {y.shape}"
    y.sum().backward()
    assert bl.weight.grad is not None, "pas de gradient sur les poids"
    with torch.no_grad():
        gamma = bl.weight.abs().mean()
        W_q = (bl.weight / (gamma + 1e-8)).clamp(-1, 1).round()
        assert set(W_q.unique().tolist()).issubset({-1.0, 0.0, 1.0}), "poids non ternaires"
    print("BitLinear OK")

    # Vérification Cortex : forme des logits
    cfg = CortexConfig(n_layer=2, context_len=16)
    model = Cortex(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    logits = model(idx)
    assert logits.shape == (1, 16, cfg.vocab_size)
    print(f"Cortex OK — {model.num_params()/1e6:.1f}M paramètres")
