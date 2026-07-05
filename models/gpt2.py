"""
GPT-2 Small — implémentation de référence.

Architecture exacte du papier "Language Models are Unsupervised Multitask Learners"
(Radford et al., 2019), variante 124M paramètres.

Points clés du design :
  - Pre-norm (LayerNorm AVANT attention et FFN, pas après)
  - Poids des embeddings de tokens partagés avec la tête de sortie (weight tying)
  - Embeddings positionnels appris (pas sinusoïdaux)
  - Activation GELU (approximation tanh, comme l'original)
  - Biais dans toutes les couches linéaires et LayerNorms
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_interface import BaseModelConfig, LanguageModel


# ---------------------------------------------------------------------------
# Configuration GPT-2 Small
# ---------------------------------------------------------------------------

class GPT2Config(BaseModelConfig):
    """
    Hyperparamètres GPT-2 Small (124M).

    Hérite de BaseModelConfig. Les valeurs par défaut correspondent
    exactement à gpt2-small de OpenAI.
    """
    # ponytail: les valeurs par défaut de BaseModelConfig correspondent déjà
    # à GPT-2 Small (50257, 1024, 12, 12, 768) — rien à surcharger.
    pass


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

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Projections Q, K, V combinées en une seule matrice
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Projection de sortie
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

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

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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

    def __init__(self, config: GPT2Config):
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

class GPT2(LanguageModel):
    """
    GPT-2 Small (124M paramètres).

    Architecture :
        Token Embedding + Position Embedding
        → 12 × TransformerBlock (pre-norm)
        → LayerNorm finale
        → Tête linéaire (poids partagés avec Token Embedding)
    """

    def __init__(self, config: GPT2Config | None = None):
        super().__init__()
        self.config = config or GPT2Config()
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

        # Initialisation à la GPT-2
        self.apply(self._init_weights)
        # Scaling spécial des projections de sortie des couches résiduelles
        # (facteur 1/√(2·n_layer) sur c_proj)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

        print(f"GPT-2 Small initialisé — {self.num_params()/1e6:.1f}M paramètres")

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
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPT2":
        """
        Charge les poids pré-entraînés depuis HuggingFace.

        Usage :
            model = GPT2.from_pretrained("gpt2")

        Utilise le cache local si disponible, sinon télécharge.
        Les noms de paramètres HuggingFace sont mappés vers notre architecture.
        """
        from transformers import GPT2LMHeadModel

        # Essayer le cache local d'abord, sinon télécharger
        try:
            hf_model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)
            print(f"Poids '{model_name}' chargés depuis le cache local.")
        except OSError:
            print(f"Téléchargement des poids '{model_name}' depuis HuggingFace...")
            hf_model = GPT2LMHeadModel.from_pretrained(model_name)

        hf_sd = hf_model.state_dict()

        model = cls(GPT2Config())
        our_sd = model.state_dict()

        # Mapping HuggingFace → notre architecture
        # Les noms sont quasi identiques, on doit juste :
        #   1. Ignorer les buffers .attn.bias et .attn.masked_bias (masques causaux HF)
        #   2. Transposer les poids Conv1D de HF (qui stocke weight en (out, in)
        #      mais sous forme Conv1D c'est (in, out))
        keys_to_transpose = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        ]

        # Buffers de masque causal HF — à ignorer (on utilise is_causal=True)
        # ATTENTION : on utilise endswith, pas "in", sinon "attn.bias" matcherait
        # aussi "attn.c_attn.bias" et "attn.c_proj.bias" !
        skip_suffixes = (".attn.bias", ".attn.masked_bias")

        for key in hf_sd:
            if key.endswith(skip_suffixes):
                continue

            # HuggingFace nomme les blocs h.0, h.1, ... → nos blocks.0, blocks.1, ...
            our_key = key
            our_key = our_key.replace("transformer.h.", "blocks.")
            our_key = our_key.replace("transformer.wte.", "tok_embd.")
            our_key = our_key.replace("transformer.wpe.", "pos_embd.")
            our_key = our_key.replace("transformer.ln_f.", "ln_f.")

            if our_key not in our_sd:
                # lm_head.weight est tied, pas dans state_dict
                continue

            # Conv1D de HF stocke les poids transposés par rapport à nn.Linear
            if any(our_key.endswith(suffix) for suffix in keys_to_transpose):
                assert hf_sd[key].shape[::-1] == our_sd[our_key].shape
                with torch.no_grad():
                    our_sd[our_key].copy_(hf_sd[key].t())
            else:
                assert hf_sd[key].shape == our_sd[our_key].shape
                with torch.no_grad():
                    our_sd[our_key].copy_(hf_sd[key])

        model.load_state_dict(our_sd)
        print("Poids chargés avec succès.")
        return model
