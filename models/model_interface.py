"""
Interface abstraite pour modèles de langage autorégressifs.

Tout modèle (GPT-2 référence, ou tes propres variantes) implémente cette
interface pour pouvoir être entraîné, évalué et utilisé de manière uniforme.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Configuration de base — à sous-classer pour ajouter des champs spécifiques
# ---------------------------------------------------------------------------

@dataclass
class BaseModelConfig:
    """Hyperparamètres partagés par tout modèle autorégressif."""
    vocab_size: int = 50257
    context_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # 0 par défaut (pas de dropout en inférence)


# ---------------------------------------------------------------------------
# Interface modèle
# ---------------------------------------------------------------------------

class LanguageModel(ABC, nn.Module):
    """
    Contrat minimal qu'un modèle de langage autorégressif doit satisfaire.

    Méthodes à implémenter :
        forward(idx)          → logits (B, T, vocab_size)
        from_pretrained(name) → instance avec poids HuggingFace (classmethod)

    Méthodes fournies :
        loss(idx, targets)    → cross-entropy loss
        generate(idx, ...)    → génération autoregresssive
        num_params()          → nombre de paramètres (hors embeddings position.)
    """

    config: BaseModelConfig

    @abstractmethod
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Passe avant.

        Args:
            idx: (B, T) tensor d'indices de tokens, T ≤ context_len.

        Returns:
            logits: (B, T, vocab_size) scores bruts avant softmax.
        """
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name: str) -> "LanguageModel":
        """Charge les poids depuis HuggingFace pour comparaison."""
        ...

    # -- Méthodes concrètes (partagées) ------------------------------------

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy sur les logits, par chunks re-calculés au backward.

        Args:
            idx:     (B, T) tokens d'entrée.
            targets: (B, T) tokens cibles (décalés de 1 en amont par l'appelant).

        Returns:
            Scalaire, la loss moyenne.

        Sous autocast, cross_entropy promeut les logits en fp32 : en un seul appel sur
        (B·T, vocab) ça matérialise ~1.2 Go de fp32 (copie + log_softmax sauvegardé pour
        le backward) à B·T=3072, vocab=50257. Découper en chunks passés par checkpoint
        borne ce pic à ~2×chunk×vocab×4 octets, au prix d'un recalcul de la CE au
        backward (négligeable devant le forward). Même loss (somme/n == moyenne),
        mêmes gradients.
        """
        logits = self.forward(idx)  # (B, T, V)
        flat = logits.reshape(-1, logits.size(-1))
        tgt = targets.reshape(-1)
        n = flat.size(0)
        chunk = 1024
        if not torch.is_grad_enabled() or n <= chunk:
            return F.cross_entropy(flat, tgt)

        def _ce_sum(l: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(l, t, reduction="sum")

        parts = [checkpoint(_ce_sum, flat[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
                 for s in range(0, n, chunk)]
        return torch.stack(parts).sum() / n

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Génération autoregresssive token par token.

        Args:
            idx:            (B, T) contexte initial.
            max_new_tokens: nombre de tokens à générer.
            temperature:    > 1 = plus aléatoire, < 1 = plus déterministe.
            top_k:          si défini, filtre aux top_k logits avant sampling.

        Returns:
            (B, T + max_new_tokens) séquence complète.
        """
        for _ in range(max_new_tokens):
            # Tronquer au context_len si nécessaire
            idx_cond = idx[:, -self.config.context_len:]

            logits = self.forward(idx_cond)       # (B, T', V)
            logits = logits[:, -1, :] / temperature  # (B, V) — dernier step

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_params(self, exclude_pos_embd: bool = True) -> int:
        """Nombre total de paramètres (hors embeddings positionnels par défaut)."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_pos_embd and hasattr(self, "pos_embd"):
            n -= self.pos_embd.weight.numel()
        return n


# ---------------------------------------------------------------------------
# Self-check : la CE par chunks == CE directe (valeur ET gradients)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    class _Tiny(LanguageModel):
        def __init__(self):
            super().__init__()
            self.config = BaseModelConfig(vocab_size=257, context_len=512)
            self.embd = nn.Embedding(257, 16)
            self.head = nn.Linear(16, 257)

        def forward(self, idx):
            return self.head(self.embd(idx))

        @classmethod
        def from_pretrained(cls, model_name):
            raise NotImplementedError

    m = _Tiny()
    x = torch.randint(0, 257, (8, 300))   # B·T = 2400 > chunk → chemin chunké exercé
    y = torch.randint(0, 257, (8, 300))

    loss_c = m.loss(x, y)
    loss_ref = F.cross_entropy(m(x).reshape(-1, 257), y.reshape(-1))
    assert torch.allclose(loss_c, loss_ref, atol=1e-6), (loss_c.item(), loss_ref.item())

    loss_c.backward()
    g_c = m.head.weight.grad.clone()
    m.zero_grad()
    loss_ref.backward()
    g_r = m.head.weight.grad
    assert torch.allclose(g_c, g_r, atol=1e-6), "gradients chunkés ≠ gradients directs"
    print(f"loss chunkée OK — valeur ({loss_c.item():.4f}) et gradients identiques à la CE directe")
