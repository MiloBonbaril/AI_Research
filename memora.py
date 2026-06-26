"""
Memora — LLM "next-gen" à budget GPT-2 Small (~124-127M params).

Architecture hybride sub-quadratique destinée à battre GPT-2 Small :
  - Attention locale (fenêtre glissante) + GQA sur la majorité des couches
  - Couches GLA (Gated Linear Attention, récurrentes) intercalées → mémoire
    longue compressée dans un état de taille fixe (contexte non borné)
  - SwiGLU à la place de GELU-MLP
  - RMSNorm (pre-norm) à la place de LayerNorm
  - RoPE à la place des embeddings positionnels appris
  - Weight tying, QK-Norm, z-loss, aucun biais sur les projections de contenu

Suit le contrat de model_interface.LanguageModel (forward / loss / generate /
num_params / from_pretrained), comme gpt2.GPT2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_interface import BaseModelConfig, LanguageModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MemoraConfig(BaseModelConfig):
    """Hyperparamètres Memora. Hérite des champs de BaseModelConfig.

    Les champs de base (n_embd/n_layer/n_head/vocab_size/context_len) sont
    réutilisés tels quels par train.py et l'interface ; les alias `d_model`,
    `n_layers`, `n_heads` pointent dessus pour coller au document de spec.
    """
    vocab_size: int = 49152        # BPE byte-level moderne (FR/EN)
    context_len: int = 2048        # borne de troncature en génération (RoPE → pas de mur dur)
    n_layer: int = 14
    n_head: int = 12               # têtes de query
    n_embd: int = 768
    dropout: float = 0.0

    # --- GQA / dimensions attention ---
    n_kv_heads: int = 3            # têtes key/value (GQA) → cache KV /4
    head_dim: int = 64            # n_embd // n_head
    d_ff: int = 2048              # taille interne SwiGLU (~2.67x)

    # --- hybridation ---
    recurrent_layers: tuple = (3, 7, 10, 13)   # indices des couches GLA
    sliding_window: int = 512

    # --- position ---
    rope_theta: float = 10000.0

    # --- normalisation / stabilité ---
    norm_eps: float = 1e-5
    use_qk_norm: bool = True
    z_loss_weight: float = 1e-4
    logit_cap: float | None = None   # soft-capping style Gemma 2 (off par défaut)

    # --- GLA ---
    gla_low_rank: int = 16
    gla_decay_init_bias: float = 3.0  # biais d'init du gate → décroissance ~sigmoid(3)=0.95 (knob de calibration)

    tie_embeddings: bool = True
    bias: bool = False

    # alias lisibilité (read-only)
    @property
    def d_model(self) -> int: return self.n_embd
    @property
    def n_layers(self) -> int: return self.n_layer
    @property
    def n_heads(self) -> int: return self.n_head


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
    freqs = torch.outer(t, inv_freq)          # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)   # (T, hd)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd) ; cos/sin: (T, hd)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    """MLP SwiGLU : silu(W_gate x) * (W_up x) → W_down."""

    def __init__(self, c: MemoraConfig):
        super().__init__()
        self.w_gate = nn.Linear(c.n_embd, c.d_ff, bias=c.bias)
        self.w_up   = nn.Linear(c.n_embd, c.d_ff, bias=c.bias)
        self.w_down = nn.Linear(c.d_ff, c.n_embd, bias=c.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Mixers
# ---------------------------------------------------------------------------

class LocalAttention(nn.Module):
    """Attention causale à fenêtre glissante + GQA + RoPE + QK-Norm."""

    def __init__(self, c: MemoraConfig):
        super().__init__()
        self.n_head = c.n_head
        self.n_kv = c.n_kv_heads
        self.hd = c.head_dim
        self.window = c.sliding_window
        self.theta = c.rope_theta
        assert self.n_head % self.n_kv == 0, "n_head doit être multiple de n_kv_heads"

        self.q_proj = nn.Linear(c.n_embd, self.n_head * self.hd, bias=c.bias)
        self.k_proj = nn.Linear(c.n_embd, self.n_kv * self.hd, bias=c.bias)
        self.v_proj = nn.Linear(c.n_embd, self.n_kv * self.hd, bias=c.bias)
        self.o_proj = nn.Linear(self.n_head * self.hd, c.n_embd, bias=c.bias)

        self.use_qk_norm = c.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.hd, c.norm_eps)
            self.k_norm = RMSNorm(self.hd, c.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = build_rope_cache(T, self.hd, self.theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA : répéter K,V pour couvrir les têtes Q (4 têtes Q / tête KV)
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # Masque causal + fenêtre : on garde j tel que 0 <= i-j < window.
        # ponytail: masque dense (T,T) — O(T²) mémoire. Remplacer par
        # flash_attn window_size=(window-1,0) si le débit l'exige.
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(T, device=x.device)[None, :]
        allowed = (i >= j) & (i - j < self.window)
        attn_mask = torch.where(allowed, 0.0, float("-inf")).to(x.dtype)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.o_proj(y)


class GatedLinearAttention(nn.Module):
    """GLA : attention linéaire à gate de décroissance, état de taille fixe.

    Entraînement → forme chunkée parallèle (débit GPU).
    Référence/inférence → forme récurrente (état O(1) par pas, contexte illimité).
    Les deux formes sont numériquement équivalentes (cf. test __main__).
    """

    def __init__(self, c: MemoraConfig, chunk: int = 64):
        super().__init__()
        self.n_head = c.n_head
        self.hd = c.head_dim
        self.chunk = chunk
        dim = self.n_head * self.hd

        self.q_proj = nn.Linear(c.n_embd, dim, bias=c.bias)
        self.k_proj = nn.Linear(c.n_embd, dim, bias=c.bias)
        self.v_proj = nn.Linear(c.n_embd, dim, bias=c.bias)
        self.g_proj = nn.Linear(c.n_embd, dim, bias=c.bias)   # output gate
        self.o_proj = nn.Linear(dim, c.n_embd, bias=c.bias)

        # forget gate data-dependent, low-rank → sigmoïde
        self.a_low = nn.Linear(c.n_embd, c.gla_low_rank, bias=c.bias)
        self.a_high = nn.Linear(c.gla_low_rank, dim, bias=c.bias)
        self.a_bias = nn.Parameter(torch.full((dim,), c.gla_decay_init_bias))

        self.use_qk_norm = c.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.hd, c.norm_eps)
            self.k_norm = RMSNorm(self.hd, c.norm_eps)

    def _proj(self, x):
        B, T, _ = x.shape
        shp = (B, T, self.n_head, self.hd)
        q = self.q_proj(x).view(*shp).transpose(1, 2)   # (B,H,T,d)
        k = self.k_proj(x).view(*shp).transpose(1, 2)
        v = self.v_proj(x).view(*shp).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        a_logit = self.a_high(self.a_low(x)) + self.a_bias
        a = torch.sigmoid(a_logit).view(*shp).transpose(1, 2)   # décroissance ∈(0,1)
        g = torch.log(a)                                        # ≤ 0
        return q, k, v, g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, g = self._proj(x)
        o = self._chunked(q, k, v, g)
        o = o.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.n_head * self.hd)
        o = o * torch.sigmoid(self.g_proj(x))   # output gate
        return self.o_proj(o)

    def _chunked(self, q, k, v, g):
        """Forme chunkée. q,k,v,g: (B,H,T,d). Renvoie (B,H,T,d)."""
        B, H, T, d = q.shape
        C = self.chunk
        pad = (C - T % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad)); g = F.pad(g, (0, 0, 0, pad))  # g=0 → décroissance neutre
        Tp = T + pad
        nC = Tp // C

        # calcul en fp32 pour la stabilité des exp(±cumsum)
        q, k, v, g = (t.float() for t in (q, k, v, g))
        q = q.view(B, H, nC, C, d); k = k.view(B, H, nC, C, d)
        v = v.view(B, H, nC, C, d); g = g.view(B, H, nC, C, d)

        Gc = g.cumsum(dim=3)                       # (B,H,nC,C,d) cumul inclusif intra-chunk
        q_s = q * Gc.exp()                         # query mis à l'échelle
        k_s = k * (-Gc).exp()                      # key inverse-échelle

        # intra-chunk : A_{tj} = (q⊙e^Gc)·(k⊙e^-Gc), masqué t>=j
        A = torch.einsum("bhcid,bhcjd->bhcij", q_s, k_s)
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
        A = A.masked_fill(~causal, 0.0)
        o_intra = torch.einsum("bhcij,bhcjd->bhcid", A, v)

        # inter-chunk : porter l'état d'un chunk au suivant
        last = Gc[:, :, :, -1, :]                  # (B,H,nC,d) cumul total du chunk
        kbar = k * (last.unsqueeze(3) - Gc).exp()  # k_j ⊙ e^(Gc_last - Gc_j)
        # contribution de chaque chunk à l'état: kbar^T @ v  → (B,H,nC,d,d)
        S_chunk = torch.einsum("bhcjd,bhcje->bhcde", kbar, v)
        decay_chunk = last.exp()                   # (B,H,nC,d) décroissance globale du chunk

        # scan séquentiel sur les chunks (boucle courte: Tp/C itérations)
        o_inter = torch.empty_like(o_intra)
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        for c in range(nC):
            o_inter[:, :, c] = torch.einsum("bhid,bhde->bhie", q_s[:, :, c], S)
            S = decay_chunk[:, :, c].unsqueeze(-1) * S + S_chunk[:, :, c]

        o = (o_intra + o_inter).view(B, H, Tp, d)
        return o[:, :, :T].to(v.dtype if False else q.dtype)

    @torch.no_grad()
    def _recurrent(self, q, k, v, g):
        """Forme récurrente de référence (lente). Sert au test d'équivalence."""
        B, H, T, d = q.shape
        q, k, v, a = q.float(), k.float(), v.float(), g.float().exp()
        S = torch.zeros(B, H, d, d, device=q.device)
        out = torch.empty(B, H, T, d, device=q.device)
        for t in range(T):
            S = a[:, :, t].unsqueeze(-1) * S + torch.einsum("bhd,bhe->bhde", k[:, :, t], v[:, :, t])
            out[:, :, t] = torch.einsum("bhd,bhde->bhe", q[:, :, t], S)
        return out


# ---------------------------------------------------------------------------
# Bloc
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, c: MemoraConfig, recurrent: bool):
        super().__init__()
        self.norm_1 = RMSNorm(c.n_embd, c.norm_eps)
        self.mixer = GatedLinearAttention(c) if recurrent else LocalAttention(c)
        self.norm_2 = RMSNorm(c.n_embd, c.norm_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class Memora(LanguageModel):
    """LLM hybride local-attention + GLA, budget GPT-2 Small."""

    def __init__(self, config: MemoraConfig | None = None):
        super().__init__()
        self.config = config or MemoraConfig()
        c = self.config
        assert c.n_embd == c.n_head * c.head_dim, "n_embd doit == n_head * head_dim"

        self.tok_embd = nn.Embedding(c.vocab_size, c.n_embd)
        rec = set(c.recurrent_layers)
        self.blocks = nn.ModuleList([Block(c, i in rec) for i in range(c.n_layer)])
        self.norm_f = RMSNorm(c.n_embd, c.norm_eps)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_embd.weight   # même objet → params comptés une fois

        self.apply(self._init_weights)
        # scaling des projections de sortie résiduelles (1/√(2·n_layers))
        scale = 0.02 / math.sqrt(2 * c.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

        print(f"Memora initialisé — {self.num_params()/1e6:.1f}M paramètres "
              f"({c.n_layer} couches, GLA={c.recurrent_layers})")

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_embd(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if self.config.logit_cap is not None:
            cap = self.config.logit_cap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(idx)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if self.training and self.config.z_loss_weight > 0:
            # z-loss : pénalise logsumexp² → stabilise à gros vocab (ajoutée, pas substituée)
            lse = torch.logsumexp(logits, dim=-1)
            ce = ce + self.config.z_loss_weight * lse.pow(2).mean()
        return ce

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Memora":
        # Architecture novatrice : aucun poids pré-entraîné HF à mapper.
        raise NotImplementedError(
            "Memora est une architecture nouvelle, sans checkpoint HuggingFace. "
            "Entraîner depuis zéro via train.py."
        )


# ---------------------------------------------------------------------------
# Self-check : shapes + équivalence GLA chunké/récurrent + z-loss
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. équivalence chunké == récurrent (le test exigé par la spec)
    cfg = MemoraConfig(vocab_size=512, n_embd=128, n_head=4, head_dim=32,
                        n_layer=4, d_ff=256, recurrent_layers=(1,), context_len=256)
    gla = GatedLinearAttention(cfg, chunk=16).eval()
    x = torch.randn(2, 53, cfg.n_embd)  # T non multiple de chunk → teste le padding
    q, k, v, g = gla._proj(x)
    o_chunk = gla._chunked(q, k, v, g)
    o_rec = gla._recurrent(q, k, v, g)
    err = (o_chunk - o_rec).abs().max().item()
    assert err < 1e-4, f"GLA chunké != récurrent (err={err})"
    print(f"GLA chunké == récurrent (err max = {err:.2e})")

    # 2. forward + shapes + sliding window (T > window) + z-loss
    cfg2 = MemoraConfig(vocab_size=512, n_embd=128, n_head=4, n_kv_heads=2, head_dim=32,
                         n_layer=4, d_ff=256, recurrent_layers=(1, 3),
                         sliding_window=8, context_len=256)
    model = Memora(cfg2)
    idx = torch.randint(0, cfg2.vocab_size, (2, 40))
    logits = model(idx)
    assert logits.shape == (2, 40, cfg2.vocab_size), logits.shape
    model.train()
    loss = model.loss(idx[:, :-1], idx[:, 1:])
    assert torch.isfinite(loss), loss
    print(f"forward OK {tuple(logits.shape)} | loss={loss.item():.3f}")

    # 3. génération (vérifie la longueur de sortie)
    model.eval()
    out = model.generate(idx[:, :5], max_new_tokens=10)
    assert out.shape == (2, 15), out.shape
    print(f"generate OK {tuple(out.shape)}")

    # 4. budget params à la config réelle
    full = Memora(MemoraConfig())
    print(f"budget réel : {full.num_params()/1e6:.1f}M params")
