"""
MemoraGLA — LLM "next-gen" à budget GPT-2 Small (~124-127M params).
Variante GLA-dominante: 10 couches GLA, 2 globales, 2 locales

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
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from model_interface import BaseModelConfig, LanguageModel

# flex_attention DOIT être compilé pour atteindre le débit d'un kernel flash : en eager
# il est ~40× plus lent que SDPA. On le compile ici une fois pour que les couches locales
# soient rapides même si le modèle global n'est pas (ou est mal) compilé (graph break sur
# la création du block_mask). torch.compile est paresseux → aucun coût à l'import.
_flex_attention = torch.compile(flex_attention)

# Kernel GLA fusé (Triton) — accélère les couches récurrentes sur GPU.
# Absent (CPU, pas de CUDA, ou non installé) → repli sur _chunked en pur torch,
# qui reste la référence testée contre _recurrent dans __main__.
try:
    from fla.ops.gla import chunk_gla as _fla_chunk_gla
except Exception:  # pragma: no cover - dépend de l'install/GPU
    _fla_chunk_gla = None


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
    head_dim: int = 64            # n_embd // n_head (couches d'attention)
    d_ff: int = 1850              # taille interne SwiGLU. Réduit de 2000 → 1850 pour compenser
                                  # le surcoût params des 6 couches GLA ajoutées (10 total vs 4) ;
                                  # reste ~126.5M (budget GPT-2). GLA coûte plus que l'attention
                                  # locale (5 proj. full-width vs 2+2 GQA), d'où la correction MLP.

    # --- hybridation ---
    # Architecture Qwen-style GLA-dominante : 10 GLA / 2 global / 2 local.
    # local = (3, 7) — les seules couches avec fenêtre glissante.
    recurrent_layers: tuple = (0, 1, 2, 4, 6, 8, 9, 11, 12, 13)   # 10 couches GLA
    global_layers: tuple = (5, 10)             # 2 couches d'attention GLOBALE (softmax dense
                                               # causale) — rétablissent le long-range que la
                                               # fenêtre 512 ampute. Free en params (mêmes
                                               # projections). Le reste = local (3, 7).
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
    gla_head_dim: int = 48            # head_dim PROPRE aux couches GLA (< head_dim attention=64).
                                      # Réduit le coût des 5 projections GLA. fla impose q/k/v/g de
                                      # même forme → on garde n_head têtes.
                                      # Variante ghd=64 d_ff=1600 → ~125.9M (+ capacité état GLA,
                                      # - capacité MLP). Passer à MemoraConfig(gla_head_dim=64,d_ff=1600).
    gla_use_rope: bool = False        # signal positionnel RoPE sur GLA (axe 11). Off par défaut :
                                      # la position vit dans l'état ; à activer comme expérience.
    gla_decay_init_bias: float = 3.0  # biais d'init du gate → décroissance ~sigmoid(3)=0.95 (knob de calibration)

    tie_embeddings: bool = True
    bias: bool = False

    # --- mémoire / débit ---
    grad_checkpoint: bool = False  # checkpoint des blocs GLA (recompute au backward) si VRAM-bound

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
    """Attention causale + GQA + RoPE + QK-Norm.

    window = int  → fenêtre glissante (couche locale, O(T·W)).
    window = None → causale pleine (couche GLOBALE, O(T²) mais flash → coût modéré).
    Projections identiques dans les deux cas → une couche globale est gratuite en params.
    """

    def __init__(self, c: MemoraConfig, window: int | None):
        super().__init__()
        self.n_head = c.n_head
        self.n_kv = c.n_kv_heads
        self.hd = c.head_dim
        self.window = window
        assert self.n_head % self.n_kv == 0, "n_head doit être multiple de n_kv_heads"

        self.q_proj = nn.Linear(c.n_embd, self.n_head * self.hd, bias=c.bias)
        self.k_proj = nn.Linear(c.n_embd, self.n_kv * self.hd, bias=c.bias)
        self.v_proj = nn.Linear(c.n_embd, self.n_kv * self.hd, bias=c.bias)
        self.o_proj = nn.Linear(self.n_head * self.hd, c.n_embd, bias=c.bias)

        self.use_qk_norm = c.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.hd, c.norm_eps)
            self.k_norm = RMSNorm(self.hd, c.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                block_masks: dict) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # cos/sin précalculés au niveau modèle (cf. Memora._rope), pas de recompute ici
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if x.is_cuda:
            # flex_attention : block_mask (local fenêtré OU global causal) caché par Memora,
            # + GQA native (enable_gqa). Kernel fusé sous torch.compile.
            y = _flex_attention(q, k, v, block_mask=block_masks[self.window], enable_gqa=True)
        else:
            # repli CPU (self-test) : flex_attention ne supporte pas le backward sur CPU.
            # SDPA + enable_gqa garde la GQA native. window=None → causal pleine.
            i = torch.arange(T, device=x.device)[:, None]
            j = torch.arange(T, device=x.device)[None, :]
            allowed = (i >= j) if self.window is None else (i >= j) & (i - j < self.window)
            attn_mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.o_proj(y)


class GatedLinearAttention(nn.Module):
    """GLA : attention linéaire à gate de décroissance, état de taille fixe.

    Entraînement → forme chunkée parallèle (débit GPU).
    Référence/inférence → forme récurrente (état O(1) par pas, contexte illimité).
    Les deux formes sont numériquement équivalentes (cf. test __main__).
    """

    def __init__(self, c: MemoraConfig, chunk: int = 16):
        # _chunked est désormais inconditionnellement stable (intra-chunk via la
        # différence Gc_i-Gc_j bornée ≤0, jamais e^{-Gc} absolu). Le plafond du chunk
        # n'est donc plus l'overflow mais la MÉMOIRE : l'intra matérialise un tenseur
        # (C,C,d) par chunk (décroissance par canal). chunk=16 = bon compromis ; on peut
        # monter (32/64) pour le débit si la VRAM suit.
        # ponytail: pour throughput max sans le tenseur (C,C,d), passer à fla (Triton).
        super().__init__()
        self.n_head = c.n_head
        self.hd = c.gla_head_dim       # head_dim propre à GLA (≤ head_dim attention) — axe 10
        self.use_rope = c.gla_use_rope
        self.rope_theta = c.rope_theta
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
        # log(décroissance) via logsigmoid (stable même si a_logit << 0, vs log(sigmoid)).
        # clamp(min=-10) : plancher de rétention e^-10≈4.5e-5/pas (rien d'utile en-dessous),
        # ceinture+bretelles qui borne aussi la forme naïve. ≤ 0 garanti.
        g = F.logsigmoid(a_logit).clamp(min=-10.0).view(*shp).transpose(1, 2)
        return q, k, v, g

    def forward(self, x: torch.Tensor, cos=None, sin=None, block_masks=None) -> torch.Tensor:
        # block_masks ignoré : pas de masque sur GLA. cos/sin (taille head_dim attention)
        # ignorés aussi : si gla_use_rope, on construit un RoPE à la dimension gla_head_dim.
        q, k, v, g = self._proj(x)            # (B,H,T,d)
        if self.use_rope:                     # signal positionnel optionnel (axe 11)
            rc, rs = build_rope_cache(x.shape[1], self.hd, self.rope_theta, x.device, q.dtype)
            q, k = apply_rope(q, rc, rs), apply_rope(k, rc, rs)
        if _fla_chunk_gla is not None and x.is_cuda:
            # Kernel Triton fusé : layout (B,T,H,d) ; scale=1.0 car q n'est pas pré-scalé
            # (notre _chunked/_recurrent non plus). Équivalent à _chunked à la précision
            # tf32 du kernel près (cf. rapport).
            # Le kernel Triton exige un dtype homogène. Sous autocast, QK-Norm repromeut q,k
            # en fp32 et g sort en fp32, alors que v reste en bf16 → on réaligne tout sur v
            # (dtype de calcul réel : bf16 sous autocast, fp32 sinon).
            dt = v.dtype
            # permute+to fusionné : une seule allocation au lieu de deux (contiguous puis cast)
            qf, kf, vf, gf = (t.permute(0, 2, 1, 3).to(dtype=dt, memory_format=torch.contiguous_format) for t in (q, k, v, g))
            o, _ = _fla_chunk_gla(qf, kf, vf, gf, scale=1.0)   # (B,T,H,d)
            o = o.reshape(x.shape[0], x.shape[1], self.n_head * self.hd)
        else:
            o = self._chunked(q, k, v, g)     # repli pur torch (CPU / pas de fla)
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

        Gc = g.cumsum(dim=3)                       # (B,H,nC,C,d) cumul inclusif, ≤0, décroissant
        q_s = q * Gc.exp()                         # query ⊙ e^Gc — BORNÉ (Gc≤0 → e^Gc≤1)

        # inter-chunk : porter l'état d'un chunk au suivant (déjà stable : last-Gc ≤ 0)
        last = Gc[:, :, :, -1, :]                  # (B,H,nC,d) cumul total du chunk
        kbar = k * (last.unsqueeze(3) - Gc).exp()  # k_j ⊙ e^(Gc_last - Gc_j), exposant ≤ 0
        S_chunk = torch.einsum("bhcjd,bhcje->bhcde", kbar, v)   # (B,H,nC,d,d)
        decay_chunk = last.exp()                   # (B,H,nC,d) décroissance globale du chunk

        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

        # scan séquentiel sur les chunks (boucle courte: Tp/C itérations)
        o = torch.empty(B, H, nC, C, d, device=q.device, dtype=q.dtype)
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        for c in range(nC):
            Gc_c = Gc[:, :, c]                     # (B,H,C,d)
            # intra-chunk STABLE : on n'écrit jamais e^{-Gc} en absolu (qui overflow),
            # mais la différence Gc_i - Gc_j directement. Comme Gc décroît et qu'on masque
            # i≥j, l'exposant est ≤ 0 → e^(…) ∈ (0,1], jamais d'overflow quelle que soit
            # la force du gate. Décroissance par canal → tenseur (C,C,d) transitoire (libéré
            # à chaque itération ; c'est lui qui borne la taille de chunk, plus l'overflow).
            diff = Gc_c.unsqueeze(3) - Gc_c.unsqueeze(2)        # (B,H,C,C,d) = Gc_i - Gc_j
            D = diff.clamp(max=0.0).exp().masked_fill(~causal[:, :, None], 0.0)
            A = torch.einsum("bhid,bhjd,bhijd->bhij", q[:, :, c], k[:, :, c], D)
            o_intra = torch.einsum("bhij,bhjd->bhid", A, v[:, :, c])
            # inter-chunk : contribution de l'état porté (q_s borné)
            o_inter = torch.einsum("bhid,bhde->bhie", q_s[:, :, c], S)
            o[:, :, c] = o_intra + o_inter
            S = decay_chunk[:, :, c].unsqueeze(-1) * S + S_chunk[:, :, c]

        return o.view(B, H, Tp, d)[:, :, :T].to(q.dtype)

    @torch.no_grad()
    def _recurrent(self, q, k, v, g):
        """Forme récurrente de référence (lente). Sert au test d'équivalence.

        Limite connue : sans état persistant entre appels → inutilisable tel quel
        pour de l'inférence streaming réelle. Une variante stateful (qui garde S
        d'un appel au suivant) serait nécessaire pour exploiter le contexte illimité.
        """
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
    def __init__(self, c: MemoraConfig, kind: str):
        super().__init__()
        assert kind in ("gla", "local", "global")
        self.recurrent = kind == "gla"
        self.norm_1 = RMSNorm(c.n_embd, c.norm_eps)
        if kind == "gla":
            self.mixer = GatedLinearAttention(c)
        elif kind == "global":
            self.mixer = LocalAttention(c, window=None)        # causal pleine
        else:
            self.mixer = LocalAttention(c, window=c.sliding_window)
        self.norm_2 = RMSNorm(c.n_embd, c.norm_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                block_masks: dict) -> torch.Tensor:
        x = x + self.mixer(self.norm_1(x), cos, sin, block_masks)
        x = x + self.mlp(self.norm_2(x))
        return x


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class MemoraGLA(LanguageModel):
    """LLM hybride local-attention + GLA, budget GPT-2 Small."""

    def __init__(self, config: MemoraConfig | None = None):
        super().__init__()
        self.config = config or MemoraConfig()
        c = self.config
        assert c.n_embd == c.n_head * c.head_dim, "n_embd doit == n_head * head_dim"

        self.tok_embd = nn.Embedding(c.vocab_size, c.n_embd)
        rec, glob = set(c.recurrent_layers), set(c.global_layers)
        assert not (rec & glob), "une couche ne peut être à la fois GLA et globale"
        def kind(i): return "gla" if i in rec else "global" if i in glob else "local"
        self.blocks = nn.ModuleList([Block(c, kind(i)) for i in range(c.n_layer)])
        # fenêtres distinctes parmi les couches d'attention (None=globale, int=locale) →
        # un block_mask par fenêtre, caché par (T, device).
        self.attn_windows = {b.mixer.window for b in self.blocks if not b.recurrent}
        self.norm_f = RMSNorm(c.n_embd, c.norm_eps)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self._bm_cache: dict = {}   # {(T,device): {window: BlockMask}}

        # Cache RoPE (cos/sin) calculé une fois, partagé par toutes les couches locales.
        # Buffers non-persistants (recalculables) → suivent .to(device), absents du state_dict.
        cos, sin = build_rope_cache(c.context_len, c.head_dim, c.rope_theta, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_embd.weight   # même objet → params comptés une fois

        self.apply(self._init_weights)
        # scaling des projections de sortie résiduelles (1/√(2·n_layers))
        scale = 0.02 / math.sqrt(2 * c.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

        print(f"Memora initialisé — {self.num_params()/1e6:.1f}M paramètres "
              f"({c.n_layer} couches, GLA={c.recurrent_layers}, globales={c.global_layers})")

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device, dtype):
        """Renvoie (cos, sin) tronqués à T, dans le dtype de x. Étend le cache si T dépasse."""
        if T > self.rope_cos.size(0):  # contexte plus long que prévu → reconstruire
            cos, sin = build_rope_cache(T, self.config.head_dim, self.config.rope_theta,
                                        device, torch.float32)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:T].to(dtype), self.rope_sin[:T].to(dtype)

    @torch.compiler.disable  # create_block_mask n'est pas traçable → graph break si compilé
    def _block_masks(self, T: int, device):
        """{window: BlockMask} flex_attention, un par fenêtre distincte (None=causal pleine,
        int=fenêtre glissante), construit une fois par (T, device) puis caché."""
        key = (T, str(device))
        masks = self._bm_cache.get(key)
        if masks is None:
            masks = {}
            for w in self.attn_windows:
                if w is None:
                    def mod(b, h, qi, ki): return qi >= ki                  # globale : causal
                else:
                    def mod(b, h, qi, ki, _w=w): return (qi >= ki) & (qi - ki < _w)
                masks[w] = create_block_mask(mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device)
            self._bm_cache[key] = masks
        return masks

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_embd(idx)
        cos, sin = self._rope(T, x.device, x.dtype)
        block_masks = self._block_masks(T, x.device)
        ckpt = self.config.grad_checkpoint and self.training
        for block in self.blocks:
            if ckpt and block.recurrent:
                # recompute des couches GLA au backward → moins d'activations stockées
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, block_masks,
                                                      use_reentrant=False)
            else:
                x = block(x, cos, sin, block_masks)
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
    def from_pretrained(cls, model_name: str) -> "MemoraGLA":
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

    # 1b. stabilité inconditionnelle : gate poussé à décroissance ~0 (overflow naïf) → fini
    gla_x = GatedLinearAttention(cfg, chunk=64)
    with torch.no_grad():
        gla_x.a_bias.fill_(-60.0)
    assert torch.isfinite(gla_x(torch.randn(2, 80, cfg.n_embd))).all(), "overflow GLA"
    print("GLA stable même à décroissance ~0 (chunk=64)")

    # 2. forward + shapes + sliding window (T > window) + couche globale + RoPE-GLA + z-loss
    cfg2 = MemoraConfig(vocab_size=512, n_embd=128, n_head=4, n_kv_heads=2, head_dim=32,
                         n_layer=4, d_ff=256, recurrent_layers=(1, 3), global_layers=(2,),
                         gla_use_rope=True, sliding_window=8, context_len=256)
    model = MemoraGLA(cfg2)
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
    full = MemoraGLA(MemoraConfig())
    print(f"budget réel : {full.num_params()/1e6:.1f}M params")
