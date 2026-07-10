"""
Oneira — simulateur mental par-dessus le backbone Memora.

Générer token par token, c'est conduire : Memora maintient déjà, dans ses couches GLA, une
représentation compacte du passé (un résumé compressé, mis à jour en continu — cf.
models/memora.py). Ce qui manque est le SIMULATEUR : un opérateur F qui fait évoluer cette
représentation à partir d'une action envisagée, SANS générer le texte réel intermédiaire.

Le point central : on ne note JAMAIS l'état imaginé par comparaison directe à l'état réel
(régression de vecteurs — absurde, personne ne simule la forme des nuages). On le note par
ÉQUIVALENCE DE VALEUR : on compare les PRÉDICTIONS que l'état imaginé permet (via un readout
partagé) à celles que permet l'état réel. La simulation peut être floue et incomplète ; elle
doit seulement être décisionnellement juste.

Trois branches de loss (cf. Oneira.compute_losses) :
  - L_main : cross-entropy du backbone Memora, inchangée.
  - L_head : cross-entropy d'une petite tête de simulation (SimulationHead, GLA-only, LM head
    tied) posée sur les hidden states du backbone. Les couches d'attention locale/globale de
    Memora n'ont pas d'état "imaginable" — on loge donc le monde simulable dans un module
    purement récurrent, dont l'état sert d'assise à F.
  - L_sim : équivalence de valeur. Pour quelques paires (i, k) échantillonnées, on saute k
    chunks via Ŝ = S + F_k(S, z) (z = pooling des hidden states sautés) puis on relit le chunk
    d'arrivée avec Ŝ — jamais avec l'état réel.

Décision de design centrale : le pas de monde est un CHUNK (sim_chunk=64 tokens), pas un
token — un saut token-à-token est trivial ; un chunk est un "pas de temps" sémantiquement
significatif, et s'aligne sur les frontières où un kernel GLA chunké matérialise son état
(cf. StatefulGLA, généralisation du _chunked de models/memora.py).

Deux métriques diagnostiques décident de la vie ou de la mort du projet (cf. metrics de
compute_losses, à logger côté training loop) :
  - null_gap(k)  = CE(lecture(S_t))        − CE(lecture(S_réel_{t+k}))  : coût de supposer
    le monde figé. Doit être franchement positif (sinon rien à simuler à cet horizon).
  - sim_gap(k)   = CE(lecture(Ŝ))          − CE(lecture(S_réel_{t+k}))  : coût de F. Doit être
    significativement < null_gap. S'il s'en approche sans jamais s'en détacher, Oneira est
    falsifiée proprement.

Gotchas / écarts délibérés vs la spec de conversation :
  - `fla` n'est pas installé dans cet environnement et son API chunkée n'expose de toute
    façon pas de passage d'état explicite entre appels — StatefulGLA réimplémente la boucle
    de scan pur-torch de GatedLinearAttention._chunked (déjà testée dans models/memora.py),
    généralisée pour accepter/renvoyer un état non-nul. Voir le test d'équivalence en __main__.
  - Aucune logique de reset par frontière de document n'existe ENCORE ailleurs dans ce repo
    (grep sur "PG19"/"document" : rien — le dataset actuel, wikitext-103 via
    training/dataset.py, ne segmente pas par document). `compute_losses(doc_boundary_chunk=...)`
    est un point d'extension prêt à recevoir cette info le jour où PG19 est branché ; en son
    absence (None), tout le batch est traité comme un seul document.
  - Composition multi-échelle (F⁽¹⁾⁴ ≈ F⁽⁴⁾), F bilinéaire structuré, démos de planification :
    phase suivante, seulement si sim_gap << null_gap se confirme sur un vrai entraînement.
  - Warmup de λ₂ (0→0.1) et le budget kernel-launch (32 lancements/couche à c=64 sur 2048
    tokens, ~15-20% de surcoût/step) sont des préoccupations de training loop, pas de ce
    fichier — lambda_head/lambda_sim sont exposés en config comme valeurs statiques pour le
    self-check ; un futur train_oneira.py les ordonnancerait.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.memora import GatedLinearAttention, Memora, MemoraConfig, RMSNorm, SwiGLU
from models.model_interface import LanguageModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OneiraConfig(MemoraConfig):
    """Hyperparamètres du backbone (hérités de MemoraConfig) + de la couche simulateur."""

    # vocab_size par défaut MemoraConfig (49152) est incompatible avec le tokenizer tiktoken
    # gpt2 (50257) qu'utilise réellement le dataset — cf. gotcha CLAUDE.md (train.py surcharge
    # Memora à 50257). On corrige le défaut ici plutôt que de reproduire le piège.
    vocab_size: int = 50257

    # --- tête de simulation ---
    sim_chunk: int = 64             # c : taille du "pas de temps" monde, en tokens
    sim_n_layer: int = 2            # couches GLA-only de la tête de simulation
    sim_gla_head_dim: int = 64      # d_head de l'état simulable (état par tête = d_head²)

    # --- opérateur(s) F ---
    sim_horizons: tuple = (1, 4)    # k : horizons de saut simulés, en nombre de chunks
    f_hidden: int = 1024            # dim cachée du MLP F (résiduel, zero-init dernière couche)
    action_dim: int = 256           # dim de z (pooling projeté des hidden states sautés)
    f_stop_grad_state: bool = True  # stop-gradient sur S en entrée de F (stabilité premiers runs
                                     # — à retirer une fois stable, cf. docstring module)

    # --- budget de paires (i, k) échantillonnées pour L_sim ---
    sim_pairs: int = 8

    # --- poids des branches de loss (statiques ici ; warmup de lambda_sim = training loop) ---
    lambda_head: float = 1.0
    lambda_sim: float = 0.1


# ---------------------------------------------------------------------------
# StatefulGLA — généralise GatedLinearAttention._chunked avec état explicite
# ---------------------------------------------------------------------------

class StatefulGLA(GatedLinearAttention):
    """GLA à état explicite : accepte un `initial_state` et renvoie l'état final.

    Nécessaire pour chaîner les segments de la tête de simulation sur les frontières de
    chunk monde (models.memora.GatedLinearAttention part toujours de S=0 et ne renvoie pas
    l'état). Reprend exactement la même récurrence que _chunked (déjà validée dans
    models/memora.py) — seed non-nulle en plus, donc la correction suit directement de celle
    de _chunked (test d'équivalence en __main__ ci-dessous).

    Pas de repli fla : les segments sont de toute façon appelés en boucle Python (cf. note
    "coût de lancement" du docstring module), le chemin pur-torch suffit. États toujours en
    fp32 (dérive numérique sinon, même sous un backbone bf16).
    """

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, g = self._proj(x)
        o, s_final = self._chunked_stateful(q, k, v, g, initial_state)
        o = o.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.n_head * self.hd)
        o = o * torch.sigmoid(self.g_proj(x))
        return self.o_proj(o), s_final

    def _chunked_stateful(self, q, k, v, g, initial_state):
        """Identique à GatedLinearAttention._chunked, mais S part de `initial_state` (fp32)
        au lieu de zéros, et est renvoyé (fp32, jamais retronqué) en fin de scan."""
        B, H, T, d = q.shape
        C = self.chunk
        pad = (C - T % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad)); g = F.pad(g, (0, 0, 0, pad))
        Tp = T + pad
        nC = Tp // C

        q, k, v, g = (t.float() for t in (q, k, v, g))
        q = q.view(B, H, nC, C, d); k = k.view(B, H, nC, C, d)
        v = v.view(B, H, nC, C, d); g = g.view(B, H, nC, C, d)

        Gc = g.cumsum(dim=3)
        q_s = q * Gc.exp()

        last = Gc[:, :, :, -1, :]
        kbar = k * (last.unsqueeze(3) - Gc).exp()
        S_chunk = torch.einsum("bhcjd,bhcje->bhcde", kbar, v)
        decay_chunk = last.exp()

        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

        o = torch.empty(B, H, nC, C, d, device=q.device, dtype=q.dtype)
        S = initial_state.float()
        for c in range(nC):
            Gc_c = Gc[:, :, c]
            diff = Gc_c.unsqueeze(3) - Gc_c.unsqueeze(2)
            D = diff.clamp(max=0.0).exp().masked_fill(~causal[:, :, None], 0.0)
            A = torch.einsum("bhid,bhjd,bhijd->bhij", q[:, :, c], k[:, :, c], D)
            o_intra = torch.einsum("bhij,bhjd->bhid", A, v[:, :, c])
            o_inter = torch.einsum("bhid,bhde->bhie", q_s[:, :, c], S)
            o[:, :, c] = o_intra + o_inter
            S = decay_chunk[:, :, c].unsqueeze(-1) * S + S_chunk[:, :, c]

        out = o.view(B, H, Tp, d)[:, :, :T].to(q.dtype)
        return out, S


# ---------------------------------------------------------------------------
# WorldOperator — F_k : opérateur de saut résiduel, zero-init
# ---------------------------------------------------------------------------

class WorldOperator(nn.Module):
    """F_k : MLP résiduel prédisant le delta d'état à l'horizon k (en chunks monde).

    Zero-init de la dernière couche + connexion résiduelle → Ŝ = S EXACTEMENT au step 0 :
    l'opérateur démarre sur la baseline nulle "le monde ne bouge pas" et n'apprend que le
    delta (ce double choix n'est pas cosmétique, cf. docstring module). Poids partagés entre
    toutes les têtes et couches de la tête de simulation ; un embedding (couche, tête) appris
    désambiguïse l'entrée pour ce MLP partagé. Un F distinct par horizon k (pas de
    conditionnement sur k — plus simple, cf. docstring module).
    """

    def __init__(self, c: OneiraConfig, n_layers: int, n_heads: int, embd_dim: int = 16):
        super().__init__()
        d = c.sim_gla_head_dim
        state_dim = d * d
        self.layer_embd = nn.Parameter(torch.randn(n_layers, embd_dim) * 0.02)
        self.head_embd = nn.Parameter(torch.randn(n_heads, embd_dim) * 0.02)
        in_dim = state_dim + c.action_dim + 2 * embd_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, c.f_hidden),
            nn.GELU(),
            nn.Linear(c.f_hidden, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.stop_grad_state = c.f_stop_grad_state

    def forward(self, states: list[torch.Tensor], z: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for layer_idx, S in enumerate(states):
            B, H, d, _ = S.shape
            S_in = S.detach() if self.stop_grad_state else S
            flat = S_in.reshape(B, H, d * d)
            le = self.layer_embd[layer_idx].view(1, 1, -1).expand(B, H, -1)
            he = self.head_embd.view(1, H, -1).expand(B, -1, -1)
            zin = z.unsqueeze(1).expand(-1, H, -1)
            inp = torch.cat([flat, zin, le, he], dim=-1)
            delta = self.net(inp).view(B, H, d, d)
            out.append(S + delta)
        return out


# ---------------------------------------------------------------------------
# SimulationHead — tête GLA-only, LM head tied
# ---------------------------------------------------------------------------

class SimulationHead(nn.Module):
    """sim_n_layer couches GLA-only par-dessus les hidden states H du backbone.

    Existe parce que les couches d'attention locale/globale de Memora n'ont pas d'état
    "imaginable" — seul un module purement récurrent a un état de taille fixe qu'un opérateur
    F peut faire évoluer sans texte réel. LM head tied avec le backbone.
    """

    def __init__(self, c: OneiraConfig, lm_head: nn.Linear):
        super().__init__()
        gla_cfg = replace(c, gla_head_dim=c.sim_gla_head_dim)
        self.n_head = c.n_head
        self.hd = c.sim_gla_head_dim
        self.layers = nn.ModuleList([StatefulGLA(gla_cfg, chunk=16) for _ in range(c.sim_n_layer)])
        self.norms1 = nn.ModuleList([RMSNorm(c.n_embd, c.norm_eps) for _ in range(c.sim_n_layer)])
        self.mlps = nn.ModuleList([SwiGLU(c) for _ in range(c.sim_n_layer)])
        self.norms2 = nn.ModuleList([RMSNorm(c.n_embd, c.norm_eps) for _ in range(c.sim_n_layer)])
        self.norm_f = RMSNorm(c.n_embd, c.norm_eps)
        self.lm_head = lm_head   # même objet que le backbone (tied), pas une copie

    def zero_states(self, B: int, device) -> list[torch.Tensor]:
        return [torch.zeros(B, self.n_head, self.hd, self.hd, device=device, dtype=torch.float32)
                for _ in range(len(self.layers))]

    def forward(self, x: torch.Tensor, states: list[torch.Tensor]):
        """x: (B,c,d) hidden states backbone d'UN segment. states: état d'entrée par couche.
        Renvoie (y, new_states) : y = hidden states de sortie (avant lm_head)."""
        new_states = []
        for layer, n1, mlp, n2, s in zip(self.layers, self.norms1, self.mlps, self.norms2, states):
            attn_out, s_new = layer(n1(x), s)
            x = x + attn_out
            x = x + mlp(n2(x))
            new_states.append(s_new)
        return self.norm_f(x), new_states

    def readout(self, x: torch.Tensor, states: list[torch.Tensor]):
        y, new_states = self.forward(x, states)
        return self.lm_head(y), new_states


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------

class Oneira(LanguageModel):
    """Oneira — backbone Memora (inchangé) + tête de simulation + opérateur(s) F.

    Le backbone maintient déjà un état GLA compact (résumé du passé, cf. Memora). Ce qui
    manque est le SIMULATEUR : un opérateur qui fait évoluer cet état sans texte réel, noté
    par équivalence de valeur — jamais par comparaison directe à l'état "réel" (cf. docstring
    module). `forward`/`loss`/`generate` délèguent au backbone (contrat LanguageModel standard,
    compatible train.py/compare.py pour L_main seule) ; `compute_losses` implémente le pas
    d'entraînement à 3 branches complet.
    """

    def __init__(self, config: OneiraConfig | None = None):
        super().__init__()
        self.config = config or OneiraConfig()
        c = self.config
        self.backbone = Memora(c)
        self.sim_head = SimulationHead(c, self.backbone.lm_head)
        self.action_proj = nn.Linear(c.n_embd, c.action_dim, bias=False)
        self.F = nn.ModuleDict({
            str(k): WorldOperator(c, n_layers=c.sim_n_layer, n_heads=c.n_head)
            for k in c.sim_horizons
        })

        # overhead = total dédupliqué − backbone dédupliqué (le lm_head est tied, donc partagé
        # entre backbone et sim_head : sommer sim_head.parameters() seul le compterait deux fois).
        overhead = self.num_params() - self.backbone.num_params()
        print(f"Oneira initialisé — backbone Memora {self.backbone.num_params()/1e6:.1f}M "
              f"+ simulateur (tête {c.sim_n_layer} GLA + F×{len(c.sim_horizons)}) "
              f"{overhead/1e6:.1f}M")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.backbone(idx)

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.backbone.loss(idx, targets)   # conserve le z-loss de Memora

    @classmethod
    def from_pretrained(cls, model_name: str = "oneira") -> "Oneira":
        raise NotImplementedError(
            "Oneira est une architecture originale (backbone Memora + simulateur) — "
            "entraîner from scratch."
        )

    def compute_losses(self, idx: torch.Tensor, targets: torch.Tensor,
                        doc_boundary_chunk: torch.Tensor | None = None):
        """Le pas d'entraînement à 3 branches (cf. docstring module).

        Renvoie (loss_main, loss_head, loss_sim, metrics) — metrics: dict[str, float]
        (null_gap_k*/sim_gap_k*), diagnostics uniquement, à logger côté training loop.

        doc_boundary_chunk : (B,) index du DERNIER chunk valide par item de batch avant un
        changement de document, ou None. Le dataset actuel du repo (wikitext-103, cf.
        training/dataset.py) ne segmente PAS par document — ce paramètre est un point
        d'extension pour PG19 (cette logique de reset n'existe pas encore ailleurs dans le
        repo, cf. docstring module). Fourni, la borne appliquée est CONSERVATRICE (pire cas
        du batch) : certaines paires valides pour un item donné peuvent être exclues.
        """
        c = self.config
        B, T = idx.shape
        logits_main, H = self.backbone(idx, return_hidden=True)
        loss_main = F.cross_entropy(logits_main.reshape(-1, logits_main.size(-1)), targets.reshape(-1))

        chunk = c.sim_chunk
        n_chunks = T // chunk
        assert n_chunks >= 1, f"séquence ({T}) plus courte qu'un chunk monde ({chunk})"
        Tc = n_chunks * chunk

        # 2) tête de sim par segments — collecte de l'état AVANT chaque chunk (states_before[i]
        # = état réel au temps i, utilisé à la fois comme lecture "réelle" et comme point de
        # départ de F/de la baseline nulle pour toute paire (i, k)).
        states = self.sim_head.zero_states(B, idx.device)
        states_before = []
        ys = []
        for i in range(n_chunks):
            states_before.append(states)
            seg = H[:, i * chunk:(i + 1) * chunk]
            y, states = self.sim_head.forward(seg, states)
            ys.append(y)

        y_head = torch.cat(ys, dim=1)
        logits_head = self.sim_head.lm_head(y_head)
        loss_head = F.cross_entropy(logits_head.reshape(-1, logits_head.size(-1)),
                                     targets[:, :Tc].reshape(-1))

        # 3-4) équivalence de valeur : paires (i, k) échantillonnées, jamais à cheval sur une
        # frontière de document (cf. docstring de la méthode pour l'état actuel de ce garde-fou).
        metrics_raw: dict[str, list[float]] = {}
        loss_sim = logits_main.new_zeros(())
        n_sim_terms = 0

        max_k = max(c.sim_horizons)
        if n_chunks > max_k:
            boundary_hi = n_chunks
            if doc_boundary_chunk is not None:
                boundary_hi = min(boundary_hi, int(doc_boundary_chunk.min().item()) + 1)

            for _ in range(c.sim_pairs):
                k = c.sim_horizons[torch.randint(len(c.sim_horizons), (1,)).item()]
                hi = boundary_hi - k
                if hi <= 0:
                    continue
                i = torch.randint(hi, (1,)).item()
                arr = i + k

                z = self.action_proj(H[:, i * chunk:arr * chunk].mean(dim=1))
                S_hat = self.F[str(k)](states_before[i], z)
                arrival_seg = H[:, arr * chunk:(arr + 1) * chunk]
                arrival_targets = targets[:, arr * chunk:(arr + 1) * chunk].reshape(-1)

                logits_sim, _ = self.sim_head.readout(arrival_seg, S_hat)
                ce_sim = F.cross_entropy(logits_sim.reshape(-1, logits_sim.size(-1)), arrival_targets)
                loss_sim = loss_sim + ce_sim
                n_sim_terms += 1

                with torch.no_grad():
                    ce_real = F.cross_entropy(
                        logits_head[:, arr * chunk:(arr + 1) * chunk].reshape(-1, logits_head.size(-1)),
                        arrival_targets)
                    logits_null, _ = self.sim_head.readout(arrival_seg, states_before[i])
                    ce_null = F.cross_entropy(logits_null.reshape(-1, logits_null.size(-1)), arrival_targets)

                metrics_raw.setdefault(f"null_gap_k{k}", []).append((ce_null - ce_real).item())
                metrics_raw.setdefault(f"sim_gap_k{k}", []).append(ce_sim.item() - ce_real.item())

        if n_sim_terms > 0:
            loss_sim = loss_sim / n_sim_terms
        metrics = {name: sum(v) / len(v) for name, v in metrics_raw.items()}

        return loss_main, loss_head, loss_sim, metrics


# ---------------------------------------------------------------------------
# Self-check : chaînage d'états, null_gap sans F, F zero-init, gradients
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = OneiraConfig(
        vocab_size=256, n_embd=64, n_head=4, head_dim=16, n_layer=4, d_ff=128,
        recurrent_layers=(1,), global_layers=(), n_kv_heads=2, sliding_window=8,
        context_len=512, gla_head_dim=16,
        sim_chunk=8, sim_n_layer=2, sim_gla_head_dim=16, sim_horizons=(1, 2),
        sim_pairs=6, f_hidden=32, action_dim=16,
    )
    model = Oneira(cfg)
    model.train()
    B, T = 2, 40  # 5 chunks monde de 8 tokens
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # 1. backbone : forme + loss finie (contrat LanguageModel inchangé — L_main seule)
    logits = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    loss_bb = model.loss(idx, targets)
    assert torch.isfinite(loss_bb)
    print(f"backbone OK {tuple(logits.shape)} — loss_main={loss_bb.item():.3f}")

    # 2. LE test critique : segments chaînés (état explicite) == un seul passage sur la concat.
    #    Si ça casse, toute la plomberie d'états en aval est du bruit (cf. docstring module).
    _, H = model.backbone(idx, return_hidden=True)
    chunk = cfg.sim_chunk
    y_full, _ = model.sim_head.forward(H[:, :3 * chunk], model.sim_head.zero_states(B, idx.device))
    states = model.sim_head.zero_states(B, idx.device)
    ys = []
    for i in range(3):
        y_i, states = model.sim_head.forward(H[:, i * chunk:(i + 1) * chunk], states)
        ys.append(y_i)
    y_chained = torch.cat(ys, dim=1)
    err = (y_full - y_chained).abs().max().item()
    assert err < 1e-3, f"chaînage d'états cassé (err={err})"
    print(f"équivalence segments chaînés == passage unique OK (err max = {err:.2e})")

    # 3. null_gap mesurable SANS F (pose la barre avant même que F existe, cf. docstring)
    loss_main, loss_head, loss_sim, metrics = model.compute_losses(idx, targets)
    assert torch.isfinite(loss_main) and torch.isfinite(loss_head) and torch.isfinite(loss_sim)
    null_gaps = {k: metrics[f"null_gap_k{k}"] for k in cfg.sim_horizons if f"null_gap_k{k}" in metrics}
    assert null_gaps, "aucune paire échantillonnée — augmenter T ou réduire sim_chunk/sim_horizons"
    print(f"loss_main={loss_main.item():.3f} loss_head={loss_head.item():.3f} "
          f"loss_sim={loss_sim.item():.3f} | null_gap={null_gaps}")

    # 4. F résiduel : Ŝ≈S au step 0 (zero-init) + gradients (backbone ET F reçoivent un signal
    #    via loss_sim — sinon l'opérateur n'apprend jamais, cf. docstring).
    s_probe = [s + torch.randn_like(s) for s in model.sim_head.zero_states(B, idx.device)]
    z_probe = torch.randn(B, cfg.action_dim)
    s_hat = model.F[str(cfg.sim_horizons[0])](s_probe, z_probe)
    drift = max((a - b).abs().max().item() for a, b in zip(s_hat, s_probe))
    assert drift < 1e-5, f"F ne démarre pas sur la baseline nulle (drift={drift})"
    print(f"F zero-init OK — Ŝ=S au step 0 (drift={drift:.2e})")

    total_loss = loss_main + cfg.lambda_head * loss_head + cfg.lambda_sim * loss_sim
    total_loss.backward()
    bb_grad = model.backbone.tok_embd.weight.grad
    assert bb_grad is not None and torch.isfinite(bb_grad).all(), "backbone : gradient manquant"
    f_params = list(model.F[str(cfg.sim_horizons[0])].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in f_params), \
        "F ne reçoit aucun gradient — loss_sim ne le traverse pas"
    print("backward OK — backbone et F reçoivent des gradients non-nuls")

    # 5. composition multi-échelle / F bilinéaire / démos de planification : phase suivante,
    #    seulement en cas de succès de sim_gap << null_gap sur un vrai entraînement (cf. docstring).

    # budget params (info seulement — machinerie additionnelle par-dessus Memora ; pas de
    # contrainte ≤126M ici, Oneira n'est pas mise en concurrence budget-égal dans compare.py)
    full = Oneira(OneiraConfig())
    overhead = full.num_params() - full.backbone.num_params()
    print(f"Oneira(mini) plein budget : backbone {full.backbone.num_params()/1e6:.1f}M "
          f"+ simulateur {overhead/1e6:.1f}M = {full.num_params()/1e6:.1f}M")
