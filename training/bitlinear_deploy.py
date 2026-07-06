"""
bitlinear_deploy.py — chemin de déploiement ternaire pour Cortex.

L'entraînement garde un maître fp32 (nécessaire au STE). Le déploiement, lui,
n'en a plus besoin : on fige les poids en ternaire {-1,0,+1}, on les PACKE à
2 bits (4 poids/octet), et on jette le fp32. C'est là — et seulement là — que
la VRAM ternaire devient réelle.

Workflow :
    model = Cortex(...)                     # entraîné, poids fp32 chargés
    model.eval()
    report_memory(model, "avant packing")

    convert_to_inference(model)             # BitLinear -> BitLinearInference (packé)
    report_memory(model, "apres packing")   # les poids linéaires chutent ~16x

Note honnête : le packing réduit la mémoire *résidente* des poids. Le matmul
PyTorch dépacke transitoirement vers un tenseur dense (buffer temporaire plein
format) — un vrai kernel ternaire (bitnet.cpp / Triton) éviterait ce dépacking.
Pour l'objectif « faire tenir un gros modèle résident en VRAM », c'est bien la
mémoire résidente qui compte, donc ce chemin sert exactement votre but.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adaptez ce chemin d'import à votre arborescence :
from models.cortex import BitLinear


# ---------------------------------------------------------------------------
# Packing ternaire : {-1, 0, +1} -> 2 bits, 4 poids par octet
# ---------------------------------------------------------------------------

def pack_ternary(W_q: torch.Tensor) -> torch.Tensor:
    """
    W_q : tenseur de valeurs {-1, 0, +1}, forme (out, in).
    Retour : uint8 packé (~in*out/4 octets).
    Encodage : -1 -> 0b00, 0 -> 0b01, +1 -> 0b10  (via +1 : {0,1,2}).
    """
    codes = (W_q + 1).to(torch.uint8).reshape(-1)      # {-1,0,1} -> {0,1,2}
    pad = (-codes.numel()) % 4
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    codes = codes.reshape(-1, 4)
    packed = (codes[:, 0]
              | (codes[:, 1] << 2)
              | (codes[:, 2] << 4)
              | (codes[:, 3] << 6))
    return packed.contiguous()                          # uint8


def unpack_ternary(packed: torch.Tensor, out_features: int, in_features: int,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse de pack_ternary. Retourne un poids dense {-1,0,+1} en `dtype`."""
    b = packed
    q = torch.stack([b & 0b11,
                     (b >> 2) & 0b11,
                     (b >> 4) & 0b11,
                     (b >> 6) & 0b11], dim=1).reshape(-1)
    q = q[: out_features * in_features]
    W = q.to(dtype) - 1.0                                # {0,1,2} -> {-1,0,1}
    return W.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# BitLinear d'inférence : pas de maître fp32, poids packés 2 bits
# ---------------------------------------------------------------------------

class BitLinearInference(nn.Module):
    """
    Construit depuis un BitLinear entraîné. Fige et packe les poids ternaires,
    conserve gamma (échelle absmean) et la RMSNorm (SubLN). Aucun poids fp32.
    """

    def __init__(self, src: BitLinear):
        super().__init__()
        with torch.no_grad():
            W = src.weight
            gamma = W.abs().mean().clamp(min=1e-5)
            W_q = (W / gamma).round().clamp(-1, 1)

        self.out_features, self.in_features = W.shape
        self.register_buffer("packed", pack_ternary(W_q))
        self.register_buffer("gamma", gamma.clone())
        # float Python extrait UNE fois ici : le passer au kernel évite un
        # .item() (sync GPU→CPU) à chaque forward
        self.gamma_f = float(gamma)
        self.norm = src.norm                              # réutilise la SubLN
        self.bias = src.bias if src.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if x.is_cuda:
            from models.triton_kernels import ternary_matmul
            return ternary_matmul(x, self.packed, self.gamma_f, self.out_features, self.in_features)
        # CPU / MPS fallback : dépaquetage dense + F.linear
        W = unpack_ternary(self.packed, self.out_features, self.in_features, dtype=x.dtype)
        scale_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * 127.0 / scale_x).round().clamp(-128, 127)
        return F.linear(x_q, W, self.bias) * (self.gamma * scale_x / 127.0)


# ---------------------------------------------------------------------------
# Conversion récursive du modèle
# ---------------------------------------------------------------------------

def convert_to_inference(module: nn.Module) -> nn.Module:
    """Remplace en place tous les BitLinear par des BitLinearInference (packés)."""
    for name, child in list(module.named_children()):
        if isinstance(child, BitLinear):
            setattr(module, name, BitLinearInference(child))
        else:
            convert_to_inference(child)
    return module


# ---------------------------------------------------------------------------
# Rapport mémoire (poids résidents, par catégorie)
# ---------------------------------------------------------------------------

def report_memory(model: nn.Module, label: str = "") -> None:
    """Affiche les octets résidents : linéaires ternaires vs embeddings vs reste."""
    lin_bytes = emb_bytes = other_bytes = 0

    for m in model.modules():
        if isinstance(m, BitLinear):
            lin_bytes += m.weight.numel() * m.weight.element_size()
        elif isinstance(m, BitLinearInference):
            lin_bytes += m.packed.numel() * m.packed.element_size()
        elif isinstance(m, nn.Embedding):
            emb_bytes += m.weight.numel() * m.weight.element_size()

    # « reste » = normes, biais, lm_head non-tied éventuel, etc.
    counted = set()
    for m in model.modules():
        if isinstance(m, (BitLinear, BitLinearInference, nn.Embedding)):
            for p in m.parameters(recurse=False):
                counted.add(id(p))
            for b in m.buffers(recurse=False):
                counted.add(id(b))
    for p in model.parameters():
        if id(p) not in counted:
            other_bytes += p.numel() * p.element_size()

    mb = lambda n: n / 1e6
    head = f"[{label}]" if label else "[mémoire]"
    print(f"{head}")
    print(f"  poids linéaires : {mb(lin_bytes):8.1f} Mo")
    print(f"  embeddings      : {mb(emb_bytes):8.1f} Mo")
    print(f"  reste           : {mb(other_bytes):8.1f} Mo")
    print(f"  TOTAL poids     : {mb(lin_bytes + emb_bytes + other_bytes):8.1f} Mo")


# ---------------------------------------------------------------------------
# Démo autonome (sans le vrai modèle) : vérifie packing + gain
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Fabrique un BitLinear factice pour tester le round-trip et le gain.
    bl = BitLinear(768, 3072)
    with torch.no_grad():
        gamma = bl.weight.abs().mean().clamp(min=1e-5)
        W_q = (bl.weight / gamma).round().clamp(-1, 1)

    packed = pack_ternary(W_q)
    W_back = unpack_ternary(packed, 3072, 768, dtype=torch.float32)
    assert torch.equal(W_back, W_q.float()), "round-trip packing cassé"

    fp32_bytes = bl.weight.numel() * 4
    packed_bytes = packed.numel() * 1
    print("Round-trip packing OK")
    print(f"  fp32   : {fp32_bytes/1e3:7.1f} Ko")
    print(f"  packé  : {packed_bytes/1e3:7.1f} Ko  ({fp32_bytes/packed_bytes:.1f}x plus petit)")
