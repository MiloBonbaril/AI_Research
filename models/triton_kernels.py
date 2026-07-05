"""
Noyau Triton pour le matmul ternaire de BitLinear.

Le chemin PyTorch (BitLinearInference) dépaquète les poids vers un tenseur dense
fp32/bf16, puis appelle F.linear. Ce noyau fait les deux dans les registres :
  1. Dépaquetage 2 bits → {-1, 0, +1} inline (pas d'allocation de poids dense)
  2. Quantification int8 de l'activation depuis scale_x pré-calculée
  3. Accumulation float32, déquantification à la sortie

Appelé par training.bitlinear_deploy.BitLinearInference quand CUDA est disponible.
Fallback PyTorch transparent sur CPU/MPS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Noyau Triton
# ---------------------------------------------------------------------------

@triton.jit
def _ternary_mm_kernel(
    x_ptr,          # (M, K) — activations en précision de calcul
    w_ptr,          # (N * K // 4,) uint8 — poids packés 2 bits, layout row-major (N, K)
    scale_x_ptr,    # (M,) float32 — absmax par ligne (pré-calculé)
    y_ptr,          # (M, N) float32 — sortie
    gamma,          # float32 — absmean des poids
    M, N, K,
    stride_xm, stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Échelles de déquantification pour ces lignes
    scale_x = tl.load(scale_x_ptr + offs_m, mask=mask_m, other=1.0)  # (BLOCK_M,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # --- x tile (BLOCK_M, BLOCK_K) : charger + quantifier int8 ---
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)

        x_q = x_tile * (127.0 / scale_x[:, None])
        x_q = tl.minimum(tl.maximum(x_q, -128.0), 127.0)

        # --- w tile (BLOCK_N, BLOCK_K) : dépackager 2 bits → {-1, 0, +1} ---
        # w[n, k] est à la position plate n*K + k dans le buffer packé
        flat    = offs_n[:, None] * K + offs_k[None, :]    # (BLOCK_N, BLOCK_K)
        byte_i  = flat // 4
        bit_off = (flat % 4) * 2                           # ∈ {0, 2, 4, 6}

        packed = tl.load(
            w_ptr + byte_i,
            mask=mask_n[:, None] & mask_k[None, :], other=1,  # code 1 → valeur 0
        ).to(tl.int32)
        codes = (packed >> bit_off) & 0x3                  # {0, 1, 2}
        w_q   = codes.to(tl.float32) - 1.0                # {-1, 0, +1}

        # fp16 pour les tensor cores Blackwell
        acc = tl.dot(
            x_q.to(tl.float16), tl.trans(w_q.to(tl.float16)),
            acc=acc, out_dtype=tl.float32,
        )

    # Déquantifier et sauvegarder
    y = acc * (gamma / 127.0) * scale_x[:, None]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        y,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# Wrapper Python
# ---------------------------------------------------------------------------

# Tailles de blocs — validées sur RTX 5070 Ti pour K ∈ {768, 3072}
_BLOCK_M = 16
_BLOCK_N = 32
_BLOCK_K = 64


def ternary_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    gamma: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Matmul ternaire : y = dequant(quant(x) @ W_q.T).

    x          : (..., in_features) — déjà passé par SubLN
    w_packed   : (out_features * in_features // 4,) uint8
    gamma      : scalar — absmean des poids
    Retour     : (..., out_features), même dtype que x
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, in_features).contiguous()
    M = x_flat.shape[0]

    # Absmax par ligne — one-liner PyTorch, fusion possible plus tard si bottleneck
    scale_x = x_flat.float().abs().amax(dim=-1).clamp(min=1e-5)

    y = torch.empty(M, out_features, device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(out_features, _BLOCK_N))
    _ternary_mm_kernel[grid](
        x_flat, w_packed, scale_x, y,
        gamma.float().item(),
        M, out_features, in_features,
        x_flat.stride(0), x_flat.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )

    return y.to(x.dtype).reshape(*orig_shape[:-1], out_features)


# ---------------------------------------------------------------------------
# Vérification + benchmark (python -m models.triton_kernels)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import time
    from training.bitlinear_deploy import pack_ternary, unpack_ternary

    torch.manual_seed(0)
    device = "cuda"
    dtype  = torch.bfloat16

    IN, OUT = 768, 3072   # dimensions FFN typiques

    # Fabrique des poids ternaires et les packe
    W_q = torch.randint(-1, 2, (OUT, IN), device=device).float()
    packed = pack_ternary(W_q.cpu()).to(device)
    gamma  = torch.tensor(0.5, device=device)

    x = torch.randn(8, 64, IN, device=device, dtype=dtype)  # batch=8, T=64

    # --- Référence PyTorch ---
    scale_x_ref = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q_ref     = (x.float() * 127.0 / scale_x_ref).clamp(-128, 127).round()
    W_dense     = W_q.to(dtype)
    y_ref       = (x_q_ref.to(dtype) @ W_dense.T) * (gamma * scale_x_ref / 127.0)

    # --- Kernel Triton ---
    y_tri = ternary_matmul(x, packed, gamma, OUT, IN)

    max_err = (y_ref.float() - y_tri.float()).abs().max().item()
    assert max_err < 1.0, f"erreur max trop grande : {max_err:.4f}"
    print(f"Vérification OK  (erreur max = {max_err:.4f}, attendu < 1.0 — rounding int8)")

    # --- Benchmark : noyau vs référence PyTorch ---
    WARMUP, REPS = 20, 200

    def bench(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / REPS * 1e3  # ms

    def ref_fn():
        s = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = (x.float() * 127.0 / s).clamp(-128, 127).round()
        W = unpack_ternary(packed.cpu(), OUT, IN, dtype=dtype).to(device)
        return (xq.to(dtype) @ W.T) * (gamma * s / 127.0)

    def tri_fn():
        return ternary_matmul(x, packed, gamma, OUT, IN)

    t_ref = bench(ref_fn)
    t_tri = bench(tri_fn)
    print(f"PyTorch  : {t_ref:.3f} ms")
    print(f"Triton   : {t_tri:.3f} ms  ({t_ref/t_tri:.1f}x)")
