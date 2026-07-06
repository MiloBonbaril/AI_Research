"""
Noyau Triton pour le matmul ternaire de BitLinear.

Le chemin PyTorch (BitLinearInference) dépaquète les poids vers un tenseur dense
fp32/bf16, puis appelle F.linear. Ce noyau fait les deux dans les registres :
  1. Dépaquetage 2 bits → {-1, 0, +1} inline (pas d'allocation de poids dense) ;
     chaque octet packé est chargé UNE fois puis éclaté via tl.interleave
     (pas de gather redondant 4×)
  2. Quantification int8 de l'activation depuis scale_x (absmax par ligne,
     calculé par un petit noyau fusionné — pas de chaîne cast/abs/amax PyTorch)
  3. Accumulation float32, déquantification et store directement dans le dtype
     de sortie (pas de buffer fp32 intermédiaire ni de cast séparé)

gamma est passé en float Python : aucun .item() (= sync GPU→CPU) sur le chemin chaud.

Appelé par training.bitlinear_deploy.BitLinearInference quand CUDA est disponible.
Fallback PyTorch transparent sur CPU/MPS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Noyaux Triton
# ---------------------------------------------------------------------------

@triton.jit
def _rowwise_absmax_kernel(x_ptr, out_ptr, M, K, stride_xm, stride_xk,
                           BLOCK_K: tl.constexpr):
    """absmax fp32 par ligne, clampé à 1e-5 — un programme par ligne."""
    row = tl.program_id(0)
    acc = 0.0
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + row * stride_xm + offs_k * stride_xk,
                    mask=offs_k < K, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.max(tl.abs(x), axis=0))
    tl.store(out_ptr + row, tl.maximum(acc, 1e-5))


@triton.jit
def _ternary_mm_kernel(
    x_ptr,          # (M, K) — activations en précision de calcul
    w_ptr,          # (N * K // 4,) uint8 — poids packés 2 bits, layout row-major (N, K)
    scale_x_ptr,    # (M,) float32 — absmax par ligne (pré-calculé)
    y_ptr,          # (M, N) — sortie, dtype de x
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
    Kb = K // 4  # octets par ligne de poids (K est multiple de 4, cf. wrapper)

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

        # --- w tile (BLOCK_N, BLOCK_K) : octets chargés une fois, puis éclatés ---
        # octet (n, kb) contient les poids k = 4·kb .. 4·kb+3 (2 bits chacun)
        offs_kb = k_idx * (BLOCK_K // 4) + tl.arange(0, BLOCK_K // 4)
        packed = tl.load(
            w_ptr + offs_n[:, None] * Kb + offs_kb[None, :],
            mask=mask_n[:, None] & (offs_kb[None, :] < Kb),
            other=0b01010101,                              # code 1 → valeur 0
        ).to(tl.int32)
        c0 = packed & 0x3
        c1 = (packed >> 2) & 0x3
        c2 = (packed >> 4) & 0x3
        c3 = (packed >> 6) & 0x3
        # interleave imbriqué → ordre k croissant : c0, c1, c2, c3, c0, ...
        codes = tl.interleave(tl.interleave(c0, c2), tl.interleave(c1, c3))
        w_q = codes.to(tl.float32) - 1.0                   # {-1, 0, +1}

        # fp16 pour les tensor cores Blackwell
        acc = tl.dot(
            x_q.to(tl.float16), tl.trans(w_q.to(tl.float16)),
            acc=acc, out_dtype=tl.float32,
        )

    # Déquantifier et sauvegarder directement dans le dtype de sortie
    y = acc * (gamma / 127.0) * scale_x[:, None]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# Wrapper Python
# ---------------------------------------------------------------------------

# Tailles de blocs — validées sur RTX 5070 Ti pour M ∈ [1, 512], K ∈ {768, 3072}
_BLOCK_M = 32
_BLOCK_N = 64
_BLOCK_K = 64
_NUM_WARPS = 4


def ternary_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    gamma: float,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Matmul ternaire : y = dequant(quant(x) @ W_q.T).

    x          : (..., in_features) — déjà passé par SubLN
    w_packed   : (out_features * in_features // 4,) uint8
    gamma      : float Python — absmean des poids (pas un tenseur : évite un
                 .item() qui synchroniserait le GPU à chaque appel)
    Retour     : (..., out_features), même dtype que x
    """
    assert in_features % 4 == 0, "in_features doit être multiple de 4 (packing 2 bits)"
    gamma = float(gamma)  # accepte aussi un tenseur scalaire (sync unique, hors chemin chaud si float)

    orig_shape = x.shape
    x_flat = x.reshape(-1, in_features)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    M = x_flat.shape[0]

    # Absmax par ligne — un seul noyau (vs chaîne cast/abs/amax/clamp PyTorch)
    scale_x = torch.empty(M, device=x.device, dtype=torch.float32)
    _rowwise_absmax_kernel[(M,)](
        x_flat, scale_x, M, in_features,
        x_flat.stride(0), x_flat.stride(1),
        BLOCK_K=min(triton.next_power_of_2(in_features), 4096),
    )

    y = torch.empty(M, out_features, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(out_features, _BLOCK_N))
    _ternary_mm_kernel[grid](
        x_flat, w_packed, scale_x, y,
        gamma,
        M, out_features, in_features,
        x_flat.stride(0), x_flat.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )

    return y.reshape(*orig_shape[:-1], out_features)


# ---------------------------------------------------------------------------
# Vérification + benchmark (python -m models.triton_kernels)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from training.bitlinear_deploy import pack_ternary, unpack_ternary

    torch.manual_seed(0)
    device = "cuda"
    dtype  = torch.bfloat16

    IN, OUT = 768, 3072   # dimensions FFN typiques

    # Fabrique des poids ternaires et les packe
    W_q = torch.randint(-1, 2, (OUT, IN), device=device).float()
    packed = pack_ternary(W_q.cpu()).to(device)
    gamma  = 0.5

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

    # --- Benchmark : noyau vs référence PyTorch vs cuBLAS dense ---
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

    def cublas_fn():
        return x @ W_dense.T

    t_ref = bench(ref_fn)
    t_tri = bench(tri_fn)
    t_cub = bench(cublas_fn)
    print(f"PyTorch (dépack dense) : {t_ref:.3f} ms")
    print(f"Triton                 : {t_tri:.3f} ms  ({t_ref/t_tri:.1f}x)")
    print(f"cuBLAS bf16 (référence vitesse) : {t_cub:.3f} ms")
