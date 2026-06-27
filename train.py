"""
train.py — Entraînement complet de Memora sur WikiText-103.

Caractéristiques :
  - Schedule WSD (Warmup linéaire → Stable → Cosine Decay) : pas d'endpoint figé,
    reprend proprement sans déformer la courbe si on augmente --steps.
  - Checkpoint atomique (écriture tmp→rename) : latest.pt + best.pt
  - Reprise complète : poids, état optimiseur, step, tokens vus, RNG
  - bf16 autocast + torch.compile (fusé avec flex_attention/GLA) + grad checkpoint
  - Logging wandb : loss, grad_norm, lr, tok/s, métriques val

Usage :
    python train.py --steps 10000
    python train.py --steps 20000 --resume          # reprend depuis checkpoints/latest.pt
    python train.py --steps 5000 --no-wandb --no-compile  # debug/CPU
    python train.py --steps 50000 --batch-size 4 --grad-accum 8  # batch eff=32 séqs
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WikiTextDataset, evaluate, _amp
from memora import Memora, MemoraConfig


# ---------------------------------------------------------------------------
# Schedule WSD
# ---------------------------------------------------------------------------

def get_lr(step: int, total: int, warmup: int, decay: int,
           peak: float, min_lr: float) -> float:
    """Warmup linéaire → plateau stable → cosine decay sur les derniers `decay` steps."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step < total - decay:
        return peak
    t = (step - (total - decay)) / max(decay, 1)
    return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Checkpoint atomique
# ---------------------------------------------------------------------------

def _save(path: Path, model: torch.nn.Module, optimizer, step: int,
          best_val_loss: float, tokens_seen: int, loss_history: list, args):
    """Écriture atomique : tmp → rename, évite la corruption si crash mi-écriture."""
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "loss_history": loss_history,
        "args": vars(args),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, tmp)
    tmp.rename(path)


def _load(path: Path, model: torch.nn.Module, optimizer, device: str):
    """Restaure poids, état optimiseur et RNG. Renvoie (step, best_val_loss, tokens_seen, loss_history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # Les états RNG sont des ByteTensors CPU ; map_location peut les avoir déplacés.
    torch.set_rng_state(ckpt["rng_cpu"].cpu())
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
    return (
        ckpt["step"],
        ckpt["best_val_loss"],
        ckpt.get("tokens_seen", 0),
        ckpt.get("loss_history", []),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Entraînement complet de Memora sur WikiText-103")
    # Budget
    p.add_argument("--steps",          type=int,   default=10_000,
                   help="Steps d'optimisation totaux (défaut: 10 000)")
    p.add_argument("--warmup-steps",   type=int,   default=None,
                   help="Steps de warmup (défaut: 5%% de --steps)")
    p.add_argument("--decay-steps",    type=int,   default=None,
                   help="Steps de cosine decay final (défaut: 20%% de --steps)")
    # Optimiseur
    p.add_argument("--lr",             type=float, default=6e-4,
                   help="LR de pointe (AdamW, schedule WSD)")
    p.add_argument("--min-lr",         type=float, default=None,
                   help="LR minimal en fin de decay (défaut: lr/10)")
    p.add_argument("--weight-decay",   type=float, default=0.1)
    p.add_argument("--grad-clip",      type=float, default=1.0,
                   help="Clipping de la norme du gradient")
    p.add_argument("--dropout",        type=float, default=0.0)
    # Batch
    p.add_argument("--batch-size",     type=int,   default=2,
                   help="Micro-batch (séquences par GPU)")
    p.add_argument("--grad-accum",     type=int,   default=16,
                   help="Accumulation gradient (batch eff = batch_size × grad_accum × context_len tokens)")
    p.add_argument("--context-len",    type=int,   default=2048)
    # Checkpoint
    p.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    p.add_argument("--resume",         action="store_true",
                   help="Reprendre depuis checkpoints/latest.pt si existant")
    # Logging
    p.add_argument("--eval-interval",  type=int,   default=500,
                   help="Évaluation val toutes les N steps")
    p.add_argument("--save-interval",  type=int,   default=1000,
                   help="Sauvegarde latest.pt toutes les N steps")
    p.add_argument("--log-interval",   type=int,   default=50)
    # Infra
    p.add_argument("--device",         type=str,   default="auto")
    p.add_argument("--no-wandb",       action="store_true")
    p.add_argument("--wandb-project",  type=str,   default="memora-train")
    p.add_argument("--no-compile",     action="store_true",
                   help="Désactiver torch.compile (CUDA only)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recompute des couches GLA au backward (économise VRAM, ralentit ~15%%)")
    args = p.parse_args()

    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 20)
    decay  = args.decay_steps  if args.decay_steps  is not None else max(1, args.steps // 5)
    min_lr = args.min_lr       if args.min_lr       is not None else args.lr / 10

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    # -- Dataset (cache partagé avec compare.py) ---
    print("\nPréparation du dataset...")
    train_ds = WikiTextDataset("train",      context_len=args.context_len)
    val_ds   = WikiTextDataset("validation", context_len=args.context_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # -- Modèle + optimiseur (créés AVANT compile pour que load_state_dict utilise les clés propres) ---
    model = Memora(MemoraConfig(
        vocab_size=50257,          # tiktoken gpt2 → même vocab que GPT-2 pour comparaison équitable
        context_len=args.context_len,
        dropout=args.dropout,
        grad_checkpoint=args.grad_checkpoint,
    ))
    model.to(device)
    n_params = model.num_params()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),         # β2=0.95 : standard LLM, plus agressif que défaut 0.999
        weight_decay=args.weight_decay,
        fused=device == "cuda",    # kernel AdamW fusé (CUDA uniquement) → ~20% plus rapide
    )

    ckpt_dir    = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path   = ckpt_dir / "best.pt"

    start_step    = 0
    best_val_loss = float("inf")
    tokens_seen   = 0
    loss_history  = []

    if args.resume and latest_path.exists():
        print(f"\nReprise depuis {latest_path}...")
        start_step, best_val_loss, tokens_seen, loss_history = _load(
            latest_path, model, optimizer, device
        )
        print(f"  step={start_step:,}  best_val_loss={best_val_loss:.4f}  tokens_vus={tokens_seen:,}")
    elif args.resume:
        print(f"\n--resume : {latest_path} introuvable, entraînement depuis zéro.")

    # Compile APRÈS chargement des poids : évite le préfixe _orig_mod dans les clés du state_dict.
    # flex_attention et GLA bénéficient de l'inlining de kernels sous torch.compile.
    if not args.no_compile and device == "cuda":
        model = torch.compile(model)

    model.train()

    # -- wandb ---
    run = None
    if not args.no_wandb:
        import wandb
        eff_tokens_per_step = args.batch_size * args.grad_accum * args.context_len
        run = wandb.init(
            project=args.wandb_project,
            name=f"memora_{time.strftime('%Y%m%d_%H%M%S')}",
            resume="allow",
            config={
                "steps": args.steps,
                "warmup": warmup,
                "decay": decay,
                "lr": args.lr,
                "min_lr": min_lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "eff_tokens_per_step": eff_tokens_per_step,
                "context_len": args.context_len,
                "device": device,
                "params_M": round(n_params / 1e6, 1),
                "compile": not args.no_compile,
                "grad_checkpoint": args.grad_checkpoint,
            },
        )

    eff_batch_tok = args.batch_size * args.grad_accum * args.context_len
    print(f"\n{'='*60}")
    print(f"  Memora — {n_params/1e6:.1f}M params")
    print(f"  Steps     : {start_step:,} → {args.steps:,}")
    print(f"  Schedule  : warmup={warmup} / stable / decay={decay}  (WSD)")
    print(f"  LR        : {args.lr:.1e} → {min_lr:.1e}")
    print(f"  Batch eff : {args.batch_size * args.grad_accum} séqs × {args.context_len} tok  ({eff_batch_tok:,} tok/step)")
    print(f"  Tokens ~  : {eff_batch_tok * args.steps / 1e6:.0f}M sur tout l'entraînement")
    print(f"{'='*60}\n")

    data_iter    = iter(train_loader)
    running_loss = 0.0
    t0           = time.monotonic()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.steps, warmup, decay, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # ponytail: shuffle repart de zéro au resume → position exacte non restaurée,
                # acceptable pour la recherche (même distribution, ordre différent).
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            with _amp(device):
                loss = model.loss(x, y) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()

        tokens_seen  += x.size(0) * x.size(1) * args.grad_accum
        running_loss += accum_loss
        elapsed       = time.monotonic() - t0

        if run is not None:
            run.log({
                "train/loss":        accum_loss,
                "train/grad_norm":   grad_norm,
                "train/lr":          lr,
                "train/tokens_seen": tokens_seen,
                "train/tok_per_sec": tokens_seen / max(elapsed, 1e-9),
            }, step=step + 1)

        # -- Log console ---
        if (step + 1) % args.log_interval == 0:
            avg = running_loss / args.log_interval
            loss_history.append(avg)
            running_loss = 0.0
            tok_s = tokens_seen / max(elapsed, 1e-9)
            pct   = 100 * (step + 1) / args.steps
            print(f"  [{pct:5.1f}%] step {step+1:>6d} | loss {avg:.4f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tok_s:,.0f} tok/s")

        # -- Évaluation ---
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            val_loss = evaluate(model, val_loader, device)
            val_ppl  = math.exp(min(val_loss, 20))
            print(f"\n  [eval] step {step+1:,} | val_loss {val_loss:.4f} | ppl {val_ppl:.2f}\n")
            if run is not None:
                run.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                raw = getattr(model, "_orig_mod", model)
                _save(best_path, raw, optimizer, step + 1,
                      best_val_loss, tokens_seen, loss_history, args)
                print(f"  [best] val_loss={val_loss:.4f} → {best_path}\n")

        # -- Checkpoint périodique ---
        if (step + 1) % args.save_interval == 0:
            raw = getattr(model, "_orig_mod", model)
            _save(latest_path, raw, optimizer, step + 1,
                  best_val_loss, tokens_seen, loss_history, args)
            print(f"  [ckpt] step {step+1:,} → {latest_path}")

    # Sauvegarde finale (step peut ne pas être un multiple de save_interval)
    raw = getattr(model, "_orig_mod", model)
    _save(latest_path, raw, optimizer, args.steps,
          best_val_loss, tokens_seen, loss_history, args)

    if run is not None:
        run.summary.update({
            "best_val_loss":    best_val_loss,
            "best_val_ppl":     math.exp(min(best_val_loss, 20)),
            "tokens_seen":      tokens_seen,
        })
        run.finish()

    print(f"\nEntraînement terminé.")
    print(f"  Best val loss  : {best_val_loss:.4f}  (ppl {math.exp(min(best_val_loss, 20)):.2f})")
    print(f"  Tokens vus     : {tokens_seen:,}")
    print(f"  Checkpoints    : {ckpt_dir}/")


if __name__ == "__main__":
    main()
