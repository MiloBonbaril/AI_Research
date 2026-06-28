"""
Script d'entraînement comparatif — deux modèles, même budget compute.

Entraîne deux LanguageModel sur wikitext-103 puis compare leurs loss de validation.

Usage :
    python compare.py --max-steps 2000                   # N steps par modèle (recommandé)
    python compare.py --duration 600                     # 10 min par modèle (wall-clock)
    python compare.py --batch-size 4 --grad-accum 8      # simule batch=32

Le script :
  1. Tokenise wikitext-103 une seule fois (cache sur disque)
  2. Entraîne modèle A
  3. Entraîne modèle B
  4. Évalue les deux sur le split validation
  5. Affiche le résumé comparatif

Axe de comparaison : tokens_seen (= steps × batch × grad_accum × context_len).
  Avec --max-steps les deux modèles voient exactement le même nombre de tokens,
  indépendamment de leur vitesse wall-clock. C'est l'axe X dans W&B.
  Budget FLOPs : Memora ≈ 546 GFLOPs/step vs GPT-2 ≈ 583 GFLOPs/step (T=2048),
  soit un avantage théorique Memora de ~6 %. Le wall-clock Memora est ~1.57x plus
  lent (RTX 5070 Ti, B=2, T=2048) — flex_attention + GLA Triton moins efficaces
  que SDPA classique. torch.compile ne réduit pas l'écart (flex_attention et GLA
  sont déjà JIT-compilés au niveau module).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WikiTextDataset, evaluate, _amp
from model_interface import LanguageModel


# ---------------------------------------------------------------------------
# Boucle d'entraînement avec budget temps fixe
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Résultats d'un run d'entraînement."""
    model_name: str
    steps: int
    tokens_seen: int
    wall_time: float
    final_train_loss: float
    val_loss: float
    val_perplexity: float
    loss_history: list[float]


def train_model(
    model: LanguageModel,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    duration_seconds: float | None,
    lr: float,
    grad_accum_steps: int,
    device: str,
    log_interval: int = 50,
    max_steps: int | None = None,
    wandb_run=None,
    use_compile: bool = False,
) -> TrainResult:
    """
    Entraîne un modèle, soit pendant `duration_seconds` secondes, soit pendant
    `max_steps` steps (l'un des deux doit être fourni).

    Utilise AdamW avec cosine annealing. Le timer / compteur de steps est
    vérifié après chaque step (pas chaque micro-batch).
    """
    assert (duration_seconds is None) != (max_steps is None), \
        "fournir exactement un de duration_seconds ou max_steps"

    model.to(device)
    model.train()
    if use_compile and device == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ponytail: cosine decay vers lr/10, estimation du total ajustée à step=10
    def get_lr(step: int, est_total: int) -> float:
        if est_total <= 1:
            return lr
        progress = min(step / est_total, 1.0)
        if progress < 0.05:
            return lr * (progress / 0.05)
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr * 0.1 + (lr - lr * 0.1) * decay

    loss_history = []
    running_loss = 0.0
    step = 0
    tokens_seen = 0
    data_iter = iter(train_loader)
    est_total_steps = max_steps if max_steps is not None else 1000

    budget = f"{max_steps} steps" if max_steps is not None else f"{duration_seconds:.0f}s"
    print(f"\n{'='*60}")
    print(f"  Entraînement : {model_name}")
    print(f"  Paramètres   : {model.num_params()/1e6:.1f}M")
    print(f"  Budget max   : {budget}")
    print(f"  Grad accum   : {grad_accum_steps} micro-batches/step")
    print(f"{'='*60}\n")

    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        if max_steps is not None:
            if step >= max_steps:
                break
        elif elapsed >= duration_seconds:
            break

        if max_steps is None and step == 10:
            steps_per_sec = step / elapsed
            est_total_steps = int(steps_per_sec * duration_seconds)
        current_lr = get_lr(step, est_total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with _amp(device):
                loss = model.loss(x, y) / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        batch_tokens = x.size(0) * x.size(1) * grad_accum_steps
        tokens_seen += batch_tokens
        running_loss += accum_loss

        if wandb_run is not None:
            elapsed = time.monotonic() - start_time
            wandb_run.log({
                "train/loss": accum_loss,
                "train/lr": current_lr,
                "train/tok_per_sec": tokens_seen / elapsed,
                "tokens_seen": tokens_seen,
            }, step=step)

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            loss_history.append(avg_loss)
            elapsed = time.monotonic() - start_time
            tok_per_sec = tokens_seen / elapsed
            remaining = (f"{max_steps - step} steps" if max_steps is not None
                         else f"{duration_seconds - elapsed:.0f}s")
            print(
                f"  [{model_name}] step {step:>5d} | "
                f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"restant {remaining}"
            )
            running_loss = 0.0

    wall_time = time.monotonic() - start_time

    val_loss = evaluate(model, val_loader, device)
    val_ppl = math.exp(min(val_loss, 20))

    print(f"\n  [{model_name}] Terminé — {step} steps en {wall_time:.1f}s")
    print(f"  [{model_name}] Val loss: {val_loss:.4f} | Val perplexity: {val_ppl:.2f}\n")

    if wandb_run is not None:
        wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl, "tokens_seen": tokens_seen}, step=step)
        wandb_run.summary.update({
            "val/loss": val_loss,
            "val/perplexity": val_ppl,
            "wall_time": wall_time,
            "tokens_seen": tokens_seen,
            "steps": step,
        })

    return TrainResult(
        model_name=model_name,
        steps=step,
        tokens_seen=tokens_seen,
        wall_time=wall_time,
        final_train_loss=loss_history[-1] if loss_history else float("nan"),
        val_loss=val_loss,
        val_perplexity=val_ppl,
        loss_history=loss_history,
    )


def init_wandb(project: str, model_name: str, model: LanguageModel, args, device: str, group: str):
    """Démarre un run wandb par modèle, groupés → courbes superposées dans le projet."""
    import wandb
    run = wandb.init(
        project=project,
        name=model_name,
        group=group,
        reinit=True,
        config={
            "model": model_name,
            "params_M": round(model.num_params() / 1e6, 1),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "tokens_per_step": args.batch_size * args.grad_accum * args.context_len,
            "lr": args.lr,
            "context_len": args.context_len,
            "max_steps": args.max_steps,
            "duration": None if args.max_steps is not None else args.duration,
            "device": device,
            "comparison_axis": "tokens_seen",
        },
    )
    # tokens_seen comme axe X → courbes Memora/GPT-2 comparables à iso-compute
    run.define_metric("tokens_seen")
    run.define_metric("*", step_metric="tokens_seen")
    return run


# ---------------------------------------------------------------------------
# Résumé comparatif
# ---------------------------------------------------------------------------

def print_comparison(a: TrainResult, b: TrainResult):
    """Affiche un tableau comparatif des deux runs."""
    print("\n" + "=" * 60)
    print("  RÉSUMÉ COMPARATIF")
    print("=" * 60)
    header = f"  {'Métrique':<25} {'│ ' + a.model_name:<20} {'│ ' + b.model_name:<20}"
    print(header)
    print("  " + "─" * 56)

    rows = [
        ("Steps",          f"{a.steps:,}",              f"{b.steps:,}"),
        ("Tokens vus",     f"{a.tokens_seen:,}",        f"{b.tokens_seen:,}"),
        ("Temps mural (s)", f"{a.wall_time:.1f}",       f"{b.wall_time:.1f}"),
        ("Train loss",     f"{a.final_train_loss:.4f}", f"{b.final_train_loss:.4f}"),
        ("Val loss",       f"{a.val_loss:.4f}",         f"{b.val_loss:.4f}"),
        ("Val perplexity", f"{a.val_perplexity:.2f}",   f"{b.val_perplexity:.2f}"),
    ]

    for label, va, vb in rows:
        print(f"  {label:<25} │ {va:<18} │ {vb:<18}")

    if a.val_loss < b.val_loss:
        winner, delta = a.model_name, b.val_loss - a.val_loss
    else:
        winner, delta = b.model_name, a.val_loss - b.val_loss
    print("  " + "─" * 56)
    print(f"  🏆 Gagnant : {winner} (Δ val_loss = {delta:.4f})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entraînement comparatif de deux modèles")
    parser.add_argument("--duration", type=int, default=300, help="Durée par modèle en secondes (défaut: 300 = 5min)")
    parser.add_argument("--max-steps", type=int, default=None, help="Entraîner sur N steps au lieu d'une durée (override --duration)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--lr-memora", type=float, default=6e-4, help="LR Memora (axe 11)")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="memora-vs-gpt2")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()
    use_wandb = not args.no_wandb

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print("\nPréparation du dataset...")
    train_dataset = WikiTextDataset("train", context_len=args.context_len)
    val_dataset   = WikiTextDataset("validation", context_len=args.context_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    from gpt2 import GPT2, GPT2Config
    from memora import Memora, MemoraConfig

    model_b = Memora(MemoraConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len,
                                  grad_checkpoint=args.grad_checkpoint))
    model_b_name = "Memora"

    model_a = GPT2(GPT2Config(dropout=0.1, context_len=args.context_len))
    model_a_name = "GPT-2 Small"

    duration  = None if args.max_steps is not None else args.duration
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_a = init_wandb(args.wandb_project, model_a_name, model_a, args, device, timestamp) if use_wandb else None
    result_a = train_model(
        model_a, model_a_name, train_loader, val_loader,
        duration_seconds=duration, lr=args.lr_memora,
        grad_accum_steps=args.grad_accum, device=device,
        log_interval=args.log_interval, max_steps=args.max_steps,
        wandb_run=run_a, use_compile=not args.no_compile,
    )
    if run_a is not None:
        run_a.finish()

    model_a.cpu()
    torch.cuda.empty_cache() if device == "cuda" else None

    run_b = init_wandb(args.wandb_project, model_b_name, model_b, args, device, timestamp) if use_wandb else None
    result_b = train_model(
        model_b, model_b_name, train_loader, val_loader,
        duration_seconds=duration, lr=args.lr,
        grad_accum_steps=args.grad_accum, device=device,
        log_interval=args.log_interval, max_steps=args.max_steps,
        wandb_run=run_b, use_compile=not args.no_compile,
    )
    if run_b is not None:
        run_b.finish()

    print_comparison(result_a, result_b)

    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    for r in (result_a, result_b):
        out = {
            "model": r.model_name,
            "steps": r.steps,
            "tokens_seen": r.tokens_seen,
            "wall_time": r.wall_time,
            "final_train_loss": r.final_train_loss,
            "val_loss": r.val_loss,
            "val_perplexity": r.val_perplexity,
            "loss_history": r.loss_history,
        }
        fname = results_path / f"{timestamp}_{r.model_name.replace(' ', '_')}.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nRésultats sauvés dans {results_path}/")


if __name__ == "__main__":
    main()
