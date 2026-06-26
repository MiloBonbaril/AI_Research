"""
Script d'entraînement comparatif — deux modèles, même budget temps.

Entraîne deux LanguageModel sur wikitext-103 pendant exactement la même
durée (en secondes murales), puis compare leurs loss de validation.

Usage :
    python train.py                                    # GPT-2 vs GPT-2 (sanity check)
    python train.py --duration 600                     # 10 min par modèle
    python train.py --batch-size 4 --grad-accum 8      # simule batch=32

Le script :
  1. Tokenise wikitext-103 une seule fois (cache sur disque)
  2. Entraîne modèle A pendant --duration secondes
  3. Entraîne modèle B pendant exactement la même durée
  4. Évalue les deux sur le split validation
  5. Affiche le résumé comparatif
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_interface import LanguageModel


def _amp(device: str):
    """Autocast bf16 sur CUDA (pas de GradScaler nécessaire en bf16), no-op ailleurs."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Dataset : tokenise wikitext-103 et découpe en blocs de context_len
# ---------------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """
    Wikitext-103 tokenisé en blocs contigus de taille fixe.

    Premier appel : tokenise tout le split et sauve le tensor sur disque.
    Appels suivants : charge directement depuis le cache.
    """

    def __init__(self, split: str = "train", context_len: int = 1024, cache_dir: str = ".cache"):
        assert split in ("train", "validation", "test")
        self.context_len = context_len

        cache_path = Path(cache_dir) / f"wikitext103_{split}.pt"

        if cache_path.exists():
            print(f"  Cache trouvé : {cache_path}")
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            print(f"  Tokenisation du split '{split}'...")
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

            enc = tiktoken.get_encoding("gpt2")
            all_tokens = []
            for row in ds:
                text = row["text"]
                if text.strip():
                    all_tokens.extend(enc.encode(text))

            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.tokens, cache_path)
            print(f"  {len(self.tokens):,} tokens sauvés → {cache_path}")

        # Nombre de blocs complets (on jette le reste)
        self.n_blocks = len(self.tokens) // (context_len + 1)
        print(f"  {self.n_blocks:,} blocs de {context_len} tokens")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * (self.context_len + 1)
        chunk = self.tokens[start : start + self.context_len + 1]
        # ponytail: input = chunk[:-1], target = chunk[1:] — classique LM
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Boucle d'entraînement avec budget temps fixe
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Résultats d'un run d'entraînement."""
    model_name: str
    steps: int
    tokens_seen: int
    wall_time: float          # secondes réelles
    final_train_loss: float
    val_loss: float
    val_perplexity: float
    loss_history: list[float]  # loss par intervalle de log


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
        model = torch.compile(model)   # kernels fusés (flex_attention, GLA, SwiGLU…)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Cosine schedule — on ne connaît pas le nombre total de steps à l'avance,
    # mais on peut estimer grossièrement et ajuster dynamiquement.
    # ponytail: un simple cosine decay vers lr/10 suffit, pas besoin de scheduler PyTorch
    def get_lr(step: int, est_total: int) -> float:
        if est_total <= 1:
            return lr
        progress = min(step / est_total, 1.0)
        # Warmup 5% des steps
        if progress < 0.05:
            return lr * (progress / 0.05)
        # Cosine decay
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr * 0.1 + (lr - lr * 0.1) * decay

    loss_history = []
    running_loss = 0.0
    step = 0
    tokens_seen = 0
    data_iter = iter(train_loader)

    # Mode steps : on connaît le total exactement. Mode temps : on estime.
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
        # -- Vérifier le budget (temps ou steps) ---
        elapsed = time.monotonic() - start_time
        if max_steps is not None:
            if step >= max_steps:
                break
        elif elapsed >= duration_seconds:
            break

        # -- Mettre à jour le learning rate ---
        if max_steps is None and step == 10:
            # Affiner l'estimation : steps/sec × temps total
            steps_per_sec = step / elapsed
            est_total_steps = int(steps_per_sec * duration_seconds)
        current_lr = get_lr(step, est_total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # -- Gradient accumulation ---
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        batch_tokens = x.size(0) * x.size(1) * grad_accum_steps
        tokens_seen += batch_tokens
        running_loss += accum_loss

        # -- Log wandb (chaque step → courbes lisses) ---
        if wandb_run is not None:
            elapsed = time.monotonic() - start_time
            wandb_run.log({
                "train/loss": accum_loss,
                "train/lr": current_lr,
                "train/tokens_seen": tokens_seen,
                "train/tok_per_sec": tokens_seen / elapsed,
            }, step=step)

        # -- Log console ---
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

    # -- Évaluation validation ---
    val_loss = evaluate(model, val_loader, device)
    val_ppl = math.exp(min(val_loss, 20))  # cap pour éviter overflow

    print(f"\n  [{model_name}] Terminé — {step} steps en {wall_time:.1f}s")
    print(f"  [{model_name}] Val loss: {val_loss:.4f} | Val perplexity: {val_ppl:.2f}\n")

    if wandb_run is not None:
        wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)
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
    return wandb.init(
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
            "lr": args.lr,
            "context_len": args.context_len,
            "max_steps": args.max_steps,
            "duration": None if args.max_steps is not None else args.duration,
            "device": device,
        },
    )


@torch.no_grad()
def evaluate(model: LanguageModel, val_loader: DataLoader, device: str, max_batches: int = 100) -> float:
    """Évalue la loss moyenne sur le split validation (capped à max_batches)."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with _amp(device):
            total_loss += model.loss(x, y).item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


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
        ("Steps",          f"{a.steps:,}",           f"{b.steps:,}"),
        ("Tokens vus",     f"{a.tokens_seen:,}",     f"{b.tokens_seen:,}"),
        ("Temps mural (s)", f"{a.wall_time:.1f}",    f"{b.wall_time:.1f}"),
        ("Train loss",     f"{a.final_train_loss:.4f}", f"{b.final_train_loss:.4f}"),
        ("Val loss",       f"{a.val_loss:.4f}",      f"{b.val_loss:.4f}"),
        ("Val perplexity", f"{a.val_perplexity:.2f}",f"{b.val_perplexity:.2f}"),
    ]

    for label, va, vb in rows:
        print(f"  {label:<25} │ {va:<18} │ {vb:<18}")

    # Gagnant
    if a.val_loss < b.val_loss:
        winner = a.model_name
        delta = b.val_loss - a.val_loss
    else:
        winner = b.model_name
        delta = a.val_loss - b.val_loss
    print("  " + "─" * 56)
    print(f"  🏆 Gagnant : {winner} (Δ val_loss = {delta:.4f})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main — à adapter pour brancher tes propres modèles
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entraînement comparatif de deux modèles")
    parser.add_argument("--duration", type=int, default=300, help="Durée par modèle en secondes (défaut: 300 = 5min)")
    parser.add_argument("--max-steps", type=int, default=None, help="Entraîner sur N steps au lieu d'une durée (override --duration)")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Steps d'accumulation de gradient (batch effectif = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate max")
    parser.add_argument("--context-len", type=int, default=1024, help="Longueur de contexte")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=50, help="Fréquence de log console en steps")
    parser.add_argument("--no-wandb", action="store_true", help="Désactiver le logging Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="memora-vs-gpt2", help="Nom du projet wandb")
    parser.add_argument("--no-compile", action="store_true", help="Désactiver torch.compile (CUDA only)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing des couches GLA Memora (moins de VRAM, plus lent)")
    args = parser.parse_args()
    use_wandb = not args.no_wandb

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # -- Dataset (tokenisé une seule fois, partagé entre les deux runs) ---
    print("\nPréparation du dataset...")
    train_dataset = WikiTextDataset("train", context_len=args.context_len)
    val_dataset   = WikiTextDataset("validation", context_len=args.context_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -- Modèles à comparer ------------------------------------------------
    # MODIFIE ICI pour brancher ton propre modèle :
    #   from mon_modele import MonModele, MonConfig
    #   model_b = MonModele(MonConfig())
    #   model_b_name = "MonModele"

    from gpt2 import GPT2, GPT2Config
    from memora import Memora, MemoraConfig

    # vocab_size doit matcher le tokenizer du dataset (tiktoken gpt2 = 50257),
    # sinon les ids ≥ 49152 débordent tok_embd → CUDA device-side assert.
    # Même vocab que GPT-2 = comparaison val_loss équitable.
    model_a = Memora(MemoraConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len,
                                  grad_checkpoint=args.grad_checkpoint))
    model_a_name = "Memora"

    model_b = GPT2(GPT2Config(dropout=0.1, context_len=args.context_len))
    model_b_name = "GPT-2 Small"

    # -- Entraînement séquentiel : même budget (temps OU steps) ------------
    # --max-steps override --duration s'il est fourni.
    duration = None if args.max_steps is not None else args.duration
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # sert de group wandb ET de préfixe fichiers

    run_a = init_wandb(args.wandb_project, model_a_name, model_a, args, device, timestamp) if use_wandb else None
    result_a = train_model(
        model_a, model_a_name, train_loader, val_loader,
        duration_seconds=duration, lr=args.lr,
        grad_accum_steps=args.grad_accum, device=device,
        log_interval=args.log_interval, max_steps=args.max_steps,
        wandb_run=run_a, use_compile=not args.no_compile,
    )
    if run_a is not None:
        run_a.finish()

    # Libérer la VRAM du premier modèle
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

    # -- Résumé ---
    print_comparison(result_a, result_b)

    # -- Sauvegarde des résultats ---
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
