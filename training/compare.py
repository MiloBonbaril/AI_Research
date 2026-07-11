"""
Script d'entraînement comparatif — un seul modèle par invocation, même protocole compute.

Entraîne UN LanguageModel (choisi via --model) sur wikitext-103, l'évalue sur le split
validation, sauvegarde son état final, et logge tout dans wandb. La comparaison entre
modèles ne se fait PLUS via un tableau imprimé en fin de script (ça exigeait les deux
résultats en mémoire dans le même process) : elle se fait via wandb, en lançant le script
une fois par modèle avec le MÊME --group — les runs partagent alors le même groupe et
apparaissent superposés dans le projet wandb, courbes comparables sur l'axe tokens_seen.

Usage :
    python -m training.compare --model gpt2   --group run1 --max-steps 2000
    python -m training.compare --model oneira --group run1 --max-steps 2000
    python -m training.compare --model cortex --duration 600      # --group omis → timestamp auto
    python -m training.compare --model oneira --no-save           # smoke test, pas de checkpoint

Modèles disponibles (--model) : gpt2, cortex, deepseek, effy, memora, oneira.

Axe de comparaison : tokens_seen (= steps × batch × grad_accum × context_len). Avec
--max-steps le modèle voit un nombre de tokens fixé indépendamment de sa vitesse wall-clock —
lancer deux modèles avec le même --max-steps et le même --group les rend comparables à
iso-compute dans wandb.

Oneira est entraîné ici via son contrat LanguageModel standard (backbone Memora, L_main +
z-loss, cf. Oneira.loss) — PAS via son compute_losses() 3-branches (L_head/L_sim, cf.
models/oneira.py). Ce script compare des BACKBONES à budget égal ; brancher L_sim demande un
training loop dédié (paires (i,k), warmup de lambda_sim, chunks) hors de la portée de ce
protocole générique.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import WikiTextDataset, evaluate, _amp
from models.model_interface import LanguageModel
from models.gpt2 import GPT2, GPT2Config
from models.cortex import Cortex, CortexConfig
from models.deepseek import DeepSeek, DeepSeekConfig
from models.effy import Effy, EffyConfig
from models.memora import Memora, MemoraConfig
from models.oneira import Oneira, OneiraConfig


# ---------------------------------------------------------------------------
# Registre des modèles — un seul entraîné par invocation (cf. --model)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "gpt2":     lambda args: GPT2(GPT2Config(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
    "cortex":   lambda args: Cortex(CortexConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
    "deepseek": lambda args: DeepSeek(DeepSeekConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
    "effy":     lambda args: Effy(EffyConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
    "memora":   lambda args: Memora(MemoraConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
    "oneira":   lambda args: Oneira(OneiraConfig(vocab_size=50257, dropout=0.1, context_len=args.context_len)),
}


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

    NB: si use_compile, `model` est recompilé dans une variable LOCALE (torch.compile ne
    clone pas les paramètres) — l'objet passé par l'appelant reste la référence non-compilée
    et se retrouve entraîné (mêmes tenseurs Parameter) une fois cette fonction retournée ;
    c'est lui que main() sauvegarde ensuite.
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
    """Démarre un run wandb pour ce modèle. `group` relie plusieurs invocations du script
    (une par modèle) en un seul groupe comparable — c'est LE mécanisme de comparaison
    maintenant qu'une seule invocation n'entraîne qu'un modèle (cf. docstring module)."""
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
    # tokens_seen comme axe X → courbes comparables entre modèles à iso-compute
    run.define_metric("tokens_seen")
    run.define_metric("*", step_metric="tokens_seen")
    return run


# ---------------------------------------------------------------------------
# Checkpoint (sauvegarde finale, pas de reprise — cf. train.py pour l'entraînement resumable)
# ---------------------------------------------------------------------------

def save_checkpoint(ckpt_dir: str, group: str, model_name: str,
                     model: LanguageModel, result: TrainResult) -> Path:
    """Écriture atomique (tmp → rename, cf. training/train.py._save) de l'état final du
    modèle entraîné. Pas d'optimiseur/RNG ici : ce n'est pas un checkpoint resumable, juste
    le modèle testé + sa config + son résultat, pour réutilisation (génération, reprise
    manuelle, inspection) après une comparaison."""
    path = Path(ckpt_dir)
    path.mkdir(exist_ok=True)
    out_path = path / f"{group}_{model_name}.pt"
    tmp = out_path.with_suffix(".tmp")
    torch.save({
        "model_name": model_name,
        "model": model.state_dict(),
        "config": model.config,
        "result": asdict(result),
    }, tmp)
    tmp.rename(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entraînement comparatif (un modèle par invocation)")
    parser.add_argument("--model", type=str, default="oneira", choices=sorted(MODEL_REGISTRY),
                         help="Modèle à entraîner cette invocation (défaut: oneira)")
    parser.add_argument("--group", type=str, default=None,
                         help="Groupe wandb partagé entre plusieurs invocations à comparer "
                              "(défaut: timestamp auto — un groupe à lui seul)")
    parser.add_argument("--duration", type=int, default=300, help="Durée en secondes (défaut: 300 = 5min)")
    parser.add_argument("--max-steps", type=int, default=None, help="Entraîner sur N steps au lieu d'une durée (override --duration)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="memora-vs-gpt2")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder le modèle entraîné")
    args = parser.parse_args()
    use_wandb = not args.no_wandb
    group = args.group or time.strftime("%Y%m%d_%H%M%S")

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

    model = MODEL_REGISTRY[args.model](args)
    model_name = type(model).__name__

    duration = None if args.max_steps is not None else args.duration

    run = init_wandb(args.wandb_project, model_name, model, args, device, group) if use_wandb else None
    result = train_model(
        model, model_name, train_loader, val_loader,
        duration_seconds=duration, lr=args.lr,
        grad_accum_steps=args.grad_accum, device=device,
        log_interval=args.log_interval, max_steps=args.max_steps,
        wandb_run=run, use_compile=not args.no_compile,
    )
    if run is not None:
        run.finish()

    print(f"  Steps          : {result.steps:,}")
    print(f"  Tokens vus     : {result.tokens_seen:,}")
    print(f"  Val loss       : {result.val_loss:.4f}")
    print(f"  Val perplexity : {result.val_perplexity:.2f}")

    if not args.no_save:
        ckpt_path = save_checkpoint(args.checkpoint_dir, group, model_name, model, result)
        print(f"  Checkpoint     : {ckpt_path}")

    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    out = asdict(result)
    out["params_M"] = round(model.num_params() / 1e6, 1)
    fname = results_path / f"{group}_{model_name}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats sauvés dans {fname}")


if __name__ == "__main__":
    main()
