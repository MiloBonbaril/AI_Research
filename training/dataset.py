"""
Utilitaires partagés : dataset WikiText-103 et évaluation.

Importé par train.py (entraînement complet Memora) et compare.py
(comparaison Memora vs GPT-2).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _amp(device: str):
    """Autocast bf16 sur CUDA, no-op ailleurs."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class WikiTextDataset(Dataset):
    """
    Wikitext-103 tokenisé en blocs contigus de taille fixe.

    Premier appel : tokenise tout le split et sauve le tensor sur disque.
    Appels suivants : charge directement depuis le cache.
    """

    def __init__(self, split: str = "train", context_len: int = 1024,
                 cache_dir: str = ".cache"):
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
                if row["text"].strip():
                    all_tokens.extend(enc.encode(row["text"]))
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.tokens, cache_path)
            print(f"  {len(self.tokens):,} tokens sauvés → {cache_path}")

        self.n_blocks = len(self.tokens) // (context_len + 1)
        print(f"  {self.n_blocks:,} blocs de {context_len} tokens")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * (self.context_len + 1)
        chunk = self.tokens[start : start + self.context_len + 1]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def evaluate(model, val_loader, device: str, max_batches: int = 100) -> float:
    """Loss moyenne sur le split validation (capped à max_batches batches)."""
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
