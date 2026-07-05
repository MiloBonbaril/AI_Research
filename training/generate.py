"""
Script de génération de texte — compare plusieurs modèles avec métriques d'inférence.

Pour Cortex, les poids BitLinear sont packés en ternaire 2 bits avant la génération
(convert_to_inference) : c'est le chemin de déploiement réel, pas le maître fp32.

Usage :
    python -m training.generate                          # Cortex (packé) + GPT-2 from scratch
    python -m training.generate --pretrained             # GPT-2 avec poids HuggingFace
    python -m training.generate --prompt "Once upon"
    python -m training.generate --tokens 200
"""

from __future__ import annotations

import argparse
import time

import torch
from tiktoken import get_encoding

from models.cortex import Cortex, CortexConfig
from models.gpt2 import GPT2, GPT2Config
from models.model_interface import LanguageModel
from training.bitlinear_deploy import convert_to_inference, report_memory


def generate_with_metrics(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> tuple[str, float, float | None]:
    """
    Génère du texte et retourne (texte_complet, tok/s, VRAM_peak_MB ou None).
    Utilise torch.inference_mode() pour désactiver autograd et le suivi de version.
    """
    enc = get_encoding("gpt2")
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    model.to(device).eval()

    if device == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    if device == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tok_per_sec = max_new_tokens / elapsed
    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else None
    return enc.decode(output[0].tolist()), tok_per_sec, vram_mb


def _print_result(name: str, text: str, tok_per_sec: float, vram_mb: float | None):
    sep = "=" * 64
    metrics = f"{tok_per_sec:.1f} tok/s"
    if vram_mb is not None:
        metrics += f"  |  VRAM peak: {vram_mb:.0f} MB"
    print(f"\n{sep}\n  {name}  —  {metrics}\n{sep}")
    print(text)
    print(sep)


def run_cortex(args, device: str):
    model = Cortex(CortexConfig(vocab_size=50257))
    model.eval()

    print("\n--- Cortex : poids avant packing ---")
    report_memory(model, "avant packing")

    convert_to_inference(model)

    print("\n--- Cortex : poids après packing (2 bits/poids) ---")
    report_memory(model, "après packing")

    text, tok_per_sec, vram_mb = generate_with_metrics(
        model, args.prompt, args.tokens, args.temperature, args.top_k, device
    )
    _print_result("Cortex (packé)", text, tok_per_sec, vram_mb)
    model.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()


def run_gpt2(args, device: str):
    model = (GPT2.from_pretrained("gpt2") if args.pretrained else GPT2())
    name = "GPT-2 (pretrained)" if args.pretrained else "GPT-2"

    text, tok_per_sec, vram_mb = generate_with_metrics(
        model, args.prompt, args.tokens, args.temperature, args.top_k, device
    )
    _print_result(name, text, tok_per_sec, vram_mb)
    model.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Génération de texte — Cortex vs GPT-2")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--pretrained", action="store_true", help="Charger les poids HuggingFace pour GPT-2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    run_cortex(args, device)
    run_gpt2(args, device)


if __name__ == "__main__":
    main()
