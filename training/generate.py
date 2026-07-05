"""
Script de génération de texte — fonctionne avec n'importe quel LanguageModel.

Usage :
    python generate.py                          # GPT-2 from scratch (poids aléatoires)
    python generate.py --pretrained             # GPT-2 avec poids HuggingFace
    python generate.py --prompt "Once upon"     # prompt custom
    python generate.py --tokens 200             # nombre de tokens à générer
"""

import argparse

import torch
from tiktoken import get_encoding

from gpt2 import GPT2
from model_interface import LanguageModel


def generate(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    """Encode le prompt, génère, décode."""
    enc = get_encoding("gpt2")

    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.to(device)
    model.eval()

    output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Génération de texte")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    parser.add_argument("--tokens", type=int, default=100, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--pretrained", action="store_true", help="Charger les poids HuggingFace")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    if args.pretrained:
        model = GPT2.from_pretrained("gpt2")
    else:
        model = GPT2()

    text = generate(model, args.prompt, args.tokens, args.temperature, args.top_k, device)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()

# agy --conversation=275f74b0-eb58-4636-8e92-193b027fd88f