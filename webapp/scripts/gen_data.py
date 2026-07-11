"""
Introspecte les vraies classes de models/ (config par défaut, CPU, pas de forward)
et dump une description JSON statique de chaque architecture pour le visualiseur 3D.

Ne réencode aucune connaissance d'architecture "à la main" : on instancie le modèle
réel et on marche son arbre nn.Module, donc ça reste correct si models/ change.
Lancer depuis la racine du repo : `python webapp/scripts/gen_data.py`.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn  # noqa: E402

from models.gpt2 import GPT2, GPT2Config  # noqa: E402
from models.cortex import Cortex, CortexConfig  # noqa: E402
from models.effy import Effy, EffyConfig  # noqa: E402
from models.deepseek import DeepSeek, DeepSeekConfig  # noqa: E402
from models.memora import Memora, MemoraConfig  # noqa: E402
from models.memora_gla import MemoraGLA, MemoraConfig as MemoraGLAConfig  # noqa: E402
from models.oneira import Oneira, OneiraConfig  # noqa: E402

OUT_PATH = REPO_ROOT / "webapp" / "public" / "data" / "models.json"

DESCRIPTIONS = {
    "gpt2": "Référence GPT-2 Small — attention pleine softmax à chaque couche, "
            "coût quadratique en longueur de séquence. La ligne de base contre laquelle "
            "toutes les architectures sub-quadratiques sont mesurées.",
    "cortex": "Sparse-Cortex — poids ternaires {-1,0,+1} (BitNet b1.58) dans toutes les "
              "projections internes, FFN ReLU² à sparsité émergente. Phase actuelle : "
              "attention pleine ternarisée ; l'hybride GDN+SWA est la prochaine étape "
              "(cf. models/cortex.md).",
    "effy": "Effy — attention linéaire (noyau φ(q)·φ(k), état taille fixe) sur la "
            "majorité des couches, avec un petit nombre de couches de « rappel » en "
            "attention pleine pour le recall exact que le noyau linéaire perd.",
    "deepseek": "DeepSeek(mini) — Hyper-Connections contraintes (mHC, flux résiduels "
                "multiples via matrice doublement stochastique) + attention compressée "
                "hybride CSA (sélection top-k) / HCA (dense sur blocs compressés).",
    "memora": "Memora — hybride attention locale (fenêtre glissante + GQA + RoPE) et "
              "Gated Linear Attention (GLA, état récurrent taille fixe) pour un contexte "
              "non borné à coût linéaire sur les couches GLA.",
    "memora_gla": "Memora-GLA — variante à dominante GLA (10 couches GLA sur 14) avec "
                  "seulement 2 couches d'attention globale et 2 locales pour l'ancrage.",
    "oneira": "Oneira — backbone Memora + tête de simulation « monde » : un opérateur F "
              "par horizon temporel fait évoluer l'état GLA sans texte réel, simulant "
              "plusieurs futurs possibles en parallèle (multivers).",
}

NAMES = {
    "gpt2": "GPT-2 Small", "cortex": "Sparse-Cortex", "effy": "Effy",
    "deepseek": "DeepSeek(mini)", "memora": "Memora", "memora_gla": "Memora-GLA",
    "oneira": "Oneira",
}


def n_params(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def classify_mixer(mixer: nn.Module) -> tuple[str, dict]:
    """Retourne (kind, detail) à partir du sous-module mixer/attn d'un bloc."""
    cls = type(mixer).__name__
    detail = {}
    is_ternary = any(type(m).__name__ == "BitLinear" for m in mixer.modules())
    if is_ternary:
        detail["ternary"] = True

    if cls == "GatedLinearAttention":
        return "gla", detail
    if cls == "LocalAttention":
        window = getattr(mixer, "window", None)
        detail["window"] = window
        return ("global-attention" if window is None else "local-attention"), detail
    if cls == "LinearAttention":
        return "linear-attention", detail
    if cls == "CausalSelfAttention":
        return ("ternary-attention" if is_ternary else "full-attention"), detail
    if cls == "CompressedAttention":
        detail["block"] = getattr(mixer, "block", None)
        detail["topk"] = getattr(mixer, "topk", None) if getattr(mixer, "use_topk", False) else None
        return ("csa" if getattr(mixer, "use_topk", False) else "hca"), detail
    return "unknown", detail


def get_mixer(block: nn.Module) -> nn.Module:
    for attr in ("mixer", "attn"):
        if hasattr(block, attr):
            return getattr(block, attr)
    raise AttributeError(f"pas de mixer/attn trouvé sur {type(block).__name__}")


def describe_blocks(blocks: nn.ModuleList) -> list[dict]:
    out = []
    for i, block in enumerate(blocks):
        mixer = get_mixer(block)
        kind, detail = classify_mixer(mixer)
        mlp = getattr(block, "mlp", None)
        out.append({
            "index": i,
            "kind": kind,
            "mixer_class": type(mixer).__name__,
            "mlp_class": type(mlp).__name__ if mlp is not None else None,
            "params": n_params(block),
            "detail": detail,
        })
    return out


def describe_extra(name: str, mod: nn.Module, kind: str, seen: set[int],
                    detail: dict | None = None) -> dict:
    # dédup par identité de Parameter — sim_head référence le lm_head tied du backbone,
    # le compter tel quel gonflerait l'overhead affiché (cf. docstring Oneira.__init__).
    marginal = 0
    for p in mod.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            marginal += p.numel()
    return {
        "name": name,
        "kind": kind,
        "class": type(mod).__name__,
        "params": marginal,
        "detail": detail or {},
    }


def cfg_summary(cfg) -> dict:
    keys = ("vocab_size", "context_len", "n_layer", "n_head", "n_embd",
            "sliding_window", "recurrent_layers", "global_layers", "recall_layers",
            "csa_layers", "hc_streams", "sim_horizons", "sim_n_layer")
    d = {}
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            d[k] = list(v) if isinstance(v, tuple) else v
    return d


def build_entry(model_id: str, model, cfg, blocks: nn.ModuleList, extras: list[dict] | None = None) -> dict:
    return {
        "id": model_id,
        "name": NAMES[model_id],
        "description": DESCRIPTIONS[model_id],
        "config": cfg_summary(cfg),
        "total_params": model.num_params(),
        "blocks": describe_blocks(blocks),
        "extras": extras or [],
    }


def main():
    entries = []

    m = GPT2(GPT2Config())
    entries.append(build_entry("gpt2", m, m.config, m.blocks))

    m = Cortex(CortexConfig())
    entries.append(build_entry("cortex", m, m.config, m.blocks))

    m = Effy(EffyConfig())
    entries.append(build_entry("effy", m, m.config, m.blocks))

    m = DeepSeek(DeepSeekConfig())
    entries.append(build_entry("deepseek", m, m.config, m.blocks))

    m = Memora(MemoraConfig())
    entries.append(build_entry("memora", m, m.config, m.blocks))

    m = MemoraGLA(MemoraGLAConfig())
    entries.append(build_entry("memora_gla", m, m.config, m.blocks))

    m = Oneira(OneiraConfig())
    seen = {id(p) for p in m.backbone.parameters()}
    extras = [
        describe_extra("sim_head", m.sim_head, "simulation-head", seen,
                        {"sim_n_layer": m.config.sim_n_layer, "sim_chunk": m.config.sim_chunk}),
        describe_extra("action_proj", m.action_proj, "projection", seen,
                        {"action_dim": m.config.action_dim}),
    ]
    for horizon_key, op in m.F.items():
        extras.append(describe_extra(f"F[{horizon_key}]", op, "world-operator", seen,
                                      {"horizon_chunks": int(horizon_key)}))
    entries.append(build_entry("oneira", m, m.config, m.backbone.blocks, extras))

    for e in entries:
        assert len(e["blocks"]) == e["config"]["n_layer"], f"{e['id']}: bloc count mismatch"
        assert e["total_params"] > 0
        block_sum = sum(b["params"] for b in e["blocks"])
        extra_sum = sum(x["params"] for x in e["extras"])
        assert block_sum + extra_sum <= e["total_params"], (
            f"{e['id']}: blocs+extras ({block_sum + extra_sum}) dépasse total "
            f"({e['total_params']}) — dédup de paramètres partagés cassé"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"models": entries}, indent=2))
    print(f"{len(entries)} modèles écrits → {OUT_PATH}")
    for e in entries:
        print(f"  {e['id']:12s} {e['total_params']/1e6:6.1f}M  "
              f"{len(e['blocks'])} blocs  +{len(e['extras'])} extras")


if __name__ == "__main__":
    main()
