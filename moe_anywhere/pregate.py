"""Pre-gating: predict layer L+h experts by running layer L+h's router on layer L's residual.

When layer L finishes, its output (l_out-L) is the residual stream that layer L+1 will see,
minus L+1's attention update. Normalizing it with layer L+h's post-attention RMSNorm and
applying layer L+h's router gives a guess at L+h's experts before L+h runs.

Usage:
  python -m moe_anywhere.pregate MODEL.gguf trace.tsv trace.hidden [--k 8] [--budgets 8,16,32]
"""
import argparse
import csv
import sys

import numpy as np

LLAMA_GGUF_PY = "/home/carl0sfelipe/src/llama.cpp/gguf-py"


def rms_norm(x, weight, eps=1e-6):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def router_topk(hidden, router, norm_weight, budget, eps=1e-6):
    """Top-`budget` expert ids for each row of `hidden` ([n, n_embd]) under one layer's router."""
    logits = rms_norm(hidden, norm_weight, eps) @ router.T  # [n, n_expert]
    return np.argsort(-logits, axis=1)[:, :budget]


def load_routers(gguf_path):
    """{layer: (router [n_expert, n_embd] f32, post_attention_norm [n_embd] f32)}."""
    sys.path.insert(0, LLAMA_GGUF_PY)
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    gate, norm = {}, {}
    for t in reader.tensors:
        parts = t.name.split(".")
        if len(parts) < 3 or parts[0] != "blk":
            continue
        layer = int(parts[1])
        if t.name.endswith("ffn_gate_inp.weight"):
            n_embd, n_expert = (int(d) for d in t.shape)
            gate[layer] = np.asarray(t.data, np.float32).reshape(n_expert, n_embd)
        elif t.name.endswith("post_attention_norm.weight"):
            norm[layer] = np.asarray(t.data, np.float32).reshape(-1)
    return {l: (gate[l], norm[l]) for l in gate if l in norm}


def read_hidden(path):
    """{(prompt_id, token_pos): {layer: l_out f16}} from a moe-trace --hidden-out file."""
    data = open(path, "rb").read()
    out, off = {}, 0
    while off < len(data):
        prompt_id, pos, layer, n_embd = np.frombuffer(data, np.int32, 4, off)
        off += 16
        out.setdefault((int(prompt_id), int(pos)), {})[int(layer)] = np.frombuffer(data, np.float16, n_embd, off)
        off += 2 * int(n_embd)
    return out


def read_experts(tsv_path):
    """{(prompt_id, token_pos): {layer: set(experts)}} for decode rows."""
    out = {}
    with open(tsv_path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["phase"] == "decode":
                key = (int(r["prompt_id"]), int(r["token_pos"]))
                out.setdefault(key, {})[int(r["layer"])] = {int(x) for x in r["experts"].split(",")}
    return out


def evaluate_pregate(routers, hidden, experts, k=8, horizons=(1, 2, 3), budgets=(8, 16, 32)):
    """{(horizon, budget): mean |predicted ∩ actual| / k} over tokens with both data."""
    keys = sorted(set(hidden) & set(experts))
    layers = sorted(routers)
    result = {}
    for h in horizons:
        for budget in budgets:
            hits, n = 0, 0
            for src in layers:
                dst = src + h
                if dst not in routers:
                    continue
                rows = [key for key in keys if src in hidden[key] and dst in experts[key]]
                if not rows:
                    continue
                x = np.stack([hidden[key][src] for key in rows])
                pred = router_topk(x, *routers[dst], budget)
                for key, p in zip(rows, pred):
                    hits += len(experts[key][dst] & set(p.tolist()))
                    n += 1
            result[(h, budget)] = hits / (n * k) if n else float("nan")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("tsv")
    ap.add_argument("hidden")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budgets", default="8,16,32")
    args = ap.parse_args(argv)
    budgets = tuple(int(b) for b in args.budgets.split(","))
    r = evaluate_pregate(load_routers(args.gguf), read_hidden(args.hidden), read_experts(args.tsv),
                         k=args.k, budgets=budgets)
    for budget in budgets:
        cols = "  ".join(f"recall@h{h}={r[(h, budget)]:.4f}" for h in (1, 2, 3))
        print(f"pregate budget={budget:<3} {cols}")


if __name__ == "__main__":
    main()
