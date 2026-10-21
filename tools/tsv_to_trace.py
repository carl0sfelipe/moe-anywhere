"""Convert a moe-trace TSV (tools/trace_hook) into the Parquet trace format (moe_anywhere.trace).

Usage: python tools/tsv_to_trace.py traces/run.tsv traces/run.parquet [--phase all|prefill|decode]

The trace format is token-major with every layer present for every token. In prefill
the last layer only computes the batch's last token, so tokens missing a layer are
dropped (their count is printed).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_tsv(path, phase="all"):
    rows = defaultdict(dict)  # (prompt_id, token_pos) -> layer -> (experts, weights)
    task_of = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if phase != "all" and r["phase"] != phase:
                continue
            key = (int(r["prompt_id"]), int(r["token_pos"]))
            rows[key][int(r["layer"])] = (
                [int(x) for x in r["experts"].split(",")],
                [float(x) for x in r["weights"].split(",")],
            )
            task_of[key[0]] = r["task"]
    return rows, task_of


def tsv_to_trace(path, phase="all"):
    """Returns (trace dict, number of tokens dropped for missing layers)."""
    rows, task_of = read_tsv(path, phase)
    if not rows:
        raise ValueError(f"no rows in {path} for phase={phase}")
    n_layers = max(len(layers) for layers in rows.values())
    keys = sorted(k for k, layers in rows.items() if len(layers) == n_layers)
    dropped = len(rows) - len(keys)
    k = len(next(iter(rows.values()))[0][0])
    n = len(keys) * n_layers
    trace = {
        "prompt_id": np.empty(n, np.int32),
        "task": [],
        "token_pos": np.empty(n, np.int32),
        "layer": np.empty(n, np.int16),
        "experts": np.empty((n, k), np.int16),
        "weights": np.empty((n, k), np.float16),
    }
    i = 0
    for prompt_id, pos in keys:
        layers = rows[(prompt_id, pos)]
        for layer in sorted(layers):
            experts, weights = layers[layer]
            trace["prompt_id"][i] = prompt_id
            trace["task"].append(task_of[prompt_id])
            trace["token_pos"][i] = pos
            trace["layer"][i] = layer
            trace["experts"][i] = experts
            trace["weights"][i] = weights
            i += 1
    return trace, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tsv")
    ap.add_argument("parquet")
    ap.add_argument("--phase", choices=["all", "prefill", "decode"], default="all")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from moe_anywhere.trace import write_trace

    trace, dropped = tsv_to_trace(args.tsv, args.phase)
    write_trace(trace, args.parquet)
    n_tokens = len(trace["layer"]) // (int(trace["layer"].max()) + 1)
    print(f"{n_tokens} tokens x {int(trace['layer'].max()) + 1} layers -> {args.parquet} "
          f"(dropped {dropped} tokens missing a layer)")


if __name__ == "__main__":
    main()
