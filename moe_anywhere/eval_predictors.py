"""S4: evaluate expert predictors and LRU cache hit rate over a routing trace.

CLI: python -m moe_anywhere.eval_predictors <trace.parquet> --k K
         [--cache-sizes 8,32,64]
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

from moe_anywhere import trace as T
from moe_anywhere.predictors import Frequency, LRU, PrevToken, Random

PREDICTOR_ORDER = ("random", "lru", "frequency", "prev_token")
DEFAULT_HORIZONS = (1, 2, 3)


def evaluate(
    trace: dict,
    factory: Callable[[], object],
    k: int,
    horizons: Iterable[int] = (1, 2, 3),
) -> dict[int, float]:
    """Mean per-horizon recall |predict ∩ truth| / k of one predictor.

    A fresh predictor (factory()) walks the trace rows in order; right after
    observing (token, layer) it predicts layer + horizon for the same token
    and is scored against the trace's experts there. Positions where the
    predictor has no history and returns no experts are unscored (nothing
    was predicted, so there is nothing to credit) — e.g. the first token.
    """
    horizons = tuple(horizons)
    layers = trace["layer"]
    token_pos = trace["token_pos"]
    prompt_id = trace["prompt_id"]
    experts = trace["experts"]
    n_layers = int(layers.max()) + 1
    row_index = {
        (int(prompt_id[i]), int(token_pos[i]), int(layers[i])): i
        for i in range(len(layers))
    }
    predictor = factory()
    hits: dict[int, float] = {h: 0.0 for h in horizons}
    scored: dict[int, int] = {h: 0 for h in horizons}
    for i in range(len(layers)):
        layer = int(layers[i])
        predictor.observe(int(token_pos[i]), layer, experts[i])
        for horizon in horizons:
            if layer + horizon >= n_layers:
                continue
            target = row_index.get(
                (int(prompt_id[i]), int(token_pos[i]), layer + horizon)
            )
            if target is None:
                continue
            predicted = predictor.predict(layer, horizon, k)
            if not predicted:
                continue
            truth = {int(e) for e in experts[target]}
            hits[horizon] += len(predicted & truth) / k
            scored[horizon] += 1
    return {
        h: (hits[h] / scored[h] if scored[h] else 0.0) for h in horizons
    }


def cache_hit_rate(trace: dict, cache_size_per_layer: int, policy: str = "lru") -> float:
    """Fraction of expert selections already resident in a per-layer cache.

    Each selected expert is a hit if it is in its layer's cache; otherwise a
    miss that inserts it, evicting the least recently used entry when full.
    """
    if policy != "lru":
        raise ValueError(f"unknown cache policy: {policy!r}")
    if cache_size_per_layer <= 0:
        return 0.0
    caches: dict[int, OrderedDict[int, None]] = {}
    hits = 0
    total = 0
    layers = trace["layer"]
    experts = trace["experts"]
    for i in range(len(layers)):
        cache = caches.setdefault(int(layers[i]), OrderedDict())
        for expert in experts[i]:
            expert = int(expert)
            total += 1
            if expert in cache:
                hits += 1
                cache.move_to_end(expert)
            else:
                if len(cache) >= cache_size_per_layer:
                    cache.popitem(last=False)
                cache[expert] = None
    return hits / total if total else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moe_anywhere.eval_predictors",
        description="Evaluate expert predictors and LRU cache hit rate on a routing trace.",
    )
    parser.add_argument("trace", type=Path, help="routing trace in Parquet format (S3)")
    parser.add_argument("--k", type=int, required=True, help="experts to predict")
    parser.add_argument(
        "--cache-sizes",
        default="8,32,64",
        help="comma-separated per-layer cache sizes for hit_rate (default: 8,32,64)",
    )
    args = parser.parse_args(argv)
    trace = T.read_trace(args.trace)
    n_experts = int(trace["experts"].max()) + 1
    factories: dict[str, Callable[[], object]] = {
        "random": lambda: Random(n_experts),
        "lru": LRU,
        "frequency": lambda: Frequency(n_experts),
        "prev_token": PrevToken,
    }
    horizons = DEFAULT_HORIZONS
    for name in PREDICTOR_ORDER:
        recall = evaluate(trace, factories[name], args.k, horizons)
        cells = "  ".join(f"recall@{h}={recall[h]:.4f}" for h in sorted(recall))
        print(f"{name:<12}{cells}")
    print()
    print("hit_rate (lru cache, per layer):")
    sizes = [int(s) for s in args.cache_sizes.split(",") if s.strip()]
    for size in sizes:
        print(f"  size={size:<4}{cache_hit_rate(trace, size):.4f}")


if __name__ == "__main__":
    main()
