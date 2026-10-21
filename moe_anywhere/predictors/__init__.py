"""S4: expert predictors over routing traces.

Common interface: observe(token_pos, layer, experts) is called in trace
order; predict(layer, horizon, k) foresees the experts of layer + horizon
for the current token, right after layer was observed. A predictor never
sees the future of the trace.
"""
from __future__ import annotations

import numpy as np


class Random:
    """k uniform experts without repetition — the chance baseline."""

    def __init__(self, n_experts: int, seed: int = 0):
        self._n_experts = n_experts
        self._rng = np.random.default_rng(seed)

    def observe(self, token_pos: int, layer: int, experts) -> None:
        pass

    def predict(self, layer: int, horizon: int, k: int) -> set[int]:
        draw = self._rng.choice(self._n_experts, size=k, replace=False)
        return {int(e) for e in draw}


class LRU:
    """The k most recently used experts in the target layer (any earlier token)."""

    def __init__(self):
        self._last_use: dict[int, dict[int, int]] = {}
        self._clock = 0

    def observe(self, token_pos: int, layer: int, experts) -> None:
        per_layer = self._last_use.setdefault(int(layer), {})
        for expert in experts:
            self._clock += 1
            per_layer[int(expert)] = self._clock

    def predict(self, layer: int, horizon: int, k: int) -> set[int]:
        recency = self._last_use.get(int(layer) + int(horizon), {})
        freshest = sorted(recency, key=recency.__getitem__, reverse=True)
        return set(freshest[:k])


class Frequency:
    """The k most frequent experts so far in the target layer."""

    def __init__(self, n_experts: int):
        self._counts: dict[int, dict[int, int]] = {}

    def observe(self, token_pos: int, layer: int, experts) -> None:
        per_layer = self._counts.setdefault(int(layer), {})
        for expert in experts:
            expert = int(expert)
            per_layer[expert] = per_layer.get(expert, 0) + 1

    def predict(self, layer: int, horizon: int, k: int) -> set[int]:
        counts = self._counts.get(int(layer) + int(horizon), {})
        top = sorted(counts, key=lambda e: (-counts[e], e))
        return set(top[:k])


class PrevToken:
    """The experts the previous token selected in the target layer; empty if none."""

    def __init__(self):
        self._last: dict[int, set[int]] = {}

    def observe(self, token_pos: int, layer: int, experts) -> None:
        self._last[int(layer)] = {int(e) for e in experts}

    def predict(self, layer: int, horizon: int, k: int) -> set[int]:
        return set(self._last.get(int(layer) + int(horizon), ()))
