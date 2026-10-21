"""S3: routing trace format + synthetic generator.

A trace is a dict with exactly the COLUMNS keys, N rows (one per
(token, layer)), in token-major order: rows i*L..i*L+L-1 hold token i,
layers 0..L-1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ("prompt_id", "task", "token_pos", "layer", "experts", "weights")

# Weight of the layer-local Zipf ranking in synth_trace's popularity mixture.
# Purely layer-local rankings flatten the whole-trace distribution (top-10%
# of experts hold only ~24% of selections), while a purely global ranking
# makes every layer route to the same hot experts (adjacent-layer overlap
# ~0.97). The mixture keeps the whole-trace skew strong while selection sets
# stay layer-local: adjacent-layer overlap ~0.4, below the 2x chance bound
# (real Sage-A1.5 trace measured 0.235, i.e. at chance with uniform ids).
_LOCAL_POPULARITY_WEIGHT = 0.6


def synth_trace(
    n_tokens: int,
    n_layers: int,
    n_experts: int,
    k: int,
    seed: int = 0,
    zipf_s: float = 1.1,
) -> dict:
    """Deterministic synthetic trace (numpy Generator seeded by `seed`).

    Each layer's expert popularity is Zipf-skewed with exponent `zipf_s`,
    mixing a global id permutation with one owned by the layer
    (`_LOCAL_POPULARITY_WEIGHT`), so selection sets stay layer-local while
    whole-trace popularity remains skewed. Temporal correlation lives only
    within a layer: token i+1 inherits each expert of token i (same layer)
    with prob 0.5 and draws the rest from the layer's Zipf without
    repetition. Ids are never correlated across layers beyond the shared
    popularity prior.
    """
    rng = np.random.default_rng(seed)
    experts = np.zeros((n_tokens, n_layers, k), dtype=np.int16)
    weights = np.zeros((n_tokens, n_layers, k), dtype=np.float16)
    ids = np.arange(n_experts)

    def zipf_over(permutation: np.ndarray) -> np.ndarray:
        popularity = 1.0 / (permutation.astype(np.float64) + 1.0) ** zipf_s
        return popularity / popularity.sum()

    global_pop = zipf_over(rng.permutation(n_experts))
    for layer in range(n_layers):
        local_pop = zipf_over(rng.permutation(n_experts))
        popularity = (
            (1.0 - _LOCAL_POPULARITY_WEIGHT) * global_pop
            + _LOCAL_POPULARITY_WEIGHT * local_pop
        )
        prev: np.ndarray | None = None
        for token in range(n_tokens):
            if prev is None:
                chosen = rng.choice(n_experts, size=k, replace=False, p=popularity)
            else:
                inherited = np.unique(prev[rng.random(k) < 0.5])
                rest = np.setdiff1d(ids, inherited)
                need = k - len(inherited)
                if need:
                    fill = rng.choice(
                        rest,
                        size=need,
                        replace=False,
                        p=popularity[rest] / popularity[rest].sum(),
                    )
                    chosen = np.concatenate([inherited, fill])
                else:
                    chosen = inherited
            chosen = np.sort(chosen)
            experts[token, layer] = chosen
            row_w = popularity[chosen]
            weights[token, layer] = row_w / row_w.sum()
            prev = chosen
    n = n_tokens * n_layers
    return {
        "prompt_id": np.zeros(n, dtype=np.int32),
        "task": ["synthetic"] * n,
        "token_pos": np.repeat(np.arange(n_tokens, dtype=np.int32), n_layers),
        "layer": np.tile(np.arange(n_layers, dtype=np.int16), n_tokens),
        "experts": experts.reshape(n, k),
        "weights": weights.reshape(n, k),
    }


def write_trace(trace: dict, path: str | Path) -> None:
    """Write a trace to Parquet (experts/weights as fixed-size list columns)."""
    experts = np.asarray(trace["experts"])
    weights = np.asarray(trace["weights"])
    table = pa.table(
        {
            "prompt_id": pa.array(np.asarray(trace["prompt_id"]), type=pa.int32()),
            "task": pa.array(trace["task"], type=pa.string()),
            "token_pos": pa.array(np.asarray(trace["token_pos"]), type=pa.int32()),
            "layer": pa.array(np.asarray(trace["layer"]), type=pa.int16()),
            "experts": pa.FixedSizeListArray.from_arrays(
                pa.array(experts.reshape(-1), type=pa.int16()), experts.shape[1]
            ),
            "weights": pa.FixedSizeListArray.from_arrays(
                pa.array(weights.reshape(-1), type=pa.float16()), weights.shape[1]
            ),
        }
    )
    pq.write_table(table, path)


def read_trace(path: str | Path) -> dict:
    """Read a trace back with the exact dtypes and shapes it was written with."""
    table = pq.read_table(path).combine_chunks()
    k = table.schema.field("experts").type.list_size

    def column(name: str) -> pa.Array:
        col = table.column(name).combine_chunks()
        if isinstance(col, pa.ChunkedArray):
            col = col.chunks[0] if len(col.chunks) == 1 else pa.concat_arrays(col.chunks)
        return col

    def scalar(name: str, dtype) -> np.ndarray:
        return column(name).to_numpy(zero_copy_only=False).astype(dtype, copy=False)

    def fixed(name: str, dtype) -> np.ndarray:
        values = column(name).flatten().to_numpy(zero_copy_only=False)
        return values.astype(dtype, copy=False).reshape(-1, k)

    return {
        "prompt_id": scalar("prompt_id", np.int32),
        "task": table.column("task").to_pylist(),
        "token_pos": scalar("token_pos", np.int32),
        "layer": scalar("layer", np.int16),
        "experts": fixed("experts", np.int16),
        "weights": fixed("weights", np.float16),
    }
