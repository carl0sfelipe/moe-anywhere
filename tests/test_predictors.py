"""Oracle for spec S4: expert predictors and evaluator."""
import subprocess
import sys

import numpy as np
import pytest

from moe_anywhere import trace as T
from moe_anywhere.eval_predictors import cache_hit_rate, evaluate
from moe_anywhere.predictors import Frequency, LRU, PrevToken, Random

N_TOK, N_LAYERS, N_EXP, K = 300, 6, 64, 4


@pytest.fixture(scope="module")
def synth():
    return T.synth_trace(N_TOK, N_LAYERS, N_EXP, K, seed=0, zipf_s=1.1)


def _constant_trace():
    """Every layer of every token picks experts 0..K-1: repeating the last token is perfect."""
    n = N_TOK * N_LAYERS
    return {
        "prompt_id": np.zeros(n, np.int32),
        "task": ["x"] * n,
        "token_pos": np.repeat(np.arange(N_TOK, dtype=np.int32), N_LAYERS),
        "layer": np.tile(np.arange(N_LAYERS, dtype=np.int16), N_TOK),
        "experts": np.tile(np.arange(K, dtype=np.int16), (n, 1)),
        "weights": np.full((n, K), 1 / K, np.float16),
    }


def test_prev_token_perfect_on_constant_trace():
    r = evaluate(_constant_trace(), lambda: PrevToken(), k=K, horizons=(1, 2, 3))
    assert r == {1: pytest.approx(1.0), 2: pytest.approx(1.0), 3: pytest.approx(1.0)}


def test_random_is_near_chance(synth):
    r = evaluate(synth, lambda: Random(N_EXP, seed=0), k=K, horizons=(1,))
    assert r[1] == pytest.approx(K / N_EXP, abs=0.03)


def test_learned_baselines_beat_random(synth):
    rand = evaluate(synth, lambda: Random(N_EXP, seed=0), k=K, horizons=(1,))[1]
    for factory in (lambda: Frequency(N_EXP), lambda: LRU(), lambda: PrevToken()):
        assert evaluate(synth, factory, k=K, horizons=(1,))[1] > rand


def test_predictions_have_size_k(synth):
    p = Frequency(N_EXP)
    p.observe(0, 0, [1, 2, 3, 4])
    assert len(p.predict(layer=0, horizon=1, k=K)) <= K


def test_cache_hit_rate_bounds(synth):
    assert cache_hit_rate(synth, cache_size_per_layer=0) == 0.0
    total = synth["experts"].size
    distinct = len({(int(l), int(e)) for l, row in zip(synth["layer"], synth["experts"]) for e in row})
    full = cache_hit_rate(synth, cache_size_per_layer=N_EXP)
    assert full == pytest.approx(1 - distinct / total)
    mid = cache_hit_rate(synth, cache_size_per_layer=N_EXP // 4)
    assert 0.0 < mid < full


def test_cli(tmp_path, synth):
    p = tmp_path / "t.parquet"
    T.write_trace(synth, p)
    out = subprocess.run(
        [sys.executable, "-m", "moe_anywhere.eval_predictors", str(p), "--k", str(K)],
        capture_output=True, text=True, check=True,
    ).stdout
    for name in ("random", "frequency", "lru", "prev_token"):
        assert name in out
    assert "recall@" in out and "hit_rate" in out
