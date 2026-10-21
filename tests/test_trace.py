"""Oracle for spec S3: routing trace format and synthetic generator."""
import numpy as np

from moe_anywhere import trace as T

N_TOK, N_LAYERS, N_EXP, K = 200, 6, 64, 4


def _synth(seed=0):
    return T.synth_trace(N_TOK, N_LAYERS, N_EXP, K, seed=seed, zipf_s=1.1)


def test_shapes_and_dtypes():
    t = _synth()
    n = N_TOK * N_LAYERS
    assert set(t) == set(T.COLUMNS)
    assert t["experts"].shape == (n, K) and t["experts"].dtype == np.int16
    assert t["weights"].shape == (n, K) and t["weights"].dtype == np.float16
    assert t["layer"].dtype == np.int16 and t["token_pos"].dtype == np.int32
    assert len(t["task"]) == n
    # token-major order: rows [i*L:(i+1)*L] are token i, layers 0..L-1
    assert list(t["layer"][:N_LAYERS]) == list(range(N_LAYERS))
    assert (t["token_pos"][:N_LAYERS] == t["token_pos"][0]).all()


def test_rows_are_valid_topk():
    t = _synth()
    e = t["experts"]
    assert e.min() >= 0 and e.max() < N_EXP
    assert all(len(set(row)) == K for row in e)
    w = t["weights"].astype(np.float32)
    assert (w >= 0).all() and np.allclose(w.sum(axis=1), 1.0, atol=1e-2)


def test_deterministic_by_seed():
    a, b, c = _synth(1), _synth(1), _synth(2)
    assert np.array_equal(a["experts"], b["experts"])
    assert not np.array_equal(a["experts"], c["experts"])


def test_skewed_popularity():
    counts = np.bincount(_synth()["experts"].ravel(), minlength=N_EXP)
    top10 = np.sort(counts)[::-1][: N_EXP // 10].sum()
    assert top10 / counts.sum() > 0.25  # uniform would be ~0.10


def test_temporal_correlation_same_layer():
    """Real traces: consecutive tokens reuse experts within a layer."""
    e = _synth()["experts"].reshape(N_TOK, N_LAYERS, K)
    overlap = np.mean([
        len(set(e[i, l]) & set(e[i + 1, l]))
        for i in range(N_TOK - 1) for l in range(N_LAYERS)
    ])
    assert overlap > 2 * (K * K / N_EXP)  # random expectation is K*K/N_EXP


def test_no_cross_layer_id_correlation():
    """Each layer has its own experts: expert IDs at L and L+1 are unrelated."""
    e = _synth()["experts"].reshape(N_TOK, N_LAYERS, K)
    overlap = np.mean([
        len(set(e[i, l]) & set(e[i, l + 1]))
        for i in range(N_TOK) for l in range(N_LAYERS - 1)
    ])
    assert overlap < 2 * (K * K / N_EXP)


def test_parquet_roundtrip(tmp_path):
    t = _synth()
    p = tmp_path / "t.parquet"
    T.write_trace(t, p)
    r = T.read_trace(p)
    for col in T.COLUMNS:
        if col == "task":
            assert list(r[col]) == list(t[col])
        else:
            assert np.array_equal(r[col], t[col]) and r[col].dtype == t[col].dtype
