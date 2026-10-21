import pytest

from moe_anywhere.simulate import T_FIXED_MS, LayerCache, token_ms


def test_copies_hidden_under_compute_are_free():
    assert token_ms(misses=0, copies=0) == pytest.approx(T_FIXED_MS)
    assert token_ms(misses=0, copies=10) == pytest.approx(T_FIXED_MS)  # 1.7 ms < 10.6 ms


def test_excess_copies_stall():
    assert token_ms(misses=0, copies=100) == pytest.approx(100 * 0.17)


def test_layer_cache_lru_eviction():
    c = LayerCache(2)
    assert c.insert(1) and c.insert(2)
    c.touch(1)
    assert c.insert(3)  # evicts 2
    assert 1 in c and 3 in c and 2 not in c
    assert not c.insert(1)  # already cached: no copy


def test_zero_size_cache_never_copies():
    assert not LayerCache(0).insert(5)
