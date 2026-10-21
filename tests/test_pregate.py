import numpy as np

from moe_anywhere.pregate import evaluate_pregate, rms_norm, router_topk

N_EMBD, N_EXP, K = 16, 32, 4


def test_rms_norm_unit_rms():
    x = np.random.default_rng(0).normal(size=(5, N_EMBD)) * 7
    y = rms_norm(x, np.ones(N_EMBD))
    assert np.allclose(np.sqrt((y**2).mean(axis=1)), 1.0, atol=1e-4)


def test_router_topk_picks_largest_logits():
    router = np.zeros((N_EXP, N_EMBD), np.float32)
    router[[3, 9, 20, 31], 0] = [4.0, 3.0, 2.0, 1.0]
    x = np.zeros((1, N_EMBD), np.float32)
    x[0, 0] = 1.0
    assert router_topk(x, router, np.ones(N_EMBD), 4)[0].tolist() == [3, 9, 20, 31]


def test_perfect_when_residual_drives_next_router():
    """If layer L+1's experts are exactly its router's top-k on l_out-L, recall is 1."""
    rng = np.random.default_rng(1)
    routers = {l: (rng.normal(size=(N_EXP, N_EMBD)).astype(np.float32), np.ones(N_EMBD, np.float32)) for l in range(4)}
    hidden, experts = {}, {}
    for t in range(20):
        key = (0, t)
        hidden[key] = {l: rng.normal(size=N_EMBD).astype(np.float16) for l in range(4)}
        experts[key] = {
            l: set(router_topk(hidden[key][l - 1][None].astype(np.float32), *routers[l], K)[0].tolist())
            for l in range(1, 4)
        }
    r = evaluate_pregate(routers, hidden, experts, k=K, horizons=(1,), budgets=(K,))
    assert r[(1, K)] == 1.0
