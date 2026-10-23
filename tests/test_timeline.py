import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from moe_anywhere.timeline import SLRU, choose_path, run, slot_counts

UPSTREAM = Path(os.environ.get("LLAMA_DIR", os.path.expanduser("~/src/llama.cpp"))) / "src" / "llama-moe-cache.cpp"

HARNESS = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define GGML_ASSERT(x) do { if (!(x)) abort(); } while (0)
#define GGML_ABORT(...) abort()
%s
int main() {
    int n_layer, n_expert, n_slots;
    if (scanf("%%d %%d %%d", &n_layer, &n_expert, &n_slots) != 3) return 1;
    moe_cache_slru c;
    c.init(n_layer, n_expert, n_slots);
    std::vector<std::vector<int32_t>> maps(n_layer, std::vector<int32_t>(n_expert, -1));
    for (int il = 0; il < n_layer; ++il) c.slot_map[il] = maps[il].data();
    int il, k;
    std::vector<moe_cache_slru::fill> fills;
    while (scanf("%%d %%d", &il, &k) == 2) {
        std::vector<int32_t> ids(k);
        for (auto & x : ids) scanf("%%d", &x);
        size_t n_hit = 0;
        c.plan(il, ids.data(), ids.size(), fills, n_hit);
        printf("%%zu", n_hit);
        for (auto & f : fills) printf(" %%d:%%d", f.expert, f.slot);
        printf("\n");
    }
}
"""


def _upstream_slru_source():
    src = UPSTREAM.read_text()
    start = src.index("struct moe_cache_slru {")
    end = src.index("\n};\n", start) + 4
    return src[start:end]


@pytest.mark.skipif(not UPSTREAM.is_file() or shutil.which("c++") is None, reason="needs a llama.cpp checkout and c++")
def test_slru_port_matches_upstream(tmp_path):
    """Same hits, same misses and the same slot for every fill as the C++ SLRU, on random ubatches."""
    exe = tmp_path / "slru"
    cpp = tmp_path / "slru.cpp"
    cpp.write_text(HARNESS % _upstream_slru_source())
    subprocess.run(["c++", "-O1", "-std=c++17", "-o", str(exe), str(cpp)], check=True)

    rng = random.Random(0)
    n_layer, n_expert, n_slots = 3, 32, 20
    hot = list(range(12))
    plans = []
    for _ in range(600):
        il = rng.randrange(n_layer)
        k = rng.choice([4, 4, 4, 12])  # mostly decode-sized, some small multi-token ubatches
        pool = hot if rng.random() < 0.7 else range(n_expert)
        plans.append((il, [rng.choice(pool) for _ in range(k)]))
    stdin = f"{n_layer} {n_expert} {n_slots}\n" + "".join(f"{il} {len(ids)} {' '.join(map(str, ids))}\n" for il, ids in plans)
    ref = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True).stdout.splitlines()

    port = SLRU(n_slots)
    for (il, ids), line in zip(plans, ref, strict=True):
        hits, misses = port.plan(il, ids)
        got = " ".join([str(len(hits))] + [f"{e}:{port.slot_of[(il, e)]}" for e in misses])
        assert got == line


def test_slru_keeps_the_experts_of_the_current_ubatch():
    c = SLRU(4)
    c.plan(0, [0, 1, 2, 3])
    hits, misses = c.plan(0, [0, 1, 4, 5])
    assert hits == [0, 1] and misses == [4, 5]
    assert all((0, e) in c for e in (0, 1, 4, 5))


def test_prefetch_never_evicts_kept_keys():
    c = SLRU(4)
    c.plan(0, [0, 1, 2, 3])
    keep = {(0, 0), (0, 1), (0, 2)}
    assert c.prefetch(1, 7, keep)
    assert all(k in c for k in keep) and (1, 7) in c and (0, 3) not in c
    assert not c.prefetch(1, 7, keep)  # already cached: no copy


def test_slot_counts_match_upstream_e3b():
    """llama.cpp logged 562 and 1408 slots per group for 2048 and 5120 MiB on Sage-A1.5 Q4_K_M."""
    q6, q4 = 2_039_808, 1_769_472  # bytes of one expert with ffn_down in Q6_K / Q4_K
    q6_layers = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39}
    layer_bytes = {l: q6 if l in q6_layers else q4 for l in range(40)}
    for mib, want in ((2048, 562), (5120, 1408)):
        assert sorted(n for _, n in slot_counts(layer_bytes, range(40), mib, 256).values()) == [want, want]


def _steps(tokens):
    return [("decode", 1, (0, pos), {l: ids for l, ids in enumerate(layers)}) for pos, layers in enumerate(tokens)]


def test_no_prefetch_is_fixed_time_plus_serial_uploads():
    layer_bytes = {0: 2**20, 1: 2**20}
    tokens = [[[0, 1], [0, 1]], [[2, 3], [0, 1]]]  # token 1: layer 0 misses 2 and 3, layer 1 hits
    r = run(_steps(tokens), layer_bytes, [0, 1], cache_mib=64, n_expert=256, t_fixed=10.0, ms_per_mib=0.1, t_head=1.0)
    assert r["misses_per_token"] == 3.0
    assert r["ms_per_token"] == pytest.approx(10.0 + 3.0 * 0.1)


def test_perfect_prefetch_hides_copies_under_compute():
    layer_bytes = {0: 2**20, 1: 2**20}
    rng = random.Random(1)
    tokens = [[[rng.randrange(256) for _ in range(8)] for _ in range(2)] for _ in range(50)]
    preds = {(0, pos, 1): list(dict.fromkeys(layers[1])) for pos, layers in enumerate(tokens)}
    kw = dict(layer_bytes=layer_bytes, layers=[0, 1], cache_mib=64, n_expert=256, t_fixed=10.0, ms_per_mib=0.1, t_head=1.0)
    base = run(_steps(tokens), **kw)
    pf = run(_steps(tokens), preds=preds, horizon=1, budget=8, **kw)
    # only layer 1 can be prefetched, and perfectly: its misses vanish, layer 0's stay
    assert pf["misses_per_token"] < 0.6 * base["misses_per_token"]
    assert pf["tok_s"] > base["tok_s"]


# --- E4.3: the fallback guard (choose_path) ---

# Calibrated costs of the two machines that disagreed (E3e/E3f/E3g, ms).
OMARCHY = dict(t_hop=0.09, t_cpu_expert=0.115, ms_per_mib=0.079)        # 3090, PCIe 3.0 x16, Ryzen 5 3500X
OMARCHY_WHITE = dict(t_hop=0.09, t_cpu_expert=0.032, ms_per_mib=0.076)  # 5050, PCIe 4.0 x8, 7800X3D
SAGE_MIB = (20 * 2_039_808 + 20 * 1_769_472) / 40 / 2**20   # 1.816: host-bytes mean of the two layouts
BOLT_MIB = 1.063                                             # IQ2_XXS latent experts, 0.81-1.35 MB


def test_guard_picks_upstream_on_omarchy_and_hybrid_on_omarchy_white():
    """The two-term rule reproduces the measured verdicts on both machines, both models (E4.3)."""
    # misses/token replayed by tools/guard_table.py at the measured cache sizes
    assert choose_path(40, 88.2, 88.2, SAGE_MIB, **OMARCHY) == "upload"        # E3f: 32.5 vs 68.6 tok/s
    assert choose_path(40, 80.0, 80.0, BOLT_MIB, **OMARCHY) == "upload"        # E3f: 21.4 vs 52.0
    assert choose_path(40, 88.8, 88.4, SAGE_MIB, **OMARCHY_WHITE) == "cpu"     # E3e: 61.5 vs 56.7
    assert choose_path(40, 141.5, 141.5, BOLT_MIB, **OMARCHY_WHITE) == "cpu"   # E3g: 29.7 vs 28.3


def test_guard_two_terms_fix_the_omarchy_sage_mispick():
    """Per-expert arithmetic alone picks hybrid here (0.115 < 0.143); the fixed term and miss count don't."""
    assert OMARCHY["t_cpu_expert"] < SAGE_MIB * OMARCHY["ms_per_mib"]  # the one-expert rule says "cpu"...
    assert choose_path(40, 88.2, 88.2, SAGE_MIB, **OMARCHY) == "upload"  # ...and upstream won 2.1x (E3f)


def test_guard_ties_and_near_ties_stay_upstream():
    """Equal costs, or a hybrid win inside the 10% margin, keep llama.cpp's existing path."""
    tie = choose_path(40, 36, 36, 2.0, t_hop=0.09, t_cpu_expert=0.1, ms_per_mib=0.1)    # 7.2 ms both
    near = choose_path(40, 34, 36, 2.0, t_hop=0.09, t_cpu_expert=0.1, ms_per_mib=0.1)   # 7.0 vs 7.2, 3%
    clear = choose_path(40, 20, 36, 2.0, t_hop=0.09, t_cpu_expert=0.1, ms_per_mib=0.1)  # 5.6 vs 7.2
    assert (tie, near, clear) == ("upload", "upload", "cpu")


def test_guard_agrees_with_the_replay_it_is_fed_from():
    """The guard fed an actual run() replay's misses picks the same winner as comparing the two tok/s."""
    rng = random.Random(3)
    layer_bytes = {l: (2_039_808 if l < 20 else 1_769_472) for l in range(40)}
    hot = {l: rng.sample(range(256), 64) for l in range(40)}  # E1: LRU over 64 of 256 experts hits 79%

    def pick(l):
        ids = set()
        while len(ids) < 8:
            ids.add(rng.choice(hot[l]) if rng.random() < 0.7 else rng.randrange(256))
        return sorted(ids)

    steps = [("decode", 1, (0, p), {l: pick(l) for l in range(40)}) for p in range(120)]
    kw = dict(steps=steps, layer_bytes=layer_bytes, layers=list(range(40)), cache_mib=5120,
              n_expert=256, t_fixed=10.25)
    for costs, want in ((OMARCHY, "upload"), (OMARCHY_WHITE, "cpu")):
        hyb, up = run(miss_path="cpu", **kw, **costs), run(miss_path="upload", **kw, **costs)
        assert ("cpu" if hyb["tok_s"] > up["tok_s"] else "upload") == want
        assert choose_path(40, hyb["misses_per_token"], up["misses_per_token"], SAGE_MIB, **costs) == want


def test_hybrid_layer_without_cache_slots_runs_on_the_cpu():
    """A group refused cache slots (< 8) is all misses; in the hybrid they cost the CPU path, not uploads."""
    r = run(_steps([[[3, 7]]]), {0: 2**20}, [0], cache_mib=8, n_expert=256,
            t_fixed=10.0, t_head=1.0, miss_path="cpu", t_hop=0.09, t_cpu_expert=0.1)
    assert r["ms_per_token"] == pytest.approx(10.0 + 0.09 + 2 * 0.1)


def test_slot_counts_follow_each_layers_expert_count():
    """Bolt mixes 240-496 experts per layer: host bytes and the slot cap must use the real counts."""
    b = 2**20
    counts = {0: 240, 1: 496}
    assert slot_counts({0: b, 1: b}, [0, 1], 1024, counts)[b][1] == 736  # capped at 240+496, not 256x2
    assert slot_counts({0: b, 1: b}, [0, 1], 2, counts)[b][1] == 1      # budget // b - 1
