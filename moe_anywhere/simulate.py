"""E3a: simulate a per-layer VRAM expert cache (hits on GPU, misses on CPU) and predict tok/s.

Time model per decoded token, fitted on the RTX 5050 with llama-bench (see research/experiments.md):
  t = t_fixed + misses * t_cpu_expert + max(0, copies * t_copy - (t_fixed + misses * t_cpu_expert))
Cache fills (copies, RAM -> VRAM over PCIe) overlap with compute; only the excess stalls.

Policies, each with `cache` experts per layer:
  static    top-`cache` experts by frequency in the first half of the trace; no updates
  lru       every miss is copied in (LRU eviction)
  pregate   before layer L runs, insert the top-`budget` experts predicted from l_out of L-1
            (pre-gating), then serve L; misses are not copied

Usage:
  python -m moe_anywhere.simulate MODEL.gguf trace.tsv trace.hidden [--cache 32,64,128] [--budget 8,16]
"""
import argparse
from collections import OrderedDict

import numpy as np

from moe_anywhere.pregate import load_routers, read_experts, read_hidden, router_topk

# Fitted on RTX 5050 8 GB (PCIe 4.0 x8) + i9-12900HK, Sage-A1.5 Q4_K_M, llama.cpp 0a1ef30.
T_FIXED_MS = 10.6        # everything except routed experts, with experts resident on GPU
T_CPU_EXPERT_MS = 0.0413  # 13.2 ms of CPU expert work / (40 layers x 8 experts)
T_COPY_MS = 0.17          # 2.04 MB per expert over ~12 GB/s


class LayerCache:
    def __init__(self, size):
        self.size = size
        self.items = OrderedDict()

    def __contains__(self, e):
        return e in self.items

    def touch(self, e):
        self.items.move_to_end(e)

    def insert(self, e):
        """Returns True if a copy was needed."""
        if self.size == 0:
            return False
        if e in self.items:
            self.items.move_to_end(e)
            return False
        if len(self.items) >= self.size:
            self.items.popitem(last=False)
        self.items[e] = None
        return True


def token_ms(misses, copies):
    compute = T_FIXED_MS + misses * T_CPU_EXPERT_MS
    return compute + max(0.0, copies * T_COPY_MS - compute)


def simulate(experts, keys, layers, policy, cache, budget=16, routers=None, hidden=None, warm=None):
    """Returns dict with hit_rate, copies/token and predicted tok/s over `keys`."""
    caches = {l: LayerCache(cache) for l in layers}
    if policy == "static":
        for l in layers:
            counts = np.zeros(256, np.int64)
            for key in warm:
                for e in experts[key][l]:
                    counts[e] += 1
            for e in np.argsort(-counts)[:cache]:
                caches[l].insert(int(e))
    hits = total = 0
    times = []
    copies_all = 0
    for key in keys:
        misses = copies = 0
        for l in layers:
            if policy == "pregate" and l - 1 in hidden.get(key, {}) and l in routers:
                pred = router_topk(hidden[key][l - 1][None].astype(np.float32), *routers[l], budget)[0]
                copies += sum(caches[l].insert(int(e)) for e in pred)
            for e in experts[key][l]:
                total += 1
                if e in caches[l]:
                    hits += 1
                    caches[l].touch(e)
                else:
                    misses += 1
                    if policy == "lru":
                        copies += caches[l].insert(e)
        copies_all += copies
        times.append(token_ms(misses, copies))
    return {
        "hit_rate": hits / total,
        "copies_per_token": copies_all / len(keys),
        "tok_s": 1000.0 / float(np.mean(times)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("tsv")
    ap.add_argument("hidden")
    ap.add_argument("--cache", default="32,64,128")
    ap.add_argument("--budget", default="8,16")
    args = ap.parse_args(argv)

    routers = load_routers(args.gguf)
    hidden = read_hidden(args.hidden)
    experts = read_experts(args.tsv)
    # the GGUF also carries MTP layers that never route during normal decoding
    traced = {l for v in experts.values() for l in v}
    layers = sorted(l for l in routers if l in traced)
    keys = sorted(k for k in experts if len(experts[k]) == len(layers) and k in hidden)
    half = len(keys) // 2
    warm, test = keys[:half], keys[half:]

    print(f"tokens: {len(test)} evaluated (first {half} used to warm static); "
          f"baseline all-CPU experts: {1000 / token_ms(len(layers) * 8, 0):.1f} tok/s, "
          f"all-GPU ceiling: {1000 / T_FIXED_MS:.1f} tok/s")
    for cache in (int(c) for c in args.cache.split(",")):
        rows = [("static", {}), ("lru", {})] + [(f"pregate/b{b}", {"budget": int(b)}) for b in args.budget.split(",")]
        for name, extra in rows:
            policy = name.split("/")[0]
            r = simulate(experts, test, layers, policy, cache, routers=routers, hidden=hidden, warm=warm, **extra)
            print(f"cache={cache:<4} {name:<12} hit={r['hit_rate']:.3f}  copies/tok={r['copies_per_token']:6.1f}  "
                  f"tok/s={r['tok_s']:.1f}")


if __name__ == "__main__":
    main()
