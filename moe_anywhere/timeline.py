"""E3c: per-layer timeline of llama.cpp's MoE cache (`--moe-cache-mib`), with and without pre-gated prefetch.

Replaces the token-level model of simulate.py (E3a), whose overlap assumption E3b disproved.
What upstream llama.cpp (0a1ef30, src/llama-moe-cache.cpp) does, and what this file reproduces:
  - layers whose experts live in host memory are grouped by expert layout (here: ffn_down Q6_K vs Q4_K);
    each group shares ONE segmented LRU (probation + protected halves) over all of its layers' slots
  - at layer L's MoE the scheduler stops (sync point S_L), the host reads the selected experts, the SLRU
    plans hits and misses, and misses are uploaded on the compute stream before the experts run on the GPU
  - ubatches over 32 tokens bypass the cache (read-only)
Fitted on the RTX 5050 (E3b stats): t_token = 10.6 ms + uploaded MiB x 0.101 ms, i.e. no overlap at all.

Prefetch (proposed): at S_L the host also reads layer L+h's pre-gated candidates, computed on the GPU from
attn_residual-L (the residual just before L's MoE), and queues the uncached ones on a copy stream. All
uploads share one PCIe link in FIFO order. A prefetched expert is usable once its copy has landed.

Miss paths:
  upload  upstream: a miss (or an expert whose prefetch has not landed) is uploaded and the GPU waits
  cpu     hybrid (proposed): experts that are not resident run on the CPU, which costs a GPU->CPU->GPU hop
          per layer plus T_CPU_EXPERT_MS each; the cache is filled only by background copies (prefetch and,
          with admit_misses, the misses themselves), so nothing ever waits for PCIe

Usage:
  python -m moe_anywhere.timeline MODEL.gguf trace.tsv [--hidden attn_residual.hidden] [--cache-mib 2048,5120]
      [--budget 0,4,8,16] [--horizon 1]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

# Fitted on RTX 5050 8 GB (PCIe 4.0 x8) + i9-12900HK, Sage-A1.5 Q4_K_M, llama.cpp 0a1ef30, mmap'd weights.
T_FIXED_MS = 10.6        # one decoded token with every routed expert already in VRAM
MS_PER_MIB = 0.101       # RAM -> VRAM upload
T_HEAD_MS = 1.5          # part of T_FIXED outside the repeating layers (output head, embeddings); assumed
ATTN_FRACTION = 0.5      # share of a layer's time before its MoE sync point; assumed
T_CPU_EXPERT_MS = 0.038  # one routed expert on the CPU, decode (E3a: 0.33 ms per CPU layer minus the hop)
T_HOP_MS = 0.025         # GPU -> CPU -> GPU round trip in a layer (E3c: attn_norm moved to the CPU)
MAX_BACKLOG_MS = 1.0     # background copies are dropped while the PCIe queue is longer than this
MAX_BATCH = 32           # upstream: larger ubatches do not use the cache


class SLRU:
    """Port of upstream moe_cache_slru: one segmented LRU over the slots of a group of layers."""

    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.slot_of = {}                 # (layer, expert) -> slot
        self.key_of = [None] * n_slots
        self.prot = [False] * n_slots
        # two segments as ordered dicts of slots: probation (False) and protected (True), LRU first
        self.seg = {False: dict.fromkeys(range(n_slots)), True: {}}

    def __contains__(self, key):
        return key in self.slot_of

    def _move(self, s, protect):
        del self.seg[self.prot[s]][s]
        self.seg[protect][s] = None
        self.prot[s] = protect

    def touch(self, s):
        self._move(s, True)
        if len(self.seg[True]) > self.n_slots // 2:
            self._move(next(iter(self.seg[True])), False)

    def _victim(self, keep):
        for segment in (False, True):
            for s in self.seg[segment]:
                if self.key_of[s] is None or self.key_of[s] not in keep:
                    return s
        raise RuntimeError("no SLRU victim")

    def _fill(self, key, keep):
        s = self._victim(keep)
        if self.key_of[s] is not None:
            del self.slot_of[self.key_of[s]]
        self.key_of[s] = key
        self.slot_of[key] = s
        self._move(s, False)

    def plan(self, layer, ids):
        """Upstream plan(): promote hits, then fill misses in sorted order. Returns (hits, misses)."""
        uniq = list(dict.fromkeys(ids))  # first-occurrence order, as upstream promotes hits
        if len(uniq) > self.n_slots:
            raise RuntimeError("cache too small for one ubatch")
        hits = [e for e in uniq if (layer, e) in self.slot_of]
        for e in hits:
            self.touch(self.slot_of[(layer, e)])
        keep = {(layer, e) for e in uniq}
        misses = sorted(e for e in uniq if (layer, e) not in self.slot_of)
        for e in misses:
            self._fill((layer, e), keep)
        return hits, misses

    def prefetch(self, layer, expert, keep):
        """Insert a predicted expert into probation without evicting `keep`. Returns True if it was copied."""
        if (layer, expert) in self.slot_of:
            return False
        self._fill((layer, expert), keep)
        return True


def expert_bytes(gguf_path):
    """{layer: bytes of one routed expert (gate + up + down)} for layers that have routed experts."""
    from moe_anywhere.pregate import LLAMA_GGUF_PY
    import sys

    sys.path.insert(0, LLAMA_GGUF_PY)
    from gguf import GGUFReader

    out = defaultdict(int)
    for t in GGUFReader(gguf_path).tensors:
        if t.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                            "ffn_gate_up_exps.weight")):
            out[int(t.name.split(".")[1])] += int(t.n_bytes) // int(t.shape[-1])
    return dict(out)


def slot_counts(layer_bytes, layers, cache_mib):
    """Upstream sizing: groups of equal expert size; the budget is split by host bytes, minus a padding slot."""
    groups = defaultdict(list)
    for l in layers:
        groups[layer_bytes[l]].append(l)
    n_expert = 256
    host = {b: b * n_expert * len(ls) for b, ls in groups.items()}
    total = sum(host.values())
    budget = cache_mib * 1024 * 1024
    return {b: (ls, max(0, min(int(budget * host[b] / total) // b - 1, n_expert * len(ls)))) for b, ls in groups.items()}


def read_steps(tsv_path):
    """Ubatches in trace order: [(kind, n_tokens, key, {layer: [ids]})]; one prefill per prompt, one per decode token."""
    steps = []
    cur = None
    with open(tsv_path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pid, pos, layer = int(r["prompt_id"]), int(r["token_pos"]), int(r["layer"])
            ids = [int(x) for x in r["experts"].split(",")]
            key = (pid, "prefill") if r["phase"] == "prefill" else (pid, pos)
            if cur is None or cur[2] != key:
                cur = [r["phase"], set(), key, defaultdict(list)]
                steps.append(cur)
            cur[1].add(pos)
            cur[3][layer].extend(ids)
    return [(kind, len(pos), key, dict(by_layer)) for kind, pos, key, by_layer in steps]


def predictions(routers, hidden, layers, horizon, budget):
    """{(prompt_id, pos, dst_layer): ranked candidate experts} from the hidden state of dst - horizon."""
    from moe_anywhere.pregate import router_topk

    out = {}
    keys = sorted(hidden)
    for dst in layers:
        src = dst - horizon
        rows = [k for k in keys if src in hidden[k]]
        if not rows or dst not in routers:
            continue
        x = np.stack([hidden[k][src] for k in rows]).astype(np.float32)
        for k, p in zip(rows, router_topk(x, *routers[dst], budget)):
            out[(k[0], k[1], dst)] = p.tolist()
    return out


def run(steps, layer_bytes, layers, cache_mib, preds=None, horizon=1, budget=0, wait="needed",
        t_fixed=T_FIXED_MS, ms_per_mib=MS_PER_MIB, t_head=T_HEAD_MS, attn_fraction=ATTN_FRACTION,
        miss_path="upload", admit_misses=True, t_cpu_expert=T_CPU_EXPERT_MS, t_hop=T_HOP_MS,
        max_backlog_ms=MAX_BACKLOG_MS):
    """Replay the trace; returns decode stats.

    wait (upload path): 'needed' waits only for the prefetched experts a layer uses (per-copy events),
    'all' makes the host sync the copy stream at every sync point.
    Background copies (prefetch, admitted misses) are skipped while link - t > max_backlog_ms.
    """
    groups = slot_counts(layer_bytes, layers, cache_mib)
    caches, group_of = {}, {}
    for b, (ls, n_slots) in groups.items():
        cache = SLRU(n_slots) if n_slots >= 8 else None
        for l in ls:
            caches[l] = cache
            group_of[l] = b
    t_layer = (t_fixed - t_head) / len(layers)
    copy_ms = {l: layer_bytes[l] / 2**20 * ms_per_mib for l in layers}

    hits = misses = copies = 0
    t = link = 0.0
    times = []
    ready = {}  # (layer, expert) -> time its background copy lands
    for kind, n_tokens, key, by_layer in steps:
        if kind == "prefill" or n_tokens > 1:
            if n_tokens <= MAX_BATCH:
                for l in layers:
                    if caches[l] is not None and l in by_layer:
                        caches[l].plan(l, by_layer[l])
            continue
        if any(l not in by_layer for l in layers):
            continue
        start = t
        for l in layers:
            cache = caches[l]
            t += attn_fraction * t_layer
            ids = by_layer[l]
            if cache is None:
                misses += len(set(ids))
                t += (1 - attn_fraction) * t_layer + len(set(ids)) * copy_ms[l]
                continue
            keep = {(l, e) for e in ids}
            if miss_path == "cpu":
                uniq = list(dict.fromkeys(ids))
                resident = [e for e in uniq if (l, e) in cache and ready.get((l, e), 0.0) <= t]
                for e in resident:
                    cache.touch(cache.slot_of[(l, e)])
                n_cpu = len(uniq) - len(resident)
                hits += len(resident)
                misses += n_cpu
                t += t_hop + n_cpu * t_cpu_expert
                if admit_misses:
                    for e in uniq:
                        if link - t <= max_backlog_ms and cache.prefetch(l, e, keep):
                            link = max(link, t) + copy_ms[l]
                            ready[(l, e)] = link
                            copies += 1
            else:
                if wait == "all":
                    t = max(t, link)
                h, m = cache.plan(l, ids)
                for e in h:
                    t = max(t, ready.pop((l, e), t))
                hits += len(h)
                misses += len(m)
                if m:
                    t = max(t, link) + len(m) * copy_ms[l]
                    link = t
            dst = l + horizon
            if preds is not None and budget and dst in caches and caches[dst] is not None:
                for e in preds.get((key[0], key[1], dst), [])[:budget]:
                    if link - t <= max_backlog_ms and caches[dst].prefetch(dst, e, keep):
                        link = max(link, t) + copy_ms[dst]
                        ready[(dst, e)] = link
                        copies += 1
            t += (1 - attn_fraction) * t_layer
        t += t_head
        times.append(t - start)
    n = len(times)
    return {
        "tokens": n,
        "hit_rate": hits / max(1, hits + misses),
        "misses_per_token": misses / max(1, n),
        "copies_per_token": copies / max(1, n),
        "ms_per_token": float(np.mean(times)) if n else float("nan"),
        "tok_s": 1000.0 / float(np.mean(times)) if n else float("nan"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("tsv")
    ap.add_argument("--hidden", help="moe-trace --hidden-out file of attn_residual (needed for --budget > 0)")
    ap.add_argument("--cache-mib", default="2048,5120")
    ap.add_argument("--budget", default="0")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--wait", default="needed", choices=["needed", "all"])
    ap.add_argument("--ms-per-mib", type=float, default=MS_PER_MIB)
    ap.add_argument("--miss-path", default="upload", choices=["upload", "cpu"])
    ap.add_argument("--no-admit-misses", action="store_true", help="cpu path: fill the cache only by prefetch")
    ap.add_argument("--max-backlog-ms", type=float, default=MAX_BACKLOG_MS)
    args = ap.parse_args(argv)

    layer_bytes = expert_bytes(args.gguf)
    steps = read_steps(args.tsv)
    traced = {l for _, _, _, by_layer in steps for l in by_layer}
    layers = sorted(l for l in layer_bytes if l in traced)  # the GGUF also has MTP layers that never route
    budgets = [int(b) for b in args.budget.split(",")]
    preds = None
    if any(budgets):
        from moe_anywhere.pregate import load_routers, read_hidden

        preds = predictions(load_routers(args.gguf), read_hidden(args.hidden), layers, args.horizon, max(budgets))
    for cache_mib in (int(c) for c in args.cache_mib.split(",")):
        slots = {b: n for b, (_, n) in slot_counts(layer_bytes, layers, cache_mib).items()}
        for budget in budgets:
            r = run(steps, layer_bytes, layers, cache_mib, preds, args.horizon, budget, args.wait,
                    ms_per_mib=args.ms_per_mib, miss_path=args.miss_path, admit_misses=not args.no_admit_misses,
                    max_backlog_ms=args.max_backlog_ms)
            print(f"{args.miss_path:<6} cache={cache_mib:<5} slots={sorted(slots.values())} budget={budget:<3} h={args.horizon} "
                  f"hit={r['hit_rate']:.4f} misses/tok={r['misses_per_token']:6.1f} "
                  f"copies/tok={r['copies_per_token']:6.1f} tok/s={r['tok_s']:.1f} ({r['tokens']} tokens)")


if __name__ == "__main__":
    main()
