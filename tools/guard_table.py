"""E4.3: fallback-guard verdicts (moe_anywhere.timeline.choose_path) on the machines measured in E3e/E3f/E3g.

No GPU, GGUF or trace file is needed: expert sizes and counts come from the GGUF facts in HANDOFF.md and
research/nemotron-bolt.md, and the routing comes from a synthetic decode trace with a Zipf hot core
(fixed seed, so the table is reproducible). Each row replays the SAME steps through run() twice --
upstream's upload path and the hybrid's CPU path -- with that machine's calibrated costs, feeds the
per-path misses per token into choose_path, and prints the guard's verdict next to the replay's own
comparison and the measured tok/s. t_fixed shifts both replays equally and the guard ignores it, so the
verdicts do not depend on it (omarchy's was never fitted).

Usage: PYTHONPATH=. python tools/guard_table.py [--tokens 400] [--zipf 1.1]
"""
import argparse
import random

from moe_anywhere.timeline import choose_path, run

# Sage-A1.5 Q4_K_M: 40 layers, 256 experts each, top-8, two ffn_down layouts (test_slot_counts_match_upstream_e3b).
Q6, Q4 = 2_039_808, 1_769_472
Q6_LAYERS = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39}


def sage():
    layer_bytes = {l: Q6 if l in Q6_LAYERS else Q4 for l in range(40)}
    return layer_bytes, 256, 8


def bolt():
    # Nemotron 3.5 Bolt 24GB: 40 MoE layers of 240-496 experts, widths 1536-2560, IQ2_XXS up+down in the
    # 1024 latent (~2.0625 bpw) -> 0.81-1.35 MB per expert; same size can carry different expert counts.
    counts, widths = [240, 304, 368, 432, 496], [1536, 2048, 2304, 2560]
    layer_bytes, n_expert = {}, {}
    for l in range(40):
        n_expert[l] = counts[l % len(counts)]
        layer_bytes[l] = int(2 * widths[(l * 7) % len(widths)] * 1024 * 2.0625 / 8)
    return layer_bytes, n_expert, 9


MODELS = {"Sage-A1.5": sage, "Bolt-24GB": bolt}

MACHINES = {
    # E3e/E3g: 7800X3D + RTX 5050, PCIe 4.0 x8. E3f: Ryzen 5 3500X + RTX 3090, PCIe 3.0 x16.
    "omarchy-white": dict(t_fixed=10.25, t_hop=0.09, t_cpu_expert=0.032, ms_per_mib=0.076),
    "omarchy": dict(t_fixed=10.25, t_hop=0.09, t_cpu_expert=0.115, ms_per_mib=0.079),
}

# Measured decode tok/s (hybrid, upstream) for the rows the lab actually ran (E3e/E3f/E3g).
MEASURED = {
    ("omarchy-white", "Sage-A1.5", 2048): (53.3, 35.6),
    ("omarchy-white", "Sage-A1.5", 5120): (61.5, 56.7),
    ("omarchy-white", "Bolt-24GB", 2048): (29.7, 28.3),
    ("omarchy", "Sage-A1.5", 2048): (22.7, 39.0),
    ("omarchy", "Sage-A1.5", 5120): (32.5, 68.6),
    ("omarchy", "Bolt-24GB", 2048): (18.5, 40.6),
    ("omarchy", "Bolt-24GB", 5120): (21.4, 52.0),
}


def trace(n_tokens, layer_bytes, n_expert, top_k, zipf, seed):
    """Decode steps with a per-layer Zipf hot core over a per-layer permutation of the experts."""
    rng = random.Random(seed)
    layers = sorted(layer_bytes)
    perms = {l: rng.sample(range(n_expert[l] if isinstance(n_expert, dict) else n_expert),
                           n_expert[l] if isinstance(n_expert, dict) else n_expert) for l in layers}
    weights = {l: [1.0 / (r + 1) ** zipf for r in range(len(perms[l]))] for l in layers}

    def pick(l, k):
        ids = set()
        while len(ids) < k:
            ids.add(rng.choices(perms[l], weights[l])[0])
        return sorted(ids)

    first = {l: pick(l, 14) for l in layers}  # one short prefill, like the bench prompts
    steps = [("prefill", 14, (0, "prefill"), first)]
    steps += [("decode", 1, (0, pos), {l: pick(l, top_k) for l in layers}) for pos in range(n_tokens)]
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--zipf", type=float, default=1.1)
    args = ap.parse_args(argv)

    print(f"# {args.tokens} decode tokens, Zipf {args.zipf}; costs in ms; 'measured' is E3e/E3f/E3g decode tok/s")
    for machine, costs in MACHINES.items():
        for model, build in MODELS.items():
            layer_bytes, n_expert, top_k = build()
            steps = trace(args.tokens, layer_bytes, n_expert, top_k, args.zipf,
                          seed=sum(map(ord, model + machine)))
            layers = sorted(layer_bytes)
            # the guard needs one expert size: host-bytes-weighted mean (1.816 MiB for Sage)
            total_b = sum(layer_bytes[l] * (n_expert[l] if isinstance(n_expert, dict) else n_expert) for l in layers)
            total_n = sum(n_expert[l] if isinstance(n_expert, dict) else n_expert for l in layers)
            mean_mib = total_b / total_n / 2**20
            for cache_mib in (2048, 5120):
                kw = dict(steps=steps, layer_bytes=layer_bytes, layers=layers, cache_mib=cache_mib,
                          n_expert=n_expert, **costs)
                hyb, up = run(miss_path="cpu", **kw), run(miss_path="upload", **kw)
                guard = choose_path(len(layers), hyb["misses_per_token"], up["misses_per_token"],
                                    mean_mib, t_hop=costs["t_hop"], t_cpu_expert=costs["t_cpu_expert"],
                                    ms_per_mib=costs["ms_per_mib"])
                replay = "cpu" if hyb["tok_s"] > up["tok_s"] else "upload"
                one_expert = "cpu" if costs["t_cpu_expert"] < mean_mib * costs["ms_per_mib"] else "upload"
                t_hyb_x = len(layers) * costs["t_hop"] + hyb["misses_per_token"] * costs["t_cpu_expert"]
                t_up_x = up["misses_per_token"] * mean_mib * costs["ms_per_mib"]
                meas = MEASURED.get((machine, model, cache_mib))
                verdict = "-"
                if meas:
                    want = "cpu" if meas[0] > meas[1] else "upload"
                    verdict = "OK" if guard == want else f"WRONG(want {want})"
                meas_txt = f"{meas[0]}/{meas[1]}" if meas else "-/-"
                print(f"{machine:<15} {model:<10} {cache_mib:>4} MiB | hyb h={hyb['hit_rate']:.2f} "
                      f"miss={hyb['misses_per_token']:5.1f} {hyb['tok_s']:5.1f} | up h={up['hit_rate']:.2f} "
                      f"miss={up['misses_per_token']:5.1f} {up['tok_s']:5.1f} | expert={mean_mib:.3f} MiB "
                      f"t_hyb+={t_hyb_x:5.2f} t_up+={t_up_x:5.2f} ({t_hyb_x / t_up_x:4.2f}x) | "
                      f"guard={guard:<6} replay={replay:<6} 1-expert-rule={one_expert:<6} {verdict} | "
                      f"measured hyb/up = {meas_txt}")


if __name__ == "__main__":
    main()
