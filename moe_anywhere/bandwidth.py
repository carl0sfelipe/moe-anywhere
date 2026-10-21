"""Upper bound on decode tok/s when reading active weights is the bottleneck."""
import argparse
from dataclasses import dataclass

# GB/s
PRESETS = {
    "pcie3_x4": 3.5,
    "pcie3_x16": 14.0,
    "ddr4_dual": 35.0,
    "ddr4_quad": 60.0,
    "nvme_gen3": 3.0,
    "gbe1": 0.117,
    "ib40": 4.0,
}


@dataclass
class Tier:
    name: str
    bandwidth_gbs: float


def tokens_per_sec(bytes_per_token, tiers, hops=0, hop_latency_ms=0.0):
    seconds = sum(b / (tiers[name].bandwidth_gbs * 1e9) for name, b in bytes_per_token.items())
    seconds += hops * hop_latency_ms / 1000
    if seconds <= 0:
        raise ValueError("nothing to read and no hops: tok/s is undefined")
    return 1 / seconds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bytes", action="append", default=[], metavar="PRESET=BYTES",
                    help=f"bytes read per token from a tier; presets: {', '.join(PRESETS)}")
    ap.add_argument("--hops", type=int, default=0)
    ap.add_argument("--hop-latency-ms", type=float, default=0.0)
    args = ap.parse_args(argv)
    per_token = {}
    for item in args.bytes:
        name, value = item.split("=", 1)
        per_token[name] = per_token.get(name, 0.0) + float(value)
    tiers = {name: Tier(name, PRESETS[name]) for name in per_token}
    print(f"tok/s: {tokens_per_sec(per_token, tiers, args.hops, args.hop_latency_ms):.2f}")


if __name__ == "__main__":
    main()
