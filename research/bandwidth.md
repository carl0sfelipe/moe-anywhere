# Bandwidth

## Tiers

| preset | GB/s |
|---|---|
| pcie3_x4 | 3.5 |
| pcie3_x16 | 14.0 |
| ddr4_dual | 35.0 |
| ddr4_quad | 60.0 |
| nvme_gen3 | 3.0 |
| gbe1 | 0.117 |
| ib40 | 4.0 |

## Method

Decoding a token reads the active weights from wherever they live. Bandwidth-bound tok/s = 1 / (sum over tiers of bytes_per_token_in_tier / tier_bandwidth + hops × latency per hop). It is an upper bound: compute, kernel launch and scheduling overheads come on top.
