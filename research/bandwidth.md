# Bandwidth

## Tiers

<!-- presets table (S5) -->

## Method

Decoding a token reads the active weights from wherever they live. Bandwidth-bound tok/s = 1 / (sum over tiers of bytes_per_token_in_tier / tier_bandwidth + hops × latency per hop). It is an upper bound: compute, kernel launch and scheduling overheads come on top.
