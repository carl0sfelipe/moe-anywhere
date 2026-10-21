# Experiments

## E0

Baseline: llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
Status: planned

## E1

Routing traces: record (layer, token, top-k experts) and publish as an open dataset.
Status: planned

## E2

Predictors offline on CPU: random, LRU, frequency, previous-layer copy, pre-gating; metric recall@k at horizons 1/2/3.
Status: planned

## E3

Online VRAM expert cache + async prefetch vs E0.
Status: planned
