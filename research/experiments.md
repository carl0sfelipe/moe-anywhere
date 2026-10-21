# Experiments

## E0

Baseline: llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
Status: done (2026-10-21)

- Machine: RTX 5050 8 GB (SM120, driver 610.57.04), 31 GB RAM.
- Engine: llama.cpp build 0a1ef30, CUDA, `-DCMAKE_CUDA_ARCHITECTURES=120`.
- Command: `llama-bench -m Sage-A1.5-Q4_K_M.gguf -ngl 99 -ot "exps=CPU" -p 512 -n 128 -r 2`
- Result: pp512 304.23 ± 3.08 tok/s, tg128 41.05 ± 0.27 tok/s (model 20.21 GiB, 35.51B params).

## E1

Routing traces: record (layer, token, top-k experts) and publish as an open dataset.
Status: in progress

- Hook: `tools/trace_hook/moe-trace` (llama.cpp eval callback on `ffn_moe_topk-N` and `ffn_moe_weights_norm-N`). Writes TSV: prompt_id, task, token_pos, phase, layer, experts, weights.
- Smoke run (2026-10-21, Sage-A1.5 Q4_K_M, RTX 5050, `-ngl 99 -ot exps=CPU`, 2 prompts, 32 generated tokens each): 4278 rows, 0 invalid (8 distinct experts in [0, 256), weights sum to 1).
- Prefill caveat: the last layer only computes the output row, so prefill tokens other than the last have 39 of 40 layers.
- First observation: on real traces, expert IDs at layer L and L+1 overlap at chance (0.235 vs 64/256 = 0.25), because each layer has its own experts. Copying IDs across layers predicts nothing; cross-layer predictors need the hidden state or learned per-layer mappings.
- Full run (2026-10-21): `data/prompts-e1.jsonl` (74 prompts, 10 tasks), 256 generated tokens max, same machine/flags, 8 min 41 s. 831,778 TSV rows (93 MB); 19,018 complete tokens x 40 layers = 760,720 rows, 0 invalid; 1,822 prefill tokens dropped for the missing last layer.
- Same-layer locality: consecutive tokens share 2.86 of 8 experts on average (random 0.25); varies by layer from 0.41 to 3.74.
- Skew: per layer, the top 32 / 64 / 128 of 256 experts take 37% / 58% / 82% of selections (uniform: 12.5% / 25% / 50%). Only 0.5 experts per layer were never used.
- Task specificity (layer 20): the top-32 experts of two different tasks overlap 36%, so a cache tuned for one task partly misses on another.

## E2

Predictors offline on CPU: random, LRU, frequency, previous-layer copy, pre-gating; metric recall@k at horizons 1/2/3.
Status: first baselines done (2026-10-21), pre-gating pending

On the E1 trace (19,018 tokens x 40 layers, k=8): `python -m moe_anywhere.eval_predictors traces/e1-sage-a1.5.parquet --k 8 --cache-sizes 16,32,64,128,192` (2 min on CPU).

| predictor | recall@1 | recall@2 | recall@3 |
|---|---|---|---|
| random | 0.031 | 0.031 | 0.031 |
| frequency | 0.140 | 0.141 | 0.142 |
| lru | 0.364 | 0.370 | 0.373 |
| prev_token | 0.364 | 0.370 | 0.373 |

LRU and prev_token coincide at k=8: the 8 most recent experts of a layer are the previous token's. Running frequency trails because popularity shifts by task.

LRU cache per layer, hit rate by experts kept per layer (of 256):

| experts/layer | 16 | 32 | 64 | 128 | 192 |
|---|---|---|---|---|---|
| hit rate | 0.469 | 0.623 | 0.785 | 0.928 | 0.982 |

Rough VRAM cost: one routed expert is gate + up (2 x 2048 x 512, Q4_K ~4.5 bpw) plus down (512 x 2048, Q6_K ~6.56 bpw), about 2.0 MB. 64 experts/layer x 40 layers is about 5.2 GB and gives 78.5% hits, which fits next to the 2.1 GB the RTX 5050 already uses in E0. 128/layer (about 10.5 GB, 92.8%) fits a 24 GB RTX 3090.

## E3

Online VRAM expert cache + async prefetch vs E0.
Status: planned
