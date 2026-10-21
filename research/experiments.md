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

### Pre-gating (2026-10-21)

Predict layer L+h's experts by applying layer L+h's router (`ffn_gate_inp`) to layer L's output residual (`l_out-L`), RMS-normalized with layer L+h's `post_attention_norm`. Data: `data/prompts-e2-hidden.jsonl` (2 prompts per task, 20 total), 5,120 decode tokens with hidden states (`moe-trace --hidden-out`, 842 MB). Command: `python -m moe_anywhere.pregate Sage-A1.5-Q4_K_M.gguf traces/e2-hidden.tsv traces/e2.hidden --budgets 8,16,32` (41 s).

Recall of the 8 real experts when prefetching `budget` candidates:

| budget | L+1 | L+2 | L+3 |
|---|---|---|---|
| 8 | 0.824 | 0.736 | 0.685 |
| 16 | 0.959 | 0.898 | 0.854 |
| 32 | 0.988 | 0.959 | 0.935 |

Against LRU/prev_token (0.364 at budget 8), pre-gating is 2.3x better with the same budget. Prefetching 16 experts one layer ahead covers 96% of what the next layer uses; two layers ahead, 32 candidates still cover 96%, which leaves a whole layer of compute to hide the copy.

## E3

Online VRAM expert cache + async prefetch vs E0.
Status: E3a (simulation) done 2026-10-21; E3b (llama.cpp implementation) planned

### E3a — simulated hybrid cache

Measured cost model (llama-bench tg128, RTX 5050, `-ot` moving whole layers' experts to CUDA0):

| layers with experts on GPU | 0 | 4 | 8 | 12 |
|---|---|---|---|---|
| tok/s | 41.91 | 44.30 | 47.39 | 50.27 |

Each layer moved saves 0.33 ms/token, so routed experts on CPU cost about 13.2 ms of the 23.9 ms token; the rest (10.6 ms) is the all-GPU ceiling of about 94 tok/s. Copying one expert RAM to VRAM costs about 0.17 ms (2.04 MB over PCIe 4.0 x8), four times its CPU compute (0.041 ms), so fetching every expert per token is slower than the CPU and a cache only pays off when copies stay hidden under compute.

`python -m moe_anywhere.simulate Sage-A1.5-Q4_K_M.gguf traces/e2-hidden.tsv traces/e2.hidden` (hits on GPU, misses on CPU, copies overlapped with compute; 2,560 tokens evaluated, first 2,560 used to build the static cache):

| experts/layer | policy | hit rate | copies/token | predicted tok/s |
|---|---|---|---|---|
| 32 | static | 0.310 | 0 | 50.7 |
| 32 | lru | 0.614 | 123.6 | 46.5 |
| 32 | pregate b8 | 0.880 | 122.8 | 47.3 |
| 64 | static | 0.496 | 0 | 57.9 |
| 64 | lru | 0.772 | 72.8 | 67.2 |
| **64** | **pregate b8** | **0.912** | **72.1** | **70.6** |
| 64 | pregate b16 | 0.957 | 185.3 | 31.7 |
| 128 | lru | 0.919 | 25.8 | 85.2 |
| 128 | pregate b8 | 0.947 | 25.5 | 87.4 |

- On the RTX 5050 (64 experts/layer, about 5.2 GB), pre-gating with 8 candidates predicts 70.6 tok/s, 1.68x the 42 tok/s baseline.
- Budget 16 raises the hit rate but copies stop fitting under compute and tok/s collapses: the PCIe link, not prediction quality, is the limit on this card.
- 128 experts/layer (about 10.5 GB) does not fit 8 GB; on a 24 GB RTX 3090 it predicts 87 tok/s, but the cost model there must be re-measured.
- Caveat: the model assumes GPU and CPU expert work add up sequentially and that copies overlap perfectly; both need checking in E3b.
