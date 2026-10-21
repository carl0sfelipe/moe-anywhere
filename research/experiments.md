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

Rough VRAM cost: one routed expert is gate + up (2 x 2048 x 512, Q4_K, 4.5 bpw) plus down (512 x 2048), which this GGUF stores as Q6_K in 20 layers (2.04 MB per expert) and Q4_K in the other 20 (1.77 MB). 64 experts/layer x 40 layers is about 4.9 GB and gives 78.5% hits. 128/layer (about 9.8 GB, 92.8%) fits a 24 GB RTX 3090. (Corrected in the E3c review: the first version assumed 2.0 MB for every layer.)

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
Status: E3a superseded; E3b (upstream cache measured) and E3c (validated timeline + hybrid design) done 2026-10-21; E3d (hybrid implementation) next

### E3a — simulated hybrid cache (superseded by E3c)

> Review 2026-10-21: this model adds copy time per token and lets it overlap perfectly with compute, and serves misses on the CPU. llama.cpp's own cache does neither (E3b): misses are uploaded and the GPU waits. The 70.6 tok/s below is therefore not a prediction for pre-gating on llama.cpp. E3c replaces it with a per-layer timeline validated against the real cache.

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

### E3b — upstream llama.cpp `--moe-cache-mib` (2026-10-21)

llama.cpp 0a1ef30 already ships a VRAM expert cache (`src/llama-moe-cache.cpp`). Layers whose experts are in host memory are grouped by expert layout (here two groups of 20 layers: ffn_down Q6_K and Q4_K), and each group shares **one** segmented LRU (probation + protected) over all of its layers' slots. At every MoE layer the scheduler stops at the slot-map copy, the host reads the selected experts, the misses are uploaded on the compute stream, and the experts then run on the GPU. There is no prediction and no CPU path for misses. Ubatches over 32 tokens bypass the cache.

`tools/bench_decode.sh` (Sage-A1.5 Q4_K_M, `-ngl 99 -ot exps=CPU -c 8192 -fit off`, `llama-completion`, one coding prompt, 200 tokens, greedy, 2 interleaved repeats, spread under 1%):

| `--moe-cache-mib` | slots per group (per layer) | hit rate | uploaded MiB/token | decode tok/s | prefill tok/s (14 tok) |
|---|---|---|---|---|---|
| 0 | - | - | - | 44.7 | 85.9 |
| 2048 | 562 (28) | 60.4% | 234 | 29.6 | 31.8 |
| 4096 | 1126 (56) | 78.2% | 130 | 42.1 | 32.3 |
| 5120 | 1408 (70) | 83.7% | 97 | 49.1 | 31.7 |
| 5120, `--load-mode none` (pinned) | 1408 (70) | 83.7% | 97 | 52.1 | 34.4 |

- The three mmap points fit t = 10.9 ms + 0.098 ms per uploaded MiB: the upload never overlaps compute. A cache below about 4.5 GB is slower than no cache.
- Pinned host memory (`--load-mode none`, 18.9 GB of CUDA_Host) is 6% faster; async copies will need it, because cudaMemcpyAsync from pageable memory blocks the host.
- Prefill of short prompts gets 2.7x slower, because a 14-token ubatch goes through the cache and uploads 3.4 GB.
- Without `-c` and `-fit off`, llama.cpp silently shrinks the context so the cache fits (262,144 to 4,096 tokens at 5120 MiB). The first E3b table compared different contexts, and all E3 benchmarks now pin `-c 8192`.

### E3c — validated timeline, and why the miss path matters more than prediction (2026-10-21)

`moe_anywhere/timeline.py` replays a routing trace through a port of upstream's SLRU and a per-layer timeline: a sync point per MoE layer, uploads on one FIFO PCIe link, and prefetches usable only once their copy lands. Validation:

- `tests/test_timeline.py` compiles upstream's `moe_cache_slru` straight from `llama-moe-cache.cpp` and checks that the port gives the same hits, misses and slot per fill on 600 random ubatches. It also checks the slot counts (562 / 1408) against the E3b logs.
- Replaying a trace of the E3b prompt (`data/prompts-e3-bench.jsonl`) gives hit rates of 60.2% / 83.0% and 29.2 / 48.1 tok/s, against 60.4% / 83.7% and 29.6 / 49.1 measured.

Pre-gating input: at layer L's sync point the available residual is `attn_residual-L` (before L's MoE), not `l_out-L`. With `moe-trace --hidden-tensor attn_residual` on the E2 prompts (same routing as E2, byte for byte), the recall of layer L+h's experts is:

| budget | L+1 | L+2 | L+3 |
|---|---|---|---|
| 8 | 0.788 | 0.717 | 0.673 |
| 16 | 0.939 | 0.883 | 0.842 |
| 32 | 0.980 | 0.951 | 0.926 |

One GPU→CPU→GPU round trip inside a layer costs about 0.02-0.03 ms. Measured with llama-bench tg128 by moving `attn_norm` (39.24 to 37.73 tok/s) and then `post_attention_norm` (36.75) to the CPU. That puts one CPU expert at about (0.33 - 0.025) / 8 = 0.038 ms, five times cheaper than uploading it (0.19 ms).

Timeline results on the 20 E2 prompts (5,120 decode tokens, `python -m moe_anywhere.timeline MODEL traces/e2-res.tsv --hidden traces/e2-res.hidden ...`):

| design, 5120 MiB | hit rate | tok/s |
|---|---|---|
| upstream (misses uploaded, GPU waits) | 0.818 | 46.7 |
| upstream + pre-gating, budget 8, horizon 1, copy backlog capped at 0.3 ms | 0.906 | 54.6 |
| hybrid: misses run on the CPU, copied in the background for later tokens | 0.812 | 72.0 |
| hybrid + pre-gating, budget 8 | 0.823 | 72.7 |

| cache MiB | 2048 | 3072 | 4096 | 5120 | 6144 |
|---|---|---|---|---|---|
| upstream | 29.3 | 34.8 | 40.7 | 46.7 | 52.7 |
| hybrid | 60.2 | 64.7 | 68.6 | 72.0 | 74.9 |

- On this card the PCIe link is the limit: pre-gating fixes a third of the misses but every fix is a 0.19 ms upload, so it buys +17% on upstream and almost nothing once misses are cheap.
- The hybrid wins at every size and is never slower than no cache (44.7). It stays ahead in the pessimistic corner of the sensitivity sweep (CPU expert 0.08 ms, hop 0.1 ms: 51.5 tok/s).
- Background copies need a cap on the queue. Without one, large budgets flood the link, everything is in flight, and the hit rate collapses (for example 0.24 at budget 16).
- Caveats: the model serializes the CPU and GPU parts of a layer, DMA from RAM may slow the CPU experts, and the attention/MoE split of a layer and the 1.5 ms head are assumed.

Design for E3d (hybrid in `llama_moe_cache`):

1. `ggml-cpu` `mul_mat_id`: a negative expert id skips that row and writes zeros. This is the only ggml change.
2. Graph: in small ubatches the MoE runs twice and the results are added. The GPU part uses cache slots, with non-resident experts mapped to an all-zero slot. The CPU part uses host experts, with resident experts mapped to -1.
3. The cache plans without synchronous uploads. It fills slots from a second backend instance (its own stream), from pinned memory, with a host-side backlog cap. A filled expert becomes resident only after its event completes, and the current layer's experts are never evicted.
4. Pre-gating becomes an optional admission policy on top. On this card it is worth less than 1 tok/s, but on bigger models or slower links the balance must be re-measured.
