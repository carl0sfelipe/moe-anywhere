# moe-anywhere

**Run giant MoE models on the GPU you already have.**

Mixture-of-Experts models activate only a small slice of their weights per token, yet every engine still asks you to hold all of them in fast memory. moe-anywhere is an open lab for expert offloading, caching and **prefetching** (predicting which experts the next layers will use and loading them before they are needed) on consumer GPUs, from a single 8 GB card to a home cluster of RTX 3090s.

## Sponsor

moe-anywhere is sponsored by [bestmodel.run](https://bestmodel.run) — see what actually runs on your machine. Every result here is submitted to the bestmodel.run pool.

## Highlights

Measured on Sage-A1.5 (Qwen3.6-35B-A3B, 256 experts/layer, top-8) with an RTX 5050 8 GB, all in [research/experiments.md](research/experiments.md):

- **Pre-gating works:** running layer L+1's router on layer L's output recalls 82% of the next layer's experts (96% with 16 candidates), against 36% for LRU.
- **A small cache goes far:** 64 of 256 experts per layer in VRAM (about 4.9 GB) catch 79% of selections with plain LRU.
- **The miss path matters more than prediction:** llama.cpp's own expert cache (`--moe-cache-mib`) uploads every miss while the GPU waits, so it only beats experts-on-CPU above about 4.5 GB (49.1 vs 44.7 tok/s at 5 GB). A timeline validated against it to within 2% predicts +17% from pre-gating on that design. Running misses on the CPU and filling the cache in the background predicts 72 tok/s (+61%), and implementing that hybrid in llama.cpp is next (E3d).
- **Open data:** routing traces and hidden states on Hugging Face: [carlosfelipe/moe-anywhere-routing-traces](https://huggingface.co/datasets/carlosfelipe/moe-anywhere-routing-traces).

## Problem

- Large MoEs (hundreds of billions of parameters) do not fit in consumer VRAM, even when only a few billion parameters are active per token.
- The official engines target data-center or newest-generation GPUs and do not offload experts.
- Community projects cache and prefetch experts on one device, but nobody predicts experts across several consumer GPUs and nodes, and there is no open dataset of routing traces to compare predictors on.

## Results

| Model | Quant | Hardware | Technique | tok/s | Date |
|---|---|---|---|---|---|
| Sage-A1.5 (35B-A3B) | Q4_K_M | RTX 5050 8 GB + 31 GB RAM | llama.cpp `-ot exps=CPU` (E0 baseline, llama-bench tg128) | 41.05 | 2026-10-21 |
| Sage-A1.5 (35B-A3B) | Q4_K_M | RTX 5050 8 GB + 31 GB RAM | llama.cpp `-ot exps=CPU`, `tools/bench_decode.sh` | 44.7 | 2026-10-21 |
| Sage-A1.5 (35B-A3B) | Q4_K_M | RTX 5050 8 GB + 31 GB RAM | + `--moe-cache-mib 5120 --load-mode none` (E3b) | 52.1 | 2026-10-21 |

Every row will be reproducible and submitted to bestmodel.run.

## Roadmap

- **E0 — baseline:** llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
- **E1 — routing traces:** record (layer, token, top-k experts) and publish them as an open dataset.
- **E2 — predictors offline on CPU:** random, LRU, frequency, previous-layer copy, pre-gating; recall@k at horizons 1/2/3.
- **E3 — online VRAM expert cache + async prefetch,** compared against E0: upstream cache measured (E3b), timeline validated (E3c), hybrid CPU/GPU miss path in llama.cpp (E3d).

## Contributing

Start with [research/README.md](research/README.md): add a paper, add a predictor, or submit a run.

## License

Apache-2.0 — see [LICENSE](LICENSE) and https://www.apache.org/licenses/LICENSE-2.0.
