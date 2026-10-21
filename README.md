# moe-anywhere

**Run giant MoE models on the GPU you already have.**

Mixture-of-Experts models activate only a small slice of their weights per token, yet every engine still asks you to hold all of them in fast memory. moe-anywhere is an open lab for expert offloading, caching and **prefetching** (predicting which experts the next layers will use and loading them before they are needed) on consumer GPUs, from a single 8 GB card to a home cluster of RTX 3090s.

## Sponsor

moe-anywhere is sponsored by [bestmodel.run](https://bestmodel.run) — see what actually runs on your machine. Every result here is submitted to the bestmodel.run pool.

## Highlights

Measured on Sage-A1.5 (Qwen3.6-35B-A3B, 256 experts/layer, top-8) with an RTX 5050 8 GB, all in [research/experiments.md](research/experiments.md):

- **Pre-gating works:** running layer L+1's router on layer L's output recalls 82% of the next layer's experts (96% with 16 candidates), against 36% for LRU.
- **A small cache goes far:** 64 of 256 experts per layer in VRAM (about 4.9 GB) catch 79% of selections with plain LRU.
- **Hybrid MoE cache in llama.cpp:** experts missing from VRAM run on the CPU while a paced background uploader fills the cache. On the RTX 5050 it reaches 55.1 tok/s against 45.5 without cache and 50.9 for llama.cpp's own `--moe-cache-mib` (which uploads every miss while the GPU waits). At a 2 GB cache it beats upstream by 45% (44.2 vs 30.4), and prefill keeps full speed. The same idea is in the draft llama.cpp PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861). Our patch is in [patches/](patches/), and a KL-divergence test checks its output against the CPU path (E3d).
- **Prediction is not the bottleneck here:** a trace-driven timeline, validated against llama.cpp to within 2%, shows that on this PCIe link the miss path and the copy engine decide the speed. Pre-gating adds +17% on upstream's design and under 1 tok/s on the hybrid (E3c).
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
| Sage-A1.5 (35B-A3B) | Q4_K_M | RTX 5050 8 GB + 31 GB RAM | + `--moe-cache-hybrid` (E3d patch) | 55.1 | 2026-10-21 |

Every row will be reproducible and submitted to bestmodel.run.

## Roadmap

- **E0 — baseline:** llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
- **E1 — routing traces:** record (layer, token, top-k experts) and publish them as an open dataset.
- **E2 — predictors offline on CPU:** random, LRU, frequency, previous-layer copy, pre-gating; recall@k at horizons 1/2/3.
- **E3 — online VRAM expert cache + async prefetch,** compared against E0: upstream cache measured (E3b), timeline validated (E3c), hybrid CPU/GPU miss path in llama.cpp (E3d).

## Try the hybrid cache

```
cd llama.cpp && git checkout 0a1ef30 && git apply ../moe-anywhere/patches/llama.cpp-0a1ef30-moe-cache-hybrid.patch
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
build/bin/llama-cli -m MODEL.gguf -ngl 99 -ot exps=CPU --load-mode none --moe-cache-mib 5120 --moe-cache-hybrid
```

Pick `--moe-cache-mib` from the VRAM left after the dense weights and KV cache, and pin `-c` (llama.cpp otherwise shrinks the context to fit the cache). Please send your tok/s with and without the flag.

## Contributing

Start with [research/README.md](research/README.md): add a paper, add a predictor, or submit a run.

## License

Apache-2.0 — see [LICENSE](LICENSE) and https://www.apache.org/licenses/LICENSE-2.0.
