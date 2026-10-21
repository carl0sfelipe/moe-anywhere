# moe-anywhere

**Run giant MoE models on the GPU you already have.**

Mixture-of-Experts models activate only a small slice of their weights per token, yet every engine still asks you to hold all of them in fast memory. moe-anywhere is an open lab for expert offloading, caching and **prefetching** (predicting which experts the next layers will use and loading them before they are needed) on consumer GPUs, from a single 8 GB card to a home cluster of RTX 3090s.

## Sponsor

moe-anywhere is sponsored by [bestmodel.run](https://bestmodel.run) — see what actually runs on your machine. Every result here is submitted to the bestmodel.run pool.

## Problem

- Large MoEs (hundreds of billions of parameters) do not fit in consumer VRAM, even when only a few billion parameters are active per token.
- The official engines target data-center or newest-generation GPUs and do not offload experts.
- Community projects cache and prefetch experts on one device, but nobody predicts experts across several consumer GPUs and nodes, and there is no open dataset of routing traces to compare predictors on.

## Results

| Model | Quant | Hardware | Technique | tok/s | Date |
|---|---|---|---|---|---|

No results yet. Every row will be reproducible and submitted to bestmodel.run.

## Roadmap

- **E0 — baseline:** llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
- **E1 — routing traces:** record (layer, token, top-k experts) and publish them as an open dataset.
- **E2 — predictors offline on CPU:** random, LRU, frequency, previous-layer copy, pre-gating; recall@k at horizons 1/2/3.
- **E3 — online VRAM expert cache + async prefetch,** compared against E0.

## Contributing

Start with [research/README.md](research/README.md): add a paper, add a predictor, or submit a run.

## License

Apache-2.0 — see [LICENSE](LICENSE) and https://www.apache.org/licenses/LICENSE-2.0.
