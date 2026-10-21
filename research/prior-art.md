# Prior art

## Engines

### DeepInfer
DeepSeek's open-source engine (Apache-2.0); SM100 / Ascend 950 only; includes EPLB. https://github.com/deepseek-ai/DeepInfer

### cachedMoE
Vulkan/Slang engine for DeepSeek-V4.1-Flash; LRU expert cache streamed from NVMe; no prediction; bridged into DeepInfer by PR 142. https://github.com/acupof-ai/cachedMoE and https://github.com/deepseek-ai/DeepInfer/pull/142

### Flare
NVIDIA local inference server, SM120/SM121 only; KV/prefix-cache offload but no expert offload. https://github.com/NVIDIA/flare-local-inference

### llama.cpp
`-ot exps=CPU` keeps routed experts in RAM and computes them on CPU. https://github.com/ggml-org/llama.cpp

### KTransformers
CPU/GPU hybrid MoE inference.

### Unsloth
Dynamic low-bit GGUF quants for large MoEs.

## Prefetch projects

### llama-moe-cache
Expert cache + predictive prefetch for llama.cpp. https://github.com/ongunm/llama-moe-cache

### moe-prefetch
Learned per-layer prefetch heads for SSD-streamed experts. https://github.com/DimitriChrysafis/moe-prefetch

### vLLM
RFC for GPU expert cache + async pipeline. https://github.com/vllm-project/vllm/issues/38256

### FreeToken
Router lookahead prefetch of next-layer experts. https://github.com/FlashML-org/FreeToken/issues/677

### moe-offload
SSD + per-layer LRU expert cache for Apple Silicon. https://github.com/Jakevin/moe-offload

## llama.cpp expert caches

Found in the E3 review (2026-10-21). Our E3d hybrid is the same idea as #27861; what moe-anywhere adds is the trace-driven timeline, the open traces and the measurements behind each design choice.

### --moe-cache-mib (merged)
GPU LRU cache for host-resident experts: misses are uploaded before the layer runs, ubatches up to 32 tokens, single device. It is the cache measured in E3b. https://github.com/ggml-org/llama.cpp/pull/29887

### Hybrid hit/miss cache (draft PR)
Cached experts run in a second `mul_mat_id` on the GPU and uncached ones on the CPU, and the outputs are summed. A worker thread uploads in the background and per-step inserts are capped. Reported results: +31% on 2x RTX 3090 (Qwen3.8-Flash-Next), and slowdowns on cards with little VRAM. The thread documents that the batched CUDA kernels require distinct expert ids per token, so a shared zero slot crashes MMQ; E3d hit exactly this. https://github.com/ggml-org/llama.cpp/pull/27861

### Persistent expert slot pool (RFC)
Synchronous LRU slot pool, the same design as the merged cache: +84% decode on an RTX 4090 with 64 slots/layer. Testers report losses at small pools and give the break-even rule hit rate >= 1 - PCIe_BW/RAM_BW. https://github.com/ggml-org/llama.cpp/discussions/28248

### Other proposals
An RFC for hybrid hit/miss execution inside the CPU `mul_mat_id` (https://github.com/ggml-org/llama.cpp/discussions/24528), a feature request for a two-tier GPU+RAM cache with pluggable eviction (https://github.com/ggml-org/llama.cpp/issues/20757), and HybridMoE, a Vulkan fork that maps missing experts to a zero slot and runs them on the CPU (https://github.com/MohammadR3za/HybridMoE).

## Gap

The hybrid miss path exists in a draft llama.cpp PR, but nobody publishes routing traces or a timeline model that predicts which cache design pays off on which hardware. None of these caches predicts experts, and none spans several consumer GPUs and nodes.
