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

## Gap

No one does predictive expert prefetch distributed across multiple consumer GPUs and nodes, and there is no open dataset of routing traces.
