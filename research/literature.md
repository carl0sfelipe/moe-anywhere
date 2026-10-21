# Literature

## Prefetching

### Pre-gated MoE
Uses the current layer's router to predict next-layer experts.

### MoE-Infinity
Activation history drives expert prefetch and caching.

### ProMoE
Predictive expert prefetch with caching.

### AdapMoE
Adaptive gating with prefetch and caching.

### SeqMoE
Predictive, graph-compatible offloading; prefill prefetch by profiled frequency. Link: https://arxiv.org/pdf/2609.12978

### SpecPrefetch
Parameter-efficient expert prefetching for sparse MoE. Link: https://arxiv.org/pdf/2607.24787

## Offloading

### HOBBIT
Loads less important experts in lower precision to save bandwidth.

### Fiddler
Computes offloaded experts on the CPU instead of copying them to the GPU.

### MoE-Lightning
CPU-GPU pipelined MoE inference on memory-constrained GPUs.

## Placement

### EPLB
DeepSeek's expert-parallel load balancer: counts expert use, replicates hot experts, reshuffles placement.
