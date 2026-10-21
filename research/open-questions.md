# Open questions

## Questions

- How predictable is layer L+1 from layer L on real traces?
- How much VRAM cache gives 90% hit rate?
- Does expert popularity shift by task?
- Can prefetch hide inter-node latency on a consumer cluster?
- Can the CPU and GPU halves of a hybrid MoE layer run concurrently (max instead of sum)? E3d measured 0.09 ms of fixed cost per layer for the serialized CPU chain.
- Multi-token hybrid ubatches (speculative decoding) need distinct zero slots per token, or CUDA kernels that skip negative ids.
- How much do RAM-to-VRAM copies slow the CPU experts that read the same RAM?
- Where does pre-gating pay off: bigger experts, slower links (PCIe 3.0 x4 risers), or a cache much smaller than the working set?
- Should llama.cpp's cache skip ubatches of 9-32 tokens? Short prompts get 2.7x slower prefill with it (E3b).

## Candidates for the next wave (absorbed from the E4 plan draft)

- Cache-size curves 2-6 GiB for upstream and hybrid on each machine, to map break-even and the useful operating region.
- Uploader pacing sweep (chunk size, in-flight copies, queue cap) per machine; the E3d values were tuned on the first machine only.
- Bypass the cache for short prefill ubatches (9-32 tokens) in upstream mode: prefill is 2.7x slower with it (E3b), decode should not notice.
- Adaptive prefetch budget driven by the copy backlog, and cost-aware admission (admit only when expected reuse beats the upload cost). Test in `timeline.py` first.
- Robustness across prompt types (code, math, chat, long context): do hit rate and the hybrid's lead hold in every bucket?
- Stop/go rule before C++ work: the timeline must replay upstream within 2% and the hybrid within 5% on the target machine (met on both machines so far, E3e).
