# Open questions

## Questions

- How predictable is layer L+1 from layer L on real traces?
- How much VRAM cache gives 90% hit rate?
- Does expert popularity shift by task?
- Can prefetch hide inter-node latency on a consumer cluster?
- Can the CPU and GPU halves of a hybrid MoE layer run concurrently (max instead of sum)?
- How much do RAM-to-VRAM copies slow the CPU experts that read the same RAM?
- Where does pre-gating pay off: bigger experts, slower links (PCIe 3.0 x4 risers), or a cache much smaller than the working set?
- Should llama.cpp's cache skip ubatches of 9-32 tokens? Short prompts get 2.7x slower prefill with it (E3b).
