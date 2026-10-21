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

## Testable hypotheses (next wave)

### H9 — adaptive prefetch budget by link backlog

- Hypothesis: prefetch budget should shrink when copy backlog grows and expand when backlog is near zero.
- Metric: tok/s and copies/token at matched cache MiB.
- Success: higher tok/s than fixed budget=8 at equal or better hit rate.

### H10 — cost-aware admission beats accuracy-only admission

- Hypothesis: admitting only experts whose expected benefit exceeds upload cost beats LRU-only admission.
- Metric: tok/s, copies/token, hit rate.
- Success: lower copies/token and higher tok/s than the current hybrid+LRU baseline.

### H11 — selective cache bypass for short prefill ubatches

- Hypothesis: bypassing cache for short prefill ubatches (for example 9-32 tokens) avoids large prefill regressions with little decode loss.
- Metric: prefill tok/s and decode tok/s.
- Success: prefill improves materially while decode remains within measurement noise of baseline.

### H12 — phase-specific policy (prefill != decode)

- Hypothesis: prefill and decode need different cache/admission policies.
- Metric: end-to-end latency split by phase.
- Success: better prefill/decode trade-off than one single policy for both phases.

### H13 — congestion-aware miss path in hybrid mode

- Hypothesis: when copy link is saturated, prioritizing CPU misses over additional background admissions keeps throughput stable.
- Metric: tok/s under high-backlog configurations.
- Success: less throughput collapse than uncapped or static-admission settings.

### H14 — task-aware cache policy

- Hypothesis: code/math/chat/long-context prompts have different expert locality and should not share one cache policy.
- Metric: per-task tok/s delta and hit-rate delta.
- Success: one simple task-aware policy improves average tok/s and avoids regressions in major buckets.

## Decision rule candidates to validate

- Use hybrid by default when upstream cache at the same MiB is slower than no-cache or harms prefill significantly.
- Increase cache MiB only while marginal tok/s gain is worth the prefill cost.
- Prefer lower-copy policies once hit rate is high enough that PCIe becomes the limiting factor.
