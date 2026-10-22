# Checked references from the 2026-10 deep-research review

The review (an external AI deep-research run, 2026-10-21) cited many 2026 sources. These were checked by hand; everything else in it is unverified.

| Reference | Checked | Notes |
|---|---|---|
| llama.cpp PR #27861 (csantiago78) | yes, open draft, 2026-08-28 | GPU-resident LRU, misses on CPU. Body: Qwen3.8-Flash-Next, 2x RTX 3090, 18.4 to 24.2 tok/s (+31%) |
| llama.cpp discussion #24528 (leloch) | yes, 2026-06-12 | RFC for VRAM caching of hot experts with hybrid hit/miss execution |
| llama.cpp PR #29887 (am17an) | yes, merged 2026-10-07 | the `--moe-cache-mib` measured in E3b |
| llama.cpp PR #30112 | yes, merged 2026-10-08 | MoE cache over multiple GPUs |
| llama.cpp issue #30163 | yes, open | per-device `--moe-cache-mib` |
| arXiv 2402.07033 Fiddler | yes | CPU-GPU orchestration, misses on CPU |
| arXiv 2602.03495 DALI | yes | workload-aware offloading on local PCs |
| arXiv 2610.01950 MoE-CORE | yes | phase-specific offloading (prefill vs decode) |
| arXiv 2610.23006 | yes, but | real title "Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment" (up to +51.9% prefill, +39.5% decode). The abstract does not mention "Post-Fetch", the models or the hardware the review attributed to it; check the full text before citing those details |
| arXiv 2608.11688 APEX | yes | adaptive expert prefetching, edge |
| arXiv 2608.15018 S2-MoE | yes | self-speculative decoding for MoE on edge |
| sampop3, dev.to, MoE cache on a T4 | yes | Qwen3-30B-A3B, `--cpu-moe` + 10000 MiB: 9.6 to 43.2 tok/s decode, 150 to 382 prefill on a 1,436-token prompt; cache serves ubatches <= 32 only |

Not checked: ProMoE (2410.22134, the arXiv API lookup failed), community posts (FreeToken 144 tok/s, OpenClawDC, PartnerInAI, Level1Techs), the Strix Halo prices and benchmarks, and the vLLM RFCs.
