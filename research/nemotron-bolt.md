# Nemotron 3.5 Bolt: context for the E3f/E5 runs (2026-10-22)

Why this model, what is known about it, the risks for the hybrid cache, and three experiment ideas. Facts marked **(GGUF)** were read from the file itself with `gguf-py`; facts marked **(code)** from llama.cpp 36cbf415f; everything else from the sources at the end. Numbers marked *unverified* come from third-party pages and must be re-measured before we cite them.

## Why this model

- The owner asked for a recent, trending open model that does **not** fit the 24 GB RTX 3090, so the cache has real work to do. Sage-A1.5 Q4_K_M (21.7 GB) fits whole on a 3090, so a run there only measures overhead.
- NVIDIA released Nemotron 3.5 Bolt 74B-A9B in October 2026. On 2026-10-22 its official GGUF had about 102k downloads on Hugging Face, the most of any recent MoE we checked. Google Trends could not be queried from the agent, so "trending" rests on HF downloads and press coverage.
- Two files, two regimes:
  - `unsloth/...-UD-Q3_K_XL.gguf`, **39.0 GB** (GGUF): 32 GB of experts against about 16 GB of VRAM free for the cache. This is the heavy-spill regime, like Sage-A1.5 on the 8 GB RTX 5050.
  - `nvidia/...-24GB.gguf`, **22.0 GB**, trained for 2-bit with QAD (IQ2_XXS 91.7%, NVFP4 5.9%, Q8_0 2.3%). It fits by itself, but NVIDIA's own recipes ask for a 32 GB+ GPU (28 GB recipe: 65k context, q8_0 KV, LiLiCorr drafter; 32 GB recipe: 262k context). Weights + mmproj (1.73 GB) + drafter (1.0 GB) = 24.75 GB, already over a 3090 before KV. This is the **light-spill, real-user regime**: what a 3090 owner hits when following the model card.
- Quality caveat: the Q3_K_XL file is a post-hoc quant of BF16, while the 24GB file was quantization-aware trained. Bigger is not obviously better here; don't claim Q3 is the "better" model without a KLD/eval check against BF16.

## Architecture

- `nemotron_h_moe`, 88 blocks (GGUF): 40 Mamba-2, 40 LatentMoE, 8 attention (at blocks 7, 16, 25, 36, 47, 58, 69, 78). Hidden 4096, 2 KV heads, head dim 128, vocab 131k, 262k context. Mamba-2: 128 heads x 64, state 128, 8 groups.
- Order is mostly Mamba, MoE, Mamba, MoE, ...; every MoE block is preceded by a Mamba block (or by attention at the 8 anchors).
- LatentMoE (Nemotron 3 Super paper): `ffn_latent_down` projects 4096 to a latent of 1024 (GGUF `moe_latent_size`), the router and the routed experts run in the latent, and `ffn_latent_up` projects back. Experts are ungated ReLU², up + down only, no gate. The shared expert (width 5376) runs on the full 4096 hidden state. Router: sigmoid with selection bias, scale 5.0, top-9 in every MoE layer (GGUF).
- **Heterogeneous layers** (GGUF, from the Star Elastic compression of the 120B-A12B parent): experts per layer are 240, 304, 368, 432 or 496; expert width is 1536, 2048, 2304 or 2560. In the Q3_K_XL file the experts also use 7 quant types (Q5_K, Q4_K, IQ4_XS, IQ3_S, IQ3_XXS, ...), mixed per layer and per matrix. Expert size runs from 1.80 to 2.62 MB (Q3_K_XL); about 0.9-1.4 MB at IQ2_XXS (estimate: 2.06 bpw x 2 x 1024 x width).
- Per token, the routed working set is 40 layers x 9 experts x about 2.3 MB = about 830 MB at Q3_K_XL. Upper bound on miss traffic per token.
- llama.cpp support landed in PR #30520 (merged 2026-10-15, before our base 36cbf415f). Each layer's expert tensors have their own `ne[2]` (code: `nemotron-h.cpp` reads the count from the router tensor), so nothing is padded to 496.
- Speculative decoding: the LiLiCorr drafter (1.13B params, drafts 7 tokens) is used through `--spec-type draft-dflash`. NVIDIA reports mean acceptance lengths of 4.68 (greedy) and 4.25 (sampling). A third-party report cites 283 vs 132 tok/s with vs without the drafter on an RTX 5090 (*unverified*).
- Reported RTX 3090 speed with everything on the GPU at 16k context: about 85 tok/s decode, 21.3 GB peak (*unverified*, third-party). This is the ceiling the cache cannot beat; it is the reference for the light-spill regime.

## Risks for the hybrid cache

1. **The hybrid path applies** (code). `build_moe_ffn` is called with the layer's own `ne[2]` and per-layer `n_expert_used`. `hybrid_ok` requires no expert scale tensors (`*_exps_s`); trunk layers don't create them, only the MTP block may. Ungated experts (no `gate_exps`) take the up-only branch of `build_experts`; it compiles, but no model before this one exercised it with the hybrid. **Check first:** load log says the cache is on for all 40 layers, KLD matches upstream.
2. **Cache fragmentation** (code). The cache groups layers that share a device and an identical expert-tensor layout. With 4 counts x 4 widths x 7 quant mixes, almost every layer is its own group (Q3_K_XL: roughly 30+ groups out of 40 layers). Effects: each group has its own SLRU, so there is no global recency across layers; budget is split "so each group caches the same fraction of its experts", which ignores how often each layer misses; a group with fewer slots than `n_expert_used + 1` (+9 zero slots in hybrid) is refused. At small budgets several layers may get no cache at all. Log the group count and slots per group on every run.
3. **Upload pacing tuned for one machine.** The 64 KiB chunks with one copy in flight were tuned on the RTX 5050 at PCIe 4.0 x8. The 3090 here sits on PCIe at the x16 slot of a B450M (PCIe 3.0, about 13 GB/s nominal); re-measure the effective upload rate with `bench-hop.md` before trusting the timeline.
4. **CPU side is weak.** Ryzen 5 3500X: 6 cores, no SMT, AVX2 only, dual-channel DDR4. CPU expert cost will be higher than on the i9 or the 7800X3D, which favours uploads over CPU misses. IQ-family kernels on the CPU in mainline llama.cpp are known to be slower per byte than K-quants (ik_llama.cpp discussions report up to 3x faster IQ2_XXS CPU kernels than mainline).
5. **Single-token hybrid ubatches** (known trap). The hybrid needs `n_expert_used` distinct zero slots and runs one token at a time. Speculative decoding verifies 5-8 tokens per batch, so the official recipe (with the LiLiCorr drafter) cannot use the hybrid today; it falls back or must be disabled. Also, third-party tests show speculative decoding *slowing* a 35B-A3B MoE on a 3090 because the verify batch loads the union of every draft token's experts; with 9 of 240-496 experts per layer the union grows fast.
6. **Machine is shared.** The 3090 normally serves Qwen3.8-27B (`qwen-rig.service`, 21.5 GB). Runs need it stopped and restarted afterwards; prod/stag containers keep running, so expect some CPU noise. Free host RAM is about 10 GB with the server up, which is too little to compile CUDA with `-j6` (cc1plus was OOM-killed; build with `-j2`).
7. **Mamba state and context.** SSM state is fixed size, but the 8 attention layers' KV grows with context. At 65k-262k context, KV competes with the cache for VRAM; pin `-c` in every run as the hygiene rules say.

## Idea 1: quant-aware miss path

**Claim.** The hybrid's rule "every miss runs on the CPU" is wrong for some quant types. For each expert, the cheaper miss path depends on bytes to upload vs CPU cost to dequantize and multiply, and both depend on the quant type: 2-bit IQ experts are tiny to upload but expensive per byte on the CPU; K-quants are the opposite.

**Why now.** Bolt's Q3_K_XL file mixes 7 expert quant types inside one model, and the official file is almost all IQ2_XXS. One model gives a natural experiment.

**Prior art.** Fiddler, kTransformers and HybriMoE decide CPU vs GPU per expert by token count; HOBBIT and others swap precision to make transfers cheaper; DALI and PreScope model workload. We found nobody who picks the miss path from the quant type's measured CPU cost vs its upload cost. Search again before claiming it.

**Experiment.**
1. Microbench, per quant type on this CPU: ms per expert for the CPU path (`-ot exps=CPU`, one expert size at a time) and ms per MiB for uploads. Extends E3d's recalibrated cost model (0.09 ms/layer + 0.045 ms/CPU expert) with a per-type term.
2. Add a per-type miss-path choice to `moe_anywhere/timeline.py`; replay Bolt traces with "all-CPU", "all-upload" and "per-type best"; go to C++ only if per-type beats both by more than run noise (5-10%).

## Idea 2: Mamba blocks as upload shadow

**Claim.** In Bolt every MoE block is preceded by a Mamba block that touches no experts. If layer L+1's experts can be predicted from the residual *before* that Mamba block, the uploads for L+1 can run while the GPU computes Mamba, which is free time from the cache's point of view.

**What we know.** E2/E3c: pre-gating one layer ahead recalls 0.82 of experts (0.96 with 16 candidates) from `l_out`, 0.79 / 0.94 from `attn_residual`, on Sage-A1.5. Unknown for Bolt: the predictor must skip a Mamba block, and it routes in a 1024-dim latent after `ffn_latent_down`, so we would apply the next MoE block's `latent_down` and router to the earlier residual.

**Prior art.** Lookahead and pre-gated predictors exist (Pre-gated MoE, ProMoE, SP-MoE, SPICE). We found no work that measures prediction across SSM blocks in a hybrid Mamba MoE or uses those blocks as the transfer window.

**Experiment.**
1. Dump router inputs and `l_out` for Bolt (the trace tooling in `moe_anywhere/` already does this for Sage), measure recall@9 and recall@16 for "router(L+1) on residual before Mamba(L+1)".
2. Measure Mamba block time on the 3090 (it sets the window), then replay in `timeline.py` with prefetch issued one Mamba block early.

## Idea 3: the official 28 GB recipe on a 24 GB card

**Claim.** NVIDIA's own recipe (65k context, q8_0 KV, LiLiCorr drafter, mmproj) needs 28 GB. With the cache, only the experts that don't fit (about 4-6 GB out of about 19 GB of IQ2 experts) live in RAM, and the hit rate should stay near 100% for the hottest experts. That turns "needs a 32 GB card" into "runs on a 3090 at nearly full speed".

**Blocker.** Speculative decoding verifies multi-token batches, and the hybrid is single-token only (risk 5). The upstream cache (`--moe-cache-mib`) does accept multi-token batches; measure it first. A multi-token hybrid is HANDOFF follow-up 2.

**Prior art.** SP-MoE, SPICE, DraftExpert and Specter combine speculative decoding with offloading on GPU-poor setups; none targets a QAT 2-bit model at the edge of one card, and none uses llama.cpp. The useful contribution is measurement: tok/s and acceptance for the official recipe on 24 GB, with and without the cache, against the all-GPU 16k baseline.

**Experiment.** Three configs at 65k context with the drafter: everything on GPU with `-ot exps=CPU` for the N layers that don't fit (llama.cpp's stock answer); upstream `--moe-cache-mib`; and, once it exists, a multi-token hybrid. Report decode tok/s, acceptance length and VRAM peak.

## Run checklist

- `qwen-rig.service` stopped before, started after; confirm `curl 127.0.0.1:8888/v1/models` answers afterwards.
- `-c 8192 -fit off --load-mode none` for the E3f matrix; `ulimit -c 0`.
- From the load log: number of cache groups, slots per group, layers without cache.
- Effective upload MiB/s and CPU ms/expert before trusting any timeline replay.

## Sources

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF (model card: recipes, quantization, architecture)
- https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF (Q3_K_XL file used here)
- https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/ (acceptance lengths)
- https://arxiv.org/html/2604.12374 (Nemotron 3 Super, LatentMoE)
- https://github.com/ggml-org/llama.cpp/pull/30520 (Bolt support in llama.cpp)
- https://github.com/jamesburton/dotLLM/issues/1035 (third-party architecture notes; layer counts confirmed against the GGUF)
- https://www.bodegaone.ai/blog/run-nemotron-3-5-bolt-locally (24 GB fit, file sizes)
- https://hackmd.io/@thc1006/SJly6IE6Wx (speculative decoding slower on a 35B-A3B MoE on an RTX 3090)
- https://github.com/ikawrakow/ik_llama.cpp/discussions/164 (CPU speed of IQ quants, mainline vs ik_llama.cpp)
- https://arxiv.org/abs/2510.10302 (SP-MoE), https://www.emergentmind.com/papers/2608.21240 (SPICE), https://arxiv.org/abs/2607.24434 (DraftExpert), https://arxiv.org/abs/2511.14102 (Specter)
- https://arxiv.org/html/2402.07033 (Fiddler), https://ar5iv.labs.arxiv.org/html/2504.05897 (HybriMoE), https://arxiv.org/pdf/2411.01433 (HOBBIT), https://arxiv.org/html/2602.03495 (DALI)
