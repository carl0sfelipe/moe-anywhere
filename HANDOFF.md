# Handoff (2026-10-21)

State of the lab for whoever continues it, human or agent, with no prior context. Read this, then `research/experiments.md` (E0-E3d), which has every number.

## Where things are

- Repo: https://github.com/carl0sfelipe/moe-anywhere (public, branch main). Sponsor: bestmodel.run, the owner's compatibility engine. Lab findings feed it.
- Dataset: https://huggingface.co/datasets/carlosfelipe/moe-anywhere-routing-traces (traces, hidden states, `attn_residual` dumps, E3b/E3d benchmark TSVs).
- Model: `InternScience/Sage-A1.5-Q4_K_M-GGUF` / `Sage-A1.5-Q4_K_M.gguf` (Qwen3.6-35B-A3B fine-tune, arch qwen35moe; 40 layers + 1 MTP layer; 256 experts, top-8 + shared expert; 21.7 GB). Experts: ffn_down Q6_K in 20 layers (2.04 MB/expert), Q4_K in 20 (1.77 MB).
- Engine: llama.cpp commit 0a1ef30 plus `patches/llama.cpp-0a1ef30-moe-cache-hybrid.patch` (the `--moe-cache-hybrid` flag). On the first machine this lives on the local branch `moe-anywhere/hybrid` (commit f3819c2) in `~/src/llama.cpp`, which was never pushed; the patch file is the source of truth.
- First machine: mini PC, i9-12900HK, 31 GB RAM (about 46 GB/s effective for CPU experts), RTX 5050 8 GB (SM120) in a dock at PCIe 4.0 x8 (about 10 GB/s of uploads), CachyOS/omarchy.

## What we learned (details in experiments.md)

1. Routing (E1/E2): expert ids are unrelated across layers (overlap 0.235 against 0.25 by chance). Consecutive tokens share 2.86 of 8 experts. An LRU cache of 64 of 256 experts per layer gives 79% hits.
2. Pre-gating (E2/E3c): layer L+1's router applied to `l_out` of layer L recalls 0.82 of its experts, or 0.96 with 16 candidates. From `attn_residual-L`, the signal available at the cache sync point, the recall is 0.79 / 0.94.
3. llama.cpp's own cache, `--moe-cache-mib` (PR #29887, E3b): one SLRU shared by each group of layers with the same expert layout. Misses are uploaded while the GPU waits, so t = 10.9 ms + 0.098 ms per uploaded MiB, with no overlap at all. It only beats experts-on-CPU above about 4.5 GB, and it makes prefill of short prompts 2.7x slower.
4. `moe_anywhere/timeline.py` (E3c): its SLRU port is checked against the C++ source in `tests/test_timeline.py`, and its replay matches measured hit rate and tok/s within 2%. It showed that on this hardware the miss path matters more than prediction. Pre-gating adds +17% on the upload design and under 1 tok/s on the hybrid.
5. Hybrid (E3d): misses run on the CPU and a paced uploader thread fills the cache in the background.
   - Results: 55.1 tok/s against 45.5 without cache and 50.9 for upstream at 5 GB. At 2 GB it gets 44.2 against upstream's 30.4, and prefill stays at about 90 tok/s.
   - Correctness: KL divergence against no-cache is the same as upstream's (0.0062 vs 0.0060).
   - Two traps: (a) a shared zero slot gives duplicate ids per token and crashes CUDA MMQ, so it needs `n_expert_used` distinct zero slots and only single-token ubatches are allowed; (b) one memcpy per expert blocks the single host-to-device copy engine, so uploads must go in 64 KiB chunks with one in flight.
   - Recalibrated costs: about 0.09 ms fixed per layer for the CPU chain and 0.045 ms per CPU expert. The next win is overlapping the CPU and GPU halves.
6. Prior art: llama.cpp draft PR #27861 has the same hybrid design (+31% on 2x3090). Credit it everywhere; the project's value is the traces, the validated timeline and the measurements.

## Benchmark hygiene (debts that bit us)

- Always pin `-c 8192 -fit off`. Otherwise `-fit` silently shrinks the context, from 262k to 4096 tokens when the cache is on.
- Use `--load-mode none` (pinned memory, +6%); async uploads from mmap'd memory block the host.
- This machine varies 5-10% between runs, so interleave repeats with `tools/bench_decode.sh`.
- llama-bench has no `--moe-cache-mib`; use `tools/bench_decode.sh`, which runs llama-completion.
- Run crash-prone experiments with `ulimit -c 0`. Every abort leaves a 170 MB core dump and a desktop notification.

## Done: same RTX 5050 in a Ryzen 7 7800X3D desktop (E3e)

`tools/lab_run.sh` ran on `omarchy-white` (results in `results/omarchy-white-20261021-1510/`, write-up in experiments.md E3e). Hybrid 5120 MiB: 61.5 tok/s against 48.4 without cache and 56.7 for upstream. The fast-RAM hypothesis was wrong: CPU experts got only 16% cheaper (0.032 ms) while uploads got 22% cheaper (0.076 ms/MiB). With those two numbers the timeline replays within 2% for upstream and 2-5% optimistic for the hybrid. Before trusting a prediction on new hardware, measure both effective bandwidths with `bench-hop.md` and two upstream cache sizes.

Build notes for this machine: CUDA is in `/opt/cuda/bin` (not on PATH by default), and `hf` lives in `~/.venvs/hf`.

After that:
1. Overlap the CPU and GPU halves of the hybrid layer (the per-layer fixed cost of about 0.09 ms).
2. Multi-token hybrid ubatches for speculative decoding.
3. Submit results to bestmodel.run. This needs an agent token: `POST /v1/auth/tokens` with the console session in localStorage `bm_token`, then `benchmark-probe login --token ...` typed by the owner. Never paste tokens in chat, and upload only with the owner's OK.
4. DeepSeek-V4.1-Flash on the 8x3090 cluster (docs in the private repo carl0sfelipe/rig3d): re-measure the cost model there, since PCIe 3.0 risers change the balance.

## Rebased on master (2026-10-22)

`patches/llama.cpp-36cbf415f-moe-cache-hybrid.patch` is the patch rebased on llama.cpp master 36cbf415f. Only CPU builds were checked; it has no CUDA build or bench yet. The plan for upstream is in `notes/upstream-llama-cpp.md`. On the CUDA machine:

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout -b moe-anywhere/hybrid 36cbf415f
git am ../moe-anywhere/patches/llama.cpp-36cbf415f-moe-cache-hybrid.patch
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
```

Then rerun `tools/lab_run.sh` (bench + KLD) and compare it with E3d/E3e before posting anything upstream.

## Working agreements with the owner

- Talk in Portuguese. Commits use `carl0sfelipe <carlosfelipe2050@gmail.com>` with a `Co-Authored-By: Claude` trailer.
- Push to moe-anywhere and upload to the HF dataset are authorized. Ask before anything else outward-facing: llama.cpp PRs, posts, bestmodel uploads.
- Mechanical, testable work can go to llms.surf (the owner's dispatcher to cheaper models such as GLM). Specs follow the format in `specs/S*.md`: `## O que fazer`, `## Regras`, `## Dados verificados`, `VERIFICACAO:`, `## Oráculo` with `- comando:`, and an oracle that fails before the work. Check them with `~/llms.surf/bin/check-spec.sh`. Estimate tokens first to decide between dispatching and doing it yourself.
- Be honest about results: publish corrections when a claim breaks, as the README did for the 70.6 tok/s that E3b disproved.
- llama.cpp's AGENTS.md: upstream PRs must be human-understood and human-written, so prefer contributing data to PR #27861.
- Other repos: `~/src/bestmodel` has an unpushed local merge (102fdaf, the S50 `--llama-arg`); push only if asked.
