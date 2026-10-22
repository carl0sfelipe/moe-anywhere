# Handoff (2026-10-22)

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

## Next task (2026-10-22): settle the rebase on the 7800X3D, then the PR evidence

If you are an agent on `omarchy-white` (Ryzen 7 7800X3D, DDR5, RTX 5050 8 GB) or any other NVIDIA machine, this is your task. The owner wants it done end to end without further prompts: run, write up, commit and push. Report to the owner in Portuguese at the end.

### Why

E3f ran the rebased patch (`patches/llama.cpp-36cbf415f-moe-cache-hybrid.patch`) on `omarchy` (RTX 3090, Ryzen 5 3500X, DDR4, PCIe 3.0 x16). It builds with CUDA and the KLD matches upstream on Sage-A1.5 and on Nemotron 3.5 Bolt, but **the hybrid was about 2x slower than upstream there and unstable** (Sage hyb5120 32.5 vs up5120 68.6 tok/s). Nobody knows yet whether that is the machine or the rebase. On this desktop the old base gave hyb5120 61.5 / up5120 56.7 / nocache 48.4 (E3e). Read E3f in `research/experiments.md` and `research/nemotron-bolt.md` before starting.

### Setup (once)

- `git pull` in moe-anywhere. CUDA: `export PATH=/opt/cuda/bin:$PATH`. `hf`: `~/.venvs/hf/bin/hf` (or `pip install -U huggingface_hub` in a venv).
- `ulimit -c 0` before every llama.cpp run.
- Memory: about 30 GB RAM. Build with `JOBS=4` (6 CUDA jobs got OOM-killed on a 31 GB box). Run heavy jobs inside a capped scope so an OOM kills only them and not your own session, which happened on omarchy: `systemd-run --user --scope -q -p MemoryMax=22G -p MemorySwapMax=0 <command>`. Never load a model whose experts don't fit in RAM with `--load-mode none` (pinned).
- Models: Sage-A1.5 should already be in `~/models/Sage-A1.5/` (lab_run downloads it if missing). Bolt: `hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --local-dir ~/models/Nemotron-3.5-Bolt` (22 GB).
- Two llama.cpp trees so neither build is lost: `~/src/llama.cpp-moe` (new base, lab_run clones it if missing) and a worktree for the old base: `git -C ~/src/llama.cpp-moe fetch origin 0a1ef309cdbb9f2024f58a38fb4c42352c5db569 && git -C ~/src/llama.cpp-moe worktree add -f ~/src/llama.cpp-moe-old 0a1ef309cdbb9f2024f58a38fb4c42352c5db569`. A plain `git clone` of llama.cpp is slow (several minutes); a shallow fetch by full SHA is much faster, short SHAs don't work with `--depth 1`.
- `tools/bench_decode.sh` saves failed-run logs in the results folder. Read them if any config "failed".

### Steps

1. **Machine or rebase?** Same day, interleaved as much as practical:
   - `KLD=1 JOBS=4 bash tools/lab_run.sh` (36cbf415f, Sage)
   - `JOBS=4 LLAMA_BASE=0a1ef30 LLAMA_DIR=~/src/llama.cpp-moe-old bash tools/lab_run.sh` (0a1ef30, Sage)
   - Verdict: if new hyb5120 is within 5-10% of old hyb5120, the rebase is fine and omarchy's loss is the machine. If lower, bisect the 28 commits between 0a1ef30 and 36cbf415f (rebase the patch at the midpoint, `tools/bench_decode.sh` with hyb5120 and up5120, 3 repeats) and name the commit. Do not try to fix it; report.
2. **Second architecture.** `KLD=1 JOBS=4 MODEL=~/models/Nemotron-3.5-Bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf bash tools/lab_run.sh`. On 8 GB about 4.2 GB is non-expert weights, so the 5120 configs may not load; if they fail, rerun `tools/bench_decode.sh` by hand with 1024/1536/2048 MiB. Record the cache group lines from a `-lv 4` log (expect 12 groups).
3. **Why it lost on omarchy** (one cheap check each, no tuning): hyb5120 on Sage with `-t 4`, `-t 6`, `-t 8`; and CPU expert cost with background uploads off vs on (no-cache tok/s while a second process does host-to-device copies, or the uploader paused if there is a switch). This answers the open question "do RAM-to-VRAM copies slow the CPU experts that read the same RAM?".
4. Write **E3g** in `research/experiments.md` (machine, base commits, tables for steps 1-3, verdict), update "Estado do patch" in `notes/upstream-llama-cpp.md`, update this section to say what is done and what is next, run `python -m pytest -q` and `python tests/check_docs.py`, commit (author `carl0sfelipe <carlosfelipe2050@gmail.com>`, `Co-Authored-By: Claude` trailer) and push to moe-anywhere.

### Stop rules

- If the hybrid is slower or its KLD is more than about 10% above upstream's, say so plainly in E3g and stop. Do not tune to get a number back.
- If a build fails on 36cbf415f, make the smallest fix that blends with upstream, regenerate the patch with `git format-patch -1 --stdout`, and explain it here.

### What the PR will need (the owner writes and posts it)

- The same-day A/B from step 1 showing the rebase did not regress.
- Numbers on at least two machines and two architectures, **including the loss on omarchy**. Honest framing: the hybrid wins when the CPU expert path is cheaper than PCIe uploads and loses otherwise (break-even rule in E3e). A maintainer will ask when it should be on by default; the answer is a rule, not "always".
- KLD within 10% of upstream on every run (met so far).
- Known limits: single-token ubatches only, one GPU, CUDA only, intermittent failure at 2048 MiB not yet explained.
- Ideally a guard that falls back to the upstream path when CPU cost per expert exceeds upload cost per expert (idea 1 in `research/nemotron-bolt.md`). Prototype it in `moe_anywhere/timeline.py` first.
- Credit PR #27861 (same design) and coordinate there before opening a new PR.

### Hard limits

- Do not post, comment, push, open issues, or open PRs on ggml-org/llama.cpp, and do not draft text for them. Its rules ban AI-written posts and agent submissions, and the penalty is a contributor ban for the owner. The upstream plan in `notes/upstream-llama-cpp.md` is for the owner to write and post.
- Do not upload to bestmodel.run or HF without asking.

## Working agreements with the owner

- Talk in Portuguese. Commits use `carl0sfelipe <carlosfelipe2050@gmail.com>` with a `Co-Authored-By: Claude` trailer.
- Push to moe-anywhere and upload to the HF dataset are authorized. Ask before anything else outward-facing: llama.cpp PRs, posts, bestmodel uploads.
- Mechanical, testable work can go to llms.surf (the owner's dispatcher to cheaper models such as GLM). Specs follow the format in `specs/S*.md`: `## O que fazer`, `## Regras`, `## Dados verificados`, `VERIFICACAO:`, `## Oráculo` with `- comando:`, and an oracle that fails before the work. Check them with `~/llms.surf/bin/check-spec.sh`. Estimate tokens first to decide between dispatching and doing it yourself.
- Be honest about results: publish corrections when a claim breaks, as the README did for the 70.6 tok/s that E3b disproved.
- llama.cpp's AGENTS.md: upstream PRs must be human-understood and human-written, so prefer contributing data to PR #27861.
- Other repos: `~/src/bestmodel` has an unpushed local merge (102fdaf, the S50 `--llama-arg`); push only if asked.
