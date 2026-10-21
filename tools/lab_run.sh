#!/usr/bin/env bash
# One-shot E3 lab run on a new machine: system info, llama.cpp 0a1ef30 + hybrid patch, Sage-A1.5, benchmarks.
# Results land in results/<host>-<date>/ (summary.tsv plus raw logs); send that folder back.
#
# Usage (from a clone of moe-anywhere):  bash tools/lab_run.sh
# Env: MODELS (~/models/Sage-A1.5), LLAMA_DIR (~/src/llama.cpp-moe), REPS (3), KLD (0; 1 = also run the KL check)
# Needs: git, cmake, a CUDA toolkit with nvcc, nvidia-smi, python3, and `hf` (pip install -U huggingface_hub).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
models=${MODELS:-$HOME/models/Sage-A1.5}
llama=${LLAMA_DIR:-$HOME/src/llama.cpp-moe}
reps=${REPS:-3}
out=$root/results/$(hostname)-$(date +%Y%m%d-%H%M)
mkdir -p "$out"
model=$models/Sage-A1.5-Q4_K_M.gguf

for tool in git cmake nvcc nvidia-smi python3; do
    command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

echo "== system info"
{
    echo "date: $(date -Is)"
    echo "kernel: $(uname -r)"
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
    free -g | head -2
    sudo -n dmidecode -t memory 2>/dev/null | grep -E "^\s+(Type|Speed|Configured Memory Speed|Size):" | sort | uniq -c || echo "(dmidecode needs sudo: run 'sudo dmidecode -t memory' for RAM speed)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,compute_cap --format=csv
} | tee "$out/system.txt"

echo "== llama.cpp 0a1ef30 + moe-anywhere hybrid patch"
if [ ! -d "$llama" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$llama"
fi
git -C "$llama" fetch -q origin 0a1ef30 2>/dev/null || true
git -C "$llama" checkout -q -f 0a1ef30
git -C "$llama" apply "$root/patches/llama.cpp-0a1ef30-moe-cache-hybrid.patch"
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S "$llama" -B "$llama/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_BUILD_TYPE=Release >"$out/cmake.log" 2>&1
cmake --build "$llama/build" -j "$(nproc)" --target llama-completion llama-bench llama-perplexity >"$out/build.log" 2>&1
bin=$llama/build/bin

echo "== model"
if [ ! -f "$model" ]; then
    hf download InternScience/Sage-A1.5-Q4_K_M-GGUF Sage-A1.5-Q4_K_M.gguf --local-dir "$models"
fi

# the PCIe link only shows its real speed under load
( sleep 20; nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv ) >"$out/pcie-under-load.txt" 2>&1 &

echo "== llama-bench: threads, hop cost, per-layer CPU cost (tg128)"
"$bin/llama-bench" -m "$model" -ngl 99 -p 0 -n 128 -r "$reps" -ot "exps=CPU" -t 4,6,8,12,16 -o md 2>/dev/null | tee "$out/bench-threads.md"
"$bin/llama-bench" -m "$model" -ngl 99 -p 0 -n 128 -r "$reps" -o md \
    -ot "exps=CPU" -ot "exps=CPU;attn_norm=CPU" -ot "exps=CPU;attn_norm=CPU;post_attention_norm=CPU" \
    -ot "blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU" 2>/dev/null | tee "$out/bench-hop.md"

echo "== decode matrix (tools/bench_decode.sh, pinned memory, ctx 8192)"
LLAMA_BIN=$bin "$root/tools/bench_decode.sh" "$model" "$reps" \
    "nocache|--moe-cache-mib 0 --load-mode none" \
    "up2048|--moe-cache-mib 2048 --load-mode none" \
    "up5120|--moe-cache-mib 5120 --load-mode none" \
    "hyb2048|--moe-cache-mib 2048 --moe-cache-hybrid --load-mode none" \
    "hyb3072|--moe-cache-mib 3072 --moe-cache-hybrid --load-mode none" \
    "hyb5120|--moe-cache-mib 5120 --moe-cache-hybrid --load-mode none" | tee "$out/summary.tsv"

if [ "${KLD:-0}" = 1 ]; then
    echo "== KL divergence of the hybrid against experts-on-CPU (ubatch 1)"
    cat "$llama/docs/build.md" "$llama/docs/function-calling.md" >"$out/ppl.txt"
    common=(-m "$model" -ngl 99 -ot exps=CPU -fit off --load-mode none -f "$out/ppl.txt" -c 256 --chunks 4 -b 256 -ub 1)
    "$bin/llama-perplexity" "${common[@]}" --kl-divergence-base "$out/base.kld" >/dev/null 2>&1
    for cfg in "--moe-cache-mib 5120" "--moe-cache-mib 5120 --moe-cache-hybrid"; do
        # shellcheck disable=SC2086
        echo "$cfg: $("$bin/llama-perplexity" "${common[@]}" $cfg --kl-divergence-base "$out/base.kld" --kl-divergence 2>&1 | grep -E '^Mean    KLD')"
    done | tee "$out/kld.txt"
    rm -f "$out/base.kld"
fi

git -C "$llama" checkout -q -f 0a1ef30  # leave the tree clean
echo "== done: $out"
