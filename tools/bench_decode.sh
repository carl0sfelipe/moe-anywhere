#!/usr/bin/env bash
# Decode benchmark for E3: fixed context, greedy, repeats interleaved across configs to spread thermal drift.
# Prints one TSV row per run: config, rep, decode tok/s, prefill tok/s, n_ctx, cache hit rate, uploaded MiB.
#
# Usage: tools/bench_decode.sh MODEL.gguf REPS "label|extra llama args" ["label|extra llama args" ...]
#   tools/bench_decode.sh Sage-A1.5-Q4_K_M.gguf 2 "nocache|--moe-cache-mib 0" "c5120|--moe-cache-mib 5120"
# Env: LLAMA_BIN (default ~/src/llama.cpp/build/bin), CTX (8192), NGEN (200), PROMPT.
set -euo pipefail
model=$1; reps=$2; shift 2
bin=${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}
ctx=${CTX:-8192}
ngen=${NGEN:-200}
prompt=${PROMPT:-"Explain how a hash map works, with an example in Python."}

printf 'config\trep\tdecode_tok_s\tprefill_tok_s\tn_ctx\thit_rate\tuploaded_mib\n'
for rep in $(seq 1 "$reps"); do
    for cfg in "$@"; do
        label=${cfg%%|*}; extra=${cfg#*|}
        # shellcheck disable=SC2086
        log=$("$bin/llama-completion" -m "$model" -ngl 99 -ot exps=CPU -c "$ctx" -fit off -p "$prompt" \
              -n "$ngen" -no-cnv --temp 0 -lv 4 $extra 2>&1) || { echo "$label failed" >&2; continue; }
        dec=$(grep -oP ' eval time .*?\K[0-9.]+(?= tokens per second)' <<<"$log" | tail -1)
        pre=$(grep -oP 'prompt eval time .*?\K[0-9.]+(?= tokens per second)' <<<"$log" | tail -1)
        nctx=$(grep -oP 'n_ctx\s*=\s*\K[0-9]+' <<<"$log" | head -1)
        hit=$(grep -oP 'ubatch <= 8: .*hit rate = \K[0-9.]+' <<<"$log" || true)
        up=$(grep -oP 'ubatch <= 8: .*uploaded = \K[0-9.]+' <<<"$log" || true)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$rep" "$dec" "$pre" "$nctx" "${hit:--}" "${up:--}"
    done
done
