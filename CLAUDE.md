Read HANDOFF.md first: project state, key numbers, the next experiment and the working agreements with the owner. All measurements are in research/experiments.md.

- Tests: `python -m pytest -q` and `python tests/check_docs.py`. Every URL in README.md and research/*.md must be in the allowlist in tests/check_docs.py.
- Benchmarks: `tools/bench_decode.sh` (pins `-c 8192 -fit off`, interleaves repeats). Full run on a new machine: `tools/lab_run.sh`.
- Run crash-prone llama.cpp experiments with `ulimit -c 0`.
