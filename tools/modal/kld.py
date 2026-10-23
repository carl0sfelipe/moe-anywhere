"""KL-divergence run on a rented Modal GPU.

Ports the KLD block of tools/lab_run.sh to Modal so the "Next task" from
HANDOFF.md (Bolt KLD with more chunks, 1024 and 2048 MiB) can be settled on
hardware we do not own. Only KLD is measured here: it comes from the patch's
ungated up-only branch, so it does not depend on PCIe bandwidth and travels
between machines. tok/s from a rented GPU does NOT travel - the L4 exposes
PCIe Gen3 and the A10G/L40S Gen4, against ~10 GB/s on the RTX 5050 - so never
quote decode numbers from this script as another machine.

Usage (from the moe-anywhere root):
    modal run tools/modal/kld.py --chunks 1  --caches 2048            # canary
    modal run tools/modal/kld.py --chunks 16 --caches 1024,2048,5120  # real run

Cost controls: scaledown_window=2 (no idle tail), explicit timeout, explicit
memory= (billed as max(request, actual)), preemptible, model in a Volume so the
20.5 GiB download is paid once.
"""

from __future__ import annotations

import pathlib
import subprocess
import time

import modal

APP_NAME = "moe-anywhere-kld"
LLAMA_BASE = "36cbf415f"
MODEL_REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"
MODEL_FILE = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
MODEL_PATH = f"/models/{MODEL_FILE}"

# L4 is sm_89; KLD is a numeric property of the patch (experts run on CPU via
# -ot exps=CPU), so PCIe bandwidth does not matter and the cheapest adequate
# card wins. Building one arch halves the image build time.
CUDA_ARCHS = "89"

_here = pathlib.Path(__file__).resolve()
# The local entrypoint needs the patch to upload it; inside the container the
# script lives at /root/kld.py with no such parent chain, and the patch is
# already baked at /opt/moe-cache-hybrid.patch.
_LOCAL_PATCH = (
    _here.parents[2] / "patches" / f"llama.cpp-{LLAMA_BASE}-moe-cache-hybrid.patch"
    if len(_here.parents) > 2
    else None
)

app = modal.App(APP_NAME)

models = modal.Volume.from_name("moe-anywhere-models", create_if_missing=True)

# Build llama.cpp once, at image-build time, so every run reuses the cache
# instead of paying for a 3-arch CUDA compile per cold start.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("build-essential", "git", "cmake", "curl")
    .run_commands(
        "git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        f"git -C /opt/llama.cpp checkout {LLAMA_BASE}",
        "git -C /opt/llama.cpp log -1 --format='base: %H %s'",
    )
    .add_local_file(
        str(_LOCAL_PATCH or "/opt/moe-cache-hybrid.patch"),
        "/opt/moe-cache-hybrid.patch",
        copy=True,
    )
    .run_commands(
        "git -C /opt/llama.cpp apply /opt/moe-cache-hybrid.patch",
        "git -C /opt/llama.cpp diff --stat",
        # The CUDA driver (libcuda) only exists at runtime, not at image build
        # time, so ggml's libggml-cuda.so has undefined cu* symbols at build
        # time. Point CUDA::cuda_driver at the toolkit stub (symlink the bare
        # name the linker searches for) and let the executable link tolerate
        # undefined symbols inside shared libs: the real driver is mounted when
        # the container starts. -lcuda in the link flags does NOT work (it is
        # placed before the library that needs it). -j4: cc1plus/fatbinary
        # OOM-kill above that.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "
        "/usr/local/cuda/lib64/stubs/libcuda.so.1",
        "ls -l /usr/local/cuda/lib64/stubs/libcuda.so.1",
        f"cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES='{CUDA_ARCHS}' -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_EXE_LINKER_FLAGS='-Wl,--allow-shlib-undefined'",
        "cmake --build /opt/llama.cpp/build -j4 --target llama-perplexity",
        # Verify without executing: the binary cannot run at build time (no driver).
        "ls -l /opt/llama.cpp/build/bin/llama-perplexity",
        "test -x /opt/llama.cpp/build/bin/llama-perplexity",
        "nm -D /opt/llama.cpp/build/bin/libggml-cuda.so 2>/dev/null "
        "| grep -c ' U cu' || true",
    )
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

BIN = "/opt/llama.cpp/build/bin"


@app.function(volumes={"/models": models}, image=image, timeout=3600, cpu=4, memory=16384)
def download_model() -> dict:
    """CPU-only: pay the 20.5 GiB download once, into the Volume."""
    import os

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 20 * 1024**3:
        return {"skipped": True, "bytes": os.path.getsize(MODEL_PATH)}

    from huggingface_hub import hf_hub_download

    t0 = time.perf_counter()
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="/models")
    models.commit()  # without the commit the file vanishes with the container
    return {
        "skipped": False,
        "bytes": os.path.getsize(MODEL_PATH),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _run(cmd: list[str], label: str) -> tuple[str, float]:
    """Run one step, echo stderr so failures are visible, return (stdout, seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    took = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-25:]
        print(f"[{label}] FAILED in {took:.1f}s\n" + "\n".join(tail), flush=True)
        raise RuntimeError(f"{label} exited {proc.returncode}")
    print(f"[{label}] ok in {took:.1f}s", flush=True)
    return proc.stdout, took


@app.function(
    gpu="L4",  # 23 GiB, cheapest modern GPU that holds the model + cache
    volumes={"/models": models},
    image=image,
    timeout=7200,
    startup_timeout=900,
    scaledown_window=2,
    cpu=8,
    memory=32768,
)
def kld(chunks: int = 16, caches: tuple[int, ...] = (2048,)) -> dict:
    """Reproduce lab_run.sh's KLD block at the given number of chunks."""
    import os
    import shutil

    models.reload()

    # ---- audit of the first minute (know what you are paying for)
    sysinfo, _ = _run(
        [
            "bash",
            "-lc",
            "nproc; free -g | head -2; df -h / /tmp | tail -2; "
            "nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,"
            "pcie.link.width.max,compute_cap --format=csv",
        ],
        "sysinfo",
    )
    print(sysinfo, flush=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} missing - run the download step first")

    # ---- corpus: same two docs lab_run.sh concatenates
    corpus = "/tmp/ppl.txt"
    with open(corpus, "wb") as fh:
        for doc in ("docs/build.md", "docs/function-calling.md"):
            with open(f"/opt/llama.cpp/{doc}", "rb") as src:
                shutil.copyfileobj(src, fh)
    print(f"corpus: {os.path.getsize(corpus)} bytes", flush=True)

    common = [
        "-m", MODEL_PATH,
        "-ngl", "99",
        "-ot", "exps=CPU",
        "-fit", "off",
        "--load-mode", "none",
        "-f", corpus,
        "-c", "256",
        "--chunks", str(chunks),
        "-b", "256",
        "-ub", "1",
    ]

    # ---- base: experts on CPU, no cache. Independent of cache size, so once.
    base = "/tmp/base.kld"
    _run([f"{BIN}/llama-perplexity", *common, "--kl-divergence-base", base], "base")
    total = 0.0
    results: dict[str, dict] = {}

    for mib in caches:
        for label, extra in (
            ("upstream", ["--moe-cache-mib", str(mib)]),
            ("hybrid", ["--moe-cache-mib", str(mib), "--moe-cache-hybrid"]),
        ):
            out, took = _run(
                [
                    f"{BIN}/llama-perplexity",
                    *common,
                    *extra,
                    "--kl-divergence-base", base,
                    "--kl-divergence",
                ],
                f"{label}-{mib}",
            )
            total += took
            line = next(
                (l for l in out.splitlines() if l.startswith("Mean") and "KLD" in l),
                "",
            )
            # stream it: a preemption mid-run still leaves the earlier numbers
            print(f"KLD {label}-{mib}: {line}", flush=True)
            results[f"{label}-{mib}"] = {"kld_line": line, "seconds": round(took, 1)}

    # ---- the ratio the 10% bar in HANDOFF.md is about
    for mib in caches:
        up = results.get(f"upstream-{mib}", {})
        hy = results.get(f"hybrid-{mib}", {})
        try:
            up_v = float(up["kld_line"].split()[1])
            hy_v = float(hy["kld_line"].split()[1])
            delta = (hy_v - up_v) / up_v * 100
            print(f"KLD delta {mib} MiB: {delta:+.1f}% vs upstream", flush=True)
            results[f"delta-{mib}"] = {"percent": round(delta, 1)}
        except (IndexError, KeyError, ValueError, ZeroDivisionError):
            pass

    return {
        "gpu": sysinfo.splitlines()[-1] if sysinfo.strip() else "?",
        "chunks": chunks,
        "tokens": chunks * 256,
        "results": results,
        "seconds_total": round(total, 1),
        "system": sysinfo,
    }


@app.local_entrypoint()
def main(chunks: int = 16, caches: str = "1024,2048,5120"):
    if _LOCAL_PATCH is not None and not _LOCAL_PATCH.is_file():
        raise SystemExit(f"patch not found: {_LOCAL_PATCH}")
    cache_list = tuple(int(c) for c in caches.split(",") if c.strip())
    print(f"chunks={chunks} ({chunks * 256} tokens) caches={cache_list}")

    t0 = time.perf_counter()
    dl = download_model.remote()
    print(f"download: {dl}  [{time.perf_counter() - t0:.0f}s]", flush=True)

    t1 = time.perf_counter()
    out = kld.remote(chunks, cache_list)
    print(f"\ngpu step wall time: {time.perf_counter() - t1:.0f}s", flush=True)
    print("\n== result ==")
    for key, val in out["results"].items():
        print(f"{key}: {val}")
    print(f"gpu: {out['gpu']}")
    print(f"step seconds (sum): {out['seconds_total']}")
