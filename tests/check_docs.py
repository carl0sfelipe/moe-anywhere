"""Oracle for spec S1: docs skeleton. Exit 0 only when every check passes."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = {
    "README.md": ["## Sponsor", "## Problem", "## Results", "## Roadmap", "## Contributing", "## License"],
    "research/README.md": ["## Map", "## How to contribute"],
    "research/literature.md": ["## Prefetching", "## Offloading", "## Placement"],
    "research/prior-art.md": ["## Engines", "## Prefetch projects", "## Gap"],
    "research/models.md": ["## Sage-A1.5", "## DeepSeek-V4.1-Flash"],
    "research/bandwidth.md": ["## Tiers", "## Method"],
    "research/experiments.md": ["## E0", "## E1", "## E2", "## E3"],
    "research/open-questions.md": ["## Questions"],
}

NAMES = {
    "research/prior-art.md": [
        "DeepInfer", "cachedMoE", "Flare", "llama-moe-cache", "moe-prefetch", "vLLM",
        "FreeToken", "moe-offload", "KTransformers", "llama.cpp", "Unsloth",
    ],
    "research/literature.md": [
        "Pre-gated MoE", "MoE-Infinity", "ProMoE", "AdapMoE", "HOBBIT", "Fiddler",
        "MoE-Lightning", "SeqMoE", "SpecPrefetch", "EPLB",
    ],
}

# The only URLs allowed anywhere in the docs: everything else is a hallucination risk.
ALLOWED_URLS = {
    "https://bestmodel.run",
    "https://github.com/deepseek-ai/DeepInfer",
    "https://github.com/deepseek-ai/DeepInfer/pull/142",
    "https://github.com/acupof-ai/cachedMoE",
    "https://github.com/NVIDIA/flare-local-inference",
    "https://github.com/ongunm/llama-moe-cache",
    "https://github.com/DimitriChrysafis/moe-prefetch",
    "https://github.com/vllm-project/vllm/issues/38256",
    "https://github.com/FlashML-org/FreeToken/issues/677",
    "https://github.com/Jakevin/moe-offload",
    "https://github.com/ggml-org/llama.cpp",
    "https://arxiv.org/pdf/2609.12978",
    "https://arxiv.org/pdf/2607.24787",
    "https://huggingface.co/InternScience/Sage-A1.5",
    "https://www.apache.org/licenses/LICENSE-2.0",
}

URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")


def main() -> int:
    errors = []
    for rel, headings in REQUIRED.items():
        path = ROOT / rel
        if not path.is_file() or not path.read_text().strip():
            errors.append(f"missing or empty: {rel}")
            continue
        lines = path.read_text().splitlines()
        for h in headings:
            if not any(line.strip().startswith(h) for line in lines):
                errors.append(f"{rel}: missing heading '{h}'")
    for rel, names in NAMES.items():
        path = ROOT / rel
        text = path.read_text() if path.is_file() else ""
        for n in names:
            if n not in text:
                errors.append(f"{rel}: missing entry '{n}'")
    readme = ROOT / "README.md"
    if readme.is_file() and "https://bestmodel.run" not in readme.read_text():
        errors.append("README.md: sponsor link https://bestmodel.run missing")
    for path in [readme, *sorted((ROOT / "research").glob("*.md"))]:
        if not path.is_file():
            continue
        for url in URL_RE.findall(path.read_text()):
            if url.rstrip(".,;:") not in ALLOWED_URLS:
                errors.append(f"{path.relative_to(ROOT)}: URL not in allowlist: {url}")
    for e in errors:
        print("FAIL", e)
    print("ok" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
