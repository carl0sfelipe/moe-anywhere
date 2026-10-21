"""Print the MoE facts of a Hugging Face config.json as a markdown table."""
import json
import sys
from pathlib import Path

KEYS = (
    "num_layers", "n_experts", "top_k", "hidden_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "linear_attention_layers", "full_attention_layers",
)


def facts(config: dict) -> dict:
    c = config.get("text_config", config)
    layer_types = c.get("layer_types", [])
    return {
        "num_layers": c["num_hidden_layers"],
        "n_experts": c["num_experts"],
        "top_k": c["num_experts_per_tok"],
        "hidden_size": c["hidden_size"],
        "moe_intermediate_size": c["moe_intermediate_size"],
        "shared_expert_intermediate_size": c["shared_expert_intermediate_size"],
        "linear_attention_layers": layer_types.count("linear_attention"),
        "full_attention_layers": layer_types.count("full_attention"),
    }


def est_size_gb(params_billion: float, bpw: float) -> float:
    return params_billion * bpw / 8


def to_markdown(f: dict) -> str:
    rows = ["| key | value |", "|---|---|"] + [f"| {k} | {f[k]} |" for k in KEYS]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    print(to_markdown(facts(json.loads(Path(sys.argv[1]).read_text()))), end="")
