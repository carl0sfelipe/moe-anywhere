"""Oracle for spec S2: model facts from a real config.json."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "sage-a1.5-config.json"
sys.path.insert(0, str(ROOT / "tools"))


def _facts():
    import model_facts
    return model_facts.facts(json.loads(CONFIG.read_text()))


def test_facts_values():
    f = _facts()
    assert f["num_layers"] == 40
    assert f["n_experts"] == 256
    assert f["top_k"] == 8
    assert f["hidden_size"] == 2048
    assert f["moe_intermediate_size"] == 512
    assert f["shared_expert_intermediate_size"] == 512
    assert f["linear_attention_layers"] == 30
    assert f["full_attention_layers"] == 10


def test_flat_config_also_works():
    import model_facts
    flat = json.loads(CONFIG.read_text())["text_config"]
    assert model_facts.facts(flat)["n_experts"] == 256


def test_est_size_gb():
    import model_facts
    assert model_facts.est_size_gb(35, 4.0) == pytest.approx(17.5)
    assert model_facts.est_size_gb(552, 2.0) == pytest.approx(138.0)


def test_cli_prints_markdown_table():
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "model_facts.py"), str(CONFIG)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out.lstrip().startswith("|")
    assert "| n_experts | 256 |" in out
    assert "| top_k | 8 |" in out


def test_models_md_contains_generated_table():
    import model_facts
    table = model_facts.to_markdown(_facts()).strip()
    assert table in (ROOT / "research" / "models.md").read_text()
