"""Oracle for spec S5: bandwidth-bound tok/s calculator. Cases computed by hand."""
import subprocess
import sys

import pytest

from moe_anywhere.bandwidth import PRESETS, Tier, tokens_per_sec


def test_presets_exact():
    assert PRESETS == {
        "pcie3_x4": 3.5, "pcie3_x16": 14.0, "ddr4_dual": 35.0, "ddr4_quad": 60.0,
        "nvme_gen3": 3.0, "gbe1": 0.117, "ib40": 4.0,
    }


def test_single_tier():
    # 6 GB from quad-channel DDR4 at 60 GB/s = 0.1 s/token
    assert tokens_per_sec({"ram": 6e9}, {"ram": Tier("ram", 60.0)}) == pytest.approx(10.0)


def test_two_tiers_add_up():
    # 3 GB @ 60 GB/s = 0.05 s, 0.3 GB @ 3 GB/s = 0.1 s, total 0.15 s
    tiers = {"ram": Tier("ram", 60.0), "ssd": Tier("ssd", 3.0)}
    assert tokens_per_sec({"ram": 3e9, "ssd": 0.3e9}, tiers) == pytest.approx(1 / 0.15)


def test_hops_add_latency():
    # 1 GB @ 100 GB/s = 0.01 s, plus 3 hops of 5 ms = 0.015 s, total 0.025 s
    tiers = {"vram": Tier("vram", 100.0)}
    assert tokens_per_sec({"vram": 1e9}, tiers, hops=3, hop_latency_ms=5.0) == pytest.approx(40.0)


def test_nothing_to_do_is_an_error():
    with pytest.raises(ValueError):
        tokens_per_sec({}, {})


def test_unknown_tier_is_an_error():
    with pytest.raises(KeyError):
        tokens_per_sec({"disk": 1e9}, {"ram": Tier("ram", 60.0)})


def test_cli_with_preset():
    out = subprocess.run(
        [sys.executable, "-m", "moe_anywhere.bandwidth", "--bytes", "ddr4_quad=6e9"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "10.00" in out
