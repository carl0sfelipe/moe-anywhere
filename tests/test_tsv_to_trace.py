import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from tsv_to_trace import tsv_to_trace  # noqa: E402

HEADER = "prompt_id\ttask\ttoken_pos\tphase\tlayer\texperts\tweights\n"


def _row(pid, task, pos, phase, layer, experts):
    w = ",".join(["0.5", "0.5"])
    return f"{pid}\t{task}\t{pos}\t{phase}\t{layer}\t{','.join(map(str, experts))}\t{w}\n"


@pytest.fixture
def tsv(tmp_path):
    lines = [HEADER]
    # prompt 0: prefill token 0 lacks the last layer; tokens 1 (prefill, last) and 2 (decode) are complete
    lines += [_row(0, "code", 0, "prefill", 0, [1, 2])]
    lines += [_row(0, "code", 1, "prefill", l, [3, 4]) for l in range(2)]
    lines += [_row(0, "code", 2, "decode", l, [5, 6]) for l in range(2)]
    lines += [_row(1, "chat", 0, "decode", l, [7, 8]) for l in range(2)]
    p = tmp_path / "t.tsv"
    p.write_text("".join(lines))
    return p


def test_token_major_complete_tokens_only(tsv):
    t, dropped = tsv_to_trace(tsv)
    assert dropped == 1
    assert list(t["prompt_id"]) == [0, 0, 0, 0, 1, 1]
    assert list(t["token_pos"]) == [1, 1, 2, 2, 0, 0]
    assert list(t["layer"]) == [0, 1, 0, 1, 0, 1]
    assert t["task"] == ["code"] * 4 + ["chat"] * 2
    assert t["experts"].dtype == np.int16 and t["experts"].shape == (6, 2)
    assert t["weights"].dtype == np.float16
    assert t["experts"][2].tolist() == [5, 6]


def test_phase_filter(tsv):
    t, dropped = tsv_to_trace(tsv, phase="decode")
    assert dropped == 0
    assert sorted(set(zip(t["prompt_id"].tolist(), t["token_pos"].tolist()))) == [(0, 2), (1, 0)]


def test_empty_phase_is_an_error(tmp_path):
    p = tmp_path / "e.tsv"
    p.write_text(HEADER)
    with pytest.raises(ValueError):
        tsv_to_trace(p)
