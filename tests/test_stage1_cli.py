"""Torch-free CLI checks for the common Stage-I scheduler entry point."""

import json
import sys

import pytest

from pgdpo_delay import cli


def test_stage1_multigpu_dry_run_is_side_effect_free(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(sys, "argv", [
        "pgdpo-delay", "run",
        "--problem", "p1",
        "--stage", "1",
        "--protocol", "p1_u_smoke",
        "--seeds", "1,2,3,4,5",
        "--devices", "cuda:0,cuda:1,cuda:2",
        "--out-root", str(tmp_path),
        "--dry-run",
    ])
    plan = cli.main()
    assert plan["seeds"] == [1, 2, 3, 4, 5]
    assert plan["devices"] == ["cuda:0", "cuda:1", "cuda:2"]
    assert plan["problem_config"] == "main_u"
    assert len(plan["run_fingerprint"]) == 24
    assert not any(tmp_path.iterdir())
    assert "no subprocesses or artifacts" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--seeds", "1,1", "duplicates"),
        ("--seeds", "1,-2", "nonnegative"),
        ("--devices", "cuda", "explicitly indexed"),
        ("--devices", "cuda:0,cuda:00", "duplicates"),
        ("--devices", "gpu:0", "must be cpu or explicitly indexed"),
        ("--devices", "cpu,cuda:0", "may not mix"),
    ],
)
def test_stage1_cli_rejects_ambiguous_rosters(tmp_path, monkeypatch, flag,
                                               value, message):
    argv = [
        "pgdpo-delay", "run", "--problem", "p1",
        "--protocol", "p1_u_smoke", "--seeds", "1,2",
        "--devices", "cuda:0", "--out-root", str(tmp_path), "--dry-run",
    ]
    argv[argv.index(flag) + 1] = value
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match=message):
        cli.main()


def test_run_spec_refuses_cross_protocol_reuse(tmp_path):
    first = {"problem": "p1", "training": {"batch": 4}}
    second = {"problem": "p1", "training": {"batch": 8}}
    path = cli._ensure_run_spec(tmp_path, first, "fingerprint-a")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["run_fingerprint"] == "fingerprint-a"
    with pytest.raises(SystemExit, match="different Stage-I protocol"):
        cli._ensure_run_spec(tmp_path, second, "fingerprint-b")
