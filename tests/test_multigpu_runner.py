"""Torch-free tests for the dynamic seed subprocess scheduler."""

import json
from pathlib import Path
import sys

import pytest

import pgdpo_delay.core.runner as runner_module
from pgdpo_delay.core.runner import run_seed_processes


_WORKER = r"""
import hashlib
import json
from pathlib import Path
import sys
import time

stage = Path(sys.argv[1])
seed = int(sys.argv[2])
device = sys.argv[3]
fingerprint = sys.argv[4]
delay = float(sys.argv[5])
mode = sys.argv[6]

print(f"worker seed={seed} device={device}", flush=True)
print("worker stderr", file=sys.stderr, flush=True)
time.sleep(delay)
if mode == "exit":
    (stage / "partial.txt").write_text("kept after failure", encoding="utf-8")
    raise SystemExit(7)

config = {
    "problem": "fake",
    "method": "fake_worker",
    "seed": seed,
    "device": device,
}
encoded = json.dumps(config, sort_keys=True, default=str).encode()
config_hash = hashlib.sha256(encoded).hexdigest()[:16]
payloads = {
    "manifest.json": {
        "problem": "fake",
        "method": "fake_worker",
        "config_hash": config_hash,
        "seeds": {"train": seed},
        "extra": {"run_fingerprint": fingerprint, "required_metrics": ["score"]},
    },
    "config.json": config,
    "metrics.json": {"score": float(seed), "device": device},
    "status.json": {
        "status": "COMPLETE",
        "problem": "fake",
        "method": "fake_worker",
        "seed": seed,
        "run_fingerprint": fingerprint,
    },
}
if mode == "missing":
    payloads.pop("metrics.json")
for name, payload in payloads.items():
    (stage / name).write_text(json.dumps(payload), encoding="utf-8")
"""


def _worker_script(tmp_path: Path) -> Path:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER, encoding="utf-8")
    return script


def _builder(script, fingerprint, delays=None, modes=None, calls=None):
    delays = delays or {}
    modes = modes or {}

    def build(seed, device, stage_dir):
        if calls is not None:
            calls.append((seed, device, stage_dir))
        return [
            sys.executable,
            str(script),
            str(stage_dir),
            str(seed),
            device,
            fingerprint,
            str(delays.get(seed, 0.0)),
            modes.get(seed, "complete"),
        ]

    return build


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_dynamic_refill_publishes_each_seed_and_summary(tmp_path):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "p1-stage1-fixed-config"
    calls = []
    summary = run_seed_processes(
        _builder(
            script,
            fingerprint,
            # Device zero becomes free well before devices one and two.
            delays={1: 0.03, 2: 0.45, 3: 0.45, 4: 0.0, 5: 0.0},
            calls=calls,
        ),
        seeds=[1, 2, 3, 4, 5],
        devices=["cuda:0", "cuda:1", "cuda:2"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        poll_interval=0.005,
    )

    assert summary["status"] == "COMPLETE"
    assert summary["counts"] == {
        "complete": 5, "failed": 0, "skipped": 0, "cancelled": 0}
    # Slots are initially filled in order.  Later seeds must be dispatched as
    # slots become free, but their exact physical label is timing-dependent.
    assert [(seed, device) for seed, device, _ in calls[:3]] == [
        (1, "cuda:0"), (2, "cuda:1"), (3, "cuda:2")]
    assert [seed for seed, _, _ in calls] == [1, 2, 3, 4, 5]
    assert all(
        device in {"cuda:0", "cuda:1", "cuda:2"}
        for _, device, _ in calls[3:]
    )

    stage_paths = [stage for _, _, stage in calls]
    assert len(set(stage_paths)) == 5
    for seed in (1, 2, 3, 4, 5):
        seed_dir = outroot / f"seed{seed}"
        assert seed_dir.is_dir()
        status = _read_json(seed_dir / "status.json")
        assert status["status"] == "COMPLETE"
        assert status["seed"] == seed
        assert status["run_fingerprint"] == fingerprint
        log = (seed_dir / "run.log").read_text(encoding="utf-8")
        assert f"worker seed={seed}" in log
        assert "worker stderr" in log
    assert _read_json(outroot / "seed4" / "config.json")["device"] in {
        "cuda:0", "cuda:1", "cuda:2"
    }
    assert _read_json(outroot / "run_summary.json")["counts"]["complete"] == 5
    assert not any((outroot / ".staging").iterdir())


def test_failures_are_retained_and_do_not_block_other_seeds(tmp_path):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "failure-test"
    summary = run_seed_processes(
        _builder(
            script,
            fingerprint,
            modes={10: "exit", 11: "missing", 12: "complete"},
        ),
        seeds=[10, 11, 12],
        devices=["cuda:0", "cuda:1"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        poll_interval=0.005,
    )

    assert summary["status"] == "FAILED"
    assert summary["counts"] == {
        "complete": 1, "failed": 2, "skipped": 0, "cancelled": 0}
    assert not (outroot / "seed10").exists()
    assert not (outroot / "seed11").exists()
    assert (outroot / "seed12").is_dir()

    failed = sorted((outroot / "failed").iterdir())
    assert len(failed) == 2
    assert len({path.name for path in failed}) == 2
    payloads = [_read_json(path / "scheduler_failure.json") for path in failed]
    assert {payload["seed"] for payload in payloads} == {10, 11}
    assert any((path / "partial.txt").exists() for path in failed)
    assert all((path / "run.log").is_file() for path in failed)
    errors = {record["seed"]: record["error"] for record in summary["seeds"]}
    assert "exited with code 7" in errors[10]
    assert "missing regular files: metrics.json" in errors[11]


def test_live_output_prefixes_terminal_but_preserves_seed_log(
        tmp_path, capsys):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "live-output-test"
    summary = run_seed_processes(
        _builder(script, fingerprint, delays={4: 0.08}),
        seeds=[4],
        devices=["cuda:2"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        poll_interval=0.005,
        live_output=True,
    )

    assert summary["status"] == "COMPLETE"
    terminal = capsys.readouterr().out
    prefix = "[seed=4 device=cuda:2]"
    assert terminal.count(f"{prefix} worker seed=4 device=cuda:2") == 1
    assert terminal.count(f"{prefix} worker stderr") == 1
    log = (outroot / "seed4" / "run.log").read_text(encoding="utf-8")
    assert prefix not in log
    assert log.splitlines() == [
        "worker seed=4 device=cuda:2",
        "worker stderr",
    ]


def test_resume_skips_only_matching_complete_and_default_refuses(tmp_path):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "resume-test"
    run_seed_processes(
        _builder(script, fingerprint),
        seeds=[7],
        devices=["cuda:0"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        poll_interval=0.005,
    )

    calls = []
    with pytest.raises(FileExistsError, match="target already exists"):
        run_seed_processes(
            _builder(script, fingerprint, calls=calls),
            seeds=[7],
            devices=["cuda:0"],
            outroot=outroot,
            run_fingerprint=fingerprint,
        )
    assert calls == []

    resumed = run_seed_processes(
        _builder(script, fingerprint, calls=calls),
        seeds=[7],
        devices=["cuda:0"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        resume=True,
    )
    assert calls == []
    assert resumed["counts"] == {
        "complete": 0, "failed": 0, "skipped": 1, "cancelled": 0}
    assert resumed["seeds"][0]["status"] == "SKIPPED"

    status_path = outroot / "seed7" / "status.json"
    status = _read_json(status_path)
    status["run_fingerprint"] = "different-run"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(FileExistsError, match="run_fingerprint mismatch"):
        run_seed_processes(
            _builder(script, fingerprint, calls=calls),
            seeds=[7],
            devices=["cuda:0"],
            outroot=outroot,
            run_fingerprint=fingerprint,
            resume=True,
        )
    assert calls == []


def test_resume_rejects_tampered_config_even_with_complete_status(tmp_path):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "tamper-test"
    run_seed_processes(
        _builder(script, fingerprint),
        seeds=[9],
        devices=["cuda:0"],
        outroot=outroot,
        run_fingerprint=fingerprint,
        poll_interval=0.005,
    )

    config_path = outroot / "seed9" / "config.json"
    config = _read_json(config_path)
    config["device"] = "cuda:2"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FileExistsError, match="config_hash"):
        run_seed_processes(
            _builder(script, fingerprint),
            seeds=[9],
            devices=["cuda:0"],
            outroot=outroot,
            run_fingerprint=fingerprint,
            resume=True,
        )


def test_scheduler_rejects_negative_or_duplicate_seed_rosters(tmp_path):
    script = _worker_script(tmp_path)
    builder = _builder(script, "input-test")
    with pytest.raises(ValueError, match="nonnegative"):
        run_seed_processes(
            builder, seeds=[-1], devices=["cuda:0"],
            outroot=tmp_path / "negative", run_fingerprint="input-test")
    with pytest.raises(ValueError, match="unique"):
        run_seed_processes(
            builder, seeds=[1, 1], devices=["cuda:0"],
            outroot=tmp_path / "duplicate", run_fingerprint="input-test")


def test_keyboard_interrupt_reaps_children_and_writes_failed_summary(
        tmp_path, monkeypatch):
    script = _worker_script(tmp_path)
    outroot = tmp_path / "runs"
    fingerprint = "cancel-test"
    children = []
    real_popen = runner_module.subprocess.Popen
    real_sleep = runner_module.time.sleep

    def capture_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def interrupt_poll_wait(_seconds):
        # Restore the real sleep before unwinding: subprocess.wait(timeout=...)
        # also relies on the process-wide time module during cleanup.
        monkeypatch.setattr(runner_module.time, "sleep", real_sleep)
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner_module.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(runner_module.time, "sleep", interrupt_poll_wait)

    with pytest.raises(KeyboardInterrupt):
        run_seed_processes(
            _builder(script, fingerprint, delays={21: 30.0, 22: 30.0}),
            seeds=[21, 22, 23],
            devices=["cuda:0", "cuda:1"],
            outroot=outroot,
            run_fingerprint=fingerprint,
            poll_interval=0.005,
            cancel_timeout=0.2,
        )

    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    summary = _read_json(outroot / "run_summary.json")
    assert summary["status"] == "FAILED"
    assert summary["counts"] == {
        "complete": 0, "failed": 2, "skipped": 0, "cancelled": 1}
    assert [record["status"] for record in summary["seeds"]] == [
        "FAILED", "FAILED", "CANCELLED"]
    failed = list((outroot / "failed").iterdir())
    assert len(failed) == 2
    assert all((path / "scheduler_failure.json").is_file() for path in failed)
    assert not any((outroot / ".staging").iterdir())


def test_system_exit_during_command_build_is_retained_and_summarized(tmp_path):
    outroot = tmp_path / "runs"

    def exit_builder(_seed, _device, _stage_dir):
        raise SystemExit(9)

    with pytest.raises(SystemExit) as caught:
        run_seed_processes(
            exit_builder,
            seeds=[31],
            devices=["cuda:0"],
            outroot=outroot,
            run_fingerprint="builder-exit",
        )
    assert caught.value.code == 9
    summary = _read_json(outroot / "run_summary.json")
    assert summary["status"] == "FAILED"
    assert summary["counts"] == {
        "complete": 0, "failed": 1, "skipped": 0, "cancelled": 0}
    failed = list((outroot / "failed").iterdir())
    assert len(failed) == 1
    failure = _read_json(failed[0] / "scheduler_failure.json")
    assert failure["seed"] == 31
    assert "SystemExit: 9" in failure["error"]
