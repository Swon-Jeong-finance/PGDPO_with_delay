import numpy as np
import pytest

def test_p1_variants_load_and_validate():
    from pgdpo_delay.problems.p1.config import load_config
    m, s = load_config("main"), load_config("dp_small")
    assert m["H"] == 16 and s["H"] == 3 and "dp" in s
    assert m["bounds"][0] < m["bounds"][1]
    assert np.isclose(m["N"]*m["h"], m["T"])

def test_p1_unknown_config_raises():
    from pgdpo_delay.problems.p1.config import load_config
    with pytest.raises(KeyError):
        load_config("does_not_exist")


def test_p1_scientific_config_hash_binds_problem_not_optimizer():
    from copy import deepcopy
    from pgdpo_delay.problems.p1.config import (
        load_config,
        scientific_config_hash,
    )

    cfg = load_config("main_u")
    assert scientific_config_hash(cfg) == scientific_config_hash(load_config("main_u"))
    assert scientific_config_hash(cfg) != scientific_config_hash(load_config("main"))
    changed = deepcopy(cfg)
    changed["raw"]["problem"]["R"] *= 2.0
    assert scientific_config_hash(cfg) != scientific_config_hash(changed)

    budget_only = deepcopy(cfg)
    budget_only["raw"]["budgets"]["M"] *= 2
    budget_only["budgets"]["M"] *= 2
    assert scientific_config_hash(cfg) == scientific_config_hash(budget_only)

    optimizer_only = deepcopy(cfg)
    optimizer_only["raw"]["optimizer"] = {"lr": 1e-4}
    assert scientific_config_hash(cfg) == scientific_config_hash(optimizer_only)


def test_p1_estimator_contract_binds_unconstrained_variant():
    from pgdpo_delay.problems.p1 import contract
    from pgdpo_delay.problems.p1.config import load_config

    assert contract.CONTRACT_CONFIG == "main_u"
    assert load_config(contract.CONTRACT_CONFIG)["bounds"] is None

def test_canonical_shadowing_refused(tmp_path, monkeypatch):
    """review 2026-08-07 sec.4.9: a CWD copy of a canonical name with
    DIFFERENT content must be refused; an identical copy is tolerated."""
    from importlib.resources import files
    from pgdpo_delay.problems.p1.config import load_config
    canon = files("pgdpo_delay.configs").joinpath("p1", "main.yaml").read_text()
    d = tmp_path/"configs"/"p1"; d.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (d/"main.yaml").write_text(canon)            # identical: fine
    assert load_config("main")["H"] == 16
    (d/"main.yaml").write_text(canon.replace("gamma_u: 0.7", "gamma_u: 0.5"))
    with pytest.raises(RuntimeError):
        load_config("main")                      # differing shadow: refused

def test_p2_canonical_shadowing_refused(tmp_path, monkeypatch):
    """minfix review sec.4.4: symmetric regression test for the P2 loader."""
    from importlib.resources import files
    from pgdpo_delay.problems.p2.config import load_p2_config
    canon = files("pgdpo_delay.configs").joinpath("p2", "scaling.yaml").read_text()
    d = tmp_path/"configs"/"p2"; d.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (d/"scaling.yaml").write_text(canon)
    assert int(load_p2_config("scaling")["r_main"]) == 4
    (d/"scaling.yaml").write_text(canon.replace("r_main: 4", "r_main: 8"))
    with pytest.raises(RuntimeError):
        load_p2_config("scaling")

def test_cli_refuses_differing_canonical_base(tmp_path, monkeypatch):
    """minfix review sec.4.2: `main.py config --base main` must NOT read a
    stale differing CWD copy of a canonical base (the last shadowing route)."""
    import argparse
    from importlib.resources import files
    from pgdpo_delay.cli import derive_config
    canon = files("pgdpo_delay.configs").joinpath("p1", "main.yaml").read_text()
    d = tmp_path/"configs"/"p1"; d.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (d/"main.yaml").write_text(canon.replace("gamma_u: 0.7", "gamma_u: 0.5"))
    a = argparse.Namespace(problem="p1", base="main", name="derived_x",
                           set=["problem.R=0.2"], force=False)
    with pytest.raises(SystemExit):
        derive_config(a)                         # stale base: refused
    (d/"main.yaml").write_text(canon)            # identical copy: derivation ok
    derive_config(a)
    import yaml
    out = yaml.safe_load((d/"derived_x.yaml").read_text())
    assert out["problem"]["R"] == 0.2 and out["problem"]["gamma_u"] == 0.7

def test_p2_scaling_config_is_loader_source():
    from pgdpo_delay.problems.p2.config import load_p2_config
    from pgdpo_delay.problems.p2 import scaling
    raw = load_p2_config("scaling")
    assert scaling.T == raw["grid"]["T"] and scaling.R_MAIN == int(raw["r_main"])
    assert scaling.COEF["A"][raw["variant"]] is not None
