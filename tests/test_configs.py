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

def test_p2_scaling_config_is_loader_source():
    from pgdpo_delay.problems.p2.config import load_p2_config
    from pgdpo_delay.problems.p2 import scaling
    raw = load_p2_config("scaling")
    assert scaling.T == raw["grid"]["T"] and scaling.R_MAIN == int(raw["r_main"])
    assert scaling.COEF["A"][raw["variant"]] is not None
