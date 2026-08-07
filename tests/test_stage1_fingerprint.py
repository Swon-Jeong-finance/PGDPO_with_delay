"""Torch-free regression checks for Stage-I scientific config identity."""

import numpy as np

from pgdpo_delay.core.artifacts import config_hash
from pgdpo_delay.core.stage1 import _hashable


def test_array_and_raw_fields_are_part_of_stage1_config_hash():
    base = {
        "variant": "A",
        "V": np.eye(3),
        "spec": {"a": np.array([1.0, 2.0, 3.0])},
        "mats": {"A": np.arange(9.0).reshape(3, 3)},
        "raw": {"problem": {"coefficient": 0.5}},
    }
    copied = {
        "variant": "A",
        "V": base["V"].copy(),
        "spec": {"a": base["spec"]["a"].copy()},
        "mats": {"A": base["mats"]["A"].copy()},
        "raw": {"problem": {"coefficient": 0.5}},
    }
    assert config_hash(_hashable(base)) == config_hash(_hashable(copied))

    changed_matrix = dict(copied)
    changed_matrix["V"] = copied["V"].copy()
    changed_matrix["V"][0, 0] += 1e-12
    assert config_hash(_hashable(base)) != config_hash(
        _hashable(changed_matrix))

    changed_raw = dict(copied)
    changed_raw["raw"] = {"problem": {"coefficient": 0.6}}
    assert config_hash(_hashable(base)) != config_hash(_hashable(changed_raw))
