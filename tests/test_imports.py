def test_package_imports():
    import pgdpo_delay
    from pgdpo_delay.core import structured, artifacts
    from pgdpo_delay.problems.p1 import oracle, config, dynamics
    from pgdpo_delay.problems.p2 import oracle as p2_oracle
    assert config.load_config("main")["H"] == 16
