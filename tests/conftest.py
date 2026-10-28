from pathlib import Path

import pytest

from inbound_twin import Twin, generate_network, load_config, naive_plan, override

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_cfg():
    return load_config(ROOT / "config" / "base.yaml")


def run_twin(cfg, seed=1, days=5, net=None, plan=None, check=True):
    net = net or generate_network(cfg)
    plan = plan or naive_plan(net, cfg)
    return Twin(cfg, net, plan, seed=seed, days=days, check_invariants=check).run()


@pytest.fixture(scope="session")
def default_run_30(base_cfg):
    """30 days at the default config, invariants checked every 5 simulated minutes."""
    return run_twin(base_cfg, seed=1, days=30)


__all__ = ["run_twin", "override"]
