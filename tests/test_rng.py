import subprocess
import sys

import numpy as np
import pytest

from inbound_twin.rng import Kind, keyed_rng, lognormal_mean_cv


def test_same_key_same_draw():
    a = keyed_rng(5, Kind.TRAVEL, 3, 1, 2, 0, 40).standard_normal(4)
    b = keyed_rng(5, Kind.TRAVEL, 3, 1, 2, 0, 40).standard_normal(4)
    assert np.array_equal(a, b)


def test_same_key_same_draw_in_another_process():
    code = ("from inbound_twin.rng import Kind, keyed_rng;"
            "print(repr(keyed_rng(5, Kind.UNLOAD, 2, 7, 1).random()))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert float(out.stdout) == keyed_rng(5, Kind.UNLOAD, 2, 7, 1).random()


def test_different_keys_are_independent():
    draws = np.array([keyed_rng(1, Kind.TRAVEL, 0, 1, 2, 0, b).standard_normal() for b in range(4000)])
    assert abs(draws.mean()) < 0.06
    assert abs(draws.std() - 1) < 0.05
    assert abs(np.corrcoef(draws[:-1], draws[1:])[0, 1]) < 0.06
    other = keyed_rng(1, Kind.UNLOAD, 0, 1, 2, 0, 0).standard_normal()
    assert other != draws[0]


def test_negative_key_rejected():
    with pytest.raises(ValueError):
        keyed_rng(1, Kind.TRAVEL, -1)


def test_lognormal_mean_cv():
    rng = np.random.default_rng(0)
    x = np.array([lognormal_mean_cv(rng, 20.0, 0.3) for _ in range(20000)])
    assert abs(x.mean() - 20.0) < 0.2
    assert abs(x.std() / x.mean() - 0.3) < 0.01
    assert lognormal_mean_cv(rng, 20.0, 0.0) == 20.0
