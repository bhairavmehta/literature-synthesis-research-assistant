import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lsra.calibration import PlattScaler, ConformalGate


def test_platt_monotonic_and_separates():
    p = PlattScaler().fit([0.1,0.2,0.3,0.45,0.5,0.6,0.7,0.8,0.9],
                          [0,0,0,1,1,1,1,1,1])
    assert p.transform(0.9) > p.transform(0.1)
    assert p.transform(0.9) > 0.5 > p.transform(0.1)


def test_conformal_threshold_in_range():
    g = ConformalGate(alpha=0.1).fit([0.5,0.6,0.7,0.8,0.9])
    assert 0.0 <= g.threshold <= 0.9
    assert g.accept(0.95) is True


def test_conformal_empty_safe():
    g = ConformalGate().fit([])
    assert g.accept(0.0) is True
