"""E1 synthetic transport validation battery (plan Section 7.8, experiment E1)."""

from __future__ import annotations

import numpy as np

from pathway_erg.simulations.transport_validation import (
    _amplitude_scdt,
    _peak,
    run_transport_battery,
)


def test_every_battery_check_passes_with_fixed_seeds():
    checks = run_transport_battery()
    names = {c.name for c in checks}
    assert names == {
        "offset_invariance",
        "time_shift",
        "amplitude_scaling",
        "noise_sensitivity",
        "polarity_flip",
        "morphology_broadening",
        "missing_sign_mass",
        "nonuniform_sampling",
        "scdt_offset_sensitivity",
    }
    for check in checks:
        assert check.passed, f"{check.name} failed: {check.observed}"


def test_scdt_is_offset_sensitive_but_derivative_ot_is_not():
    t = np.arange(235) * 1000.0 / 1953.125
    base = _peak(t, center=40.0, amp=10.0)
    drift = float(np.mean(np.abs(_amplitude_scdt(t, base + 5.0) - _amplitude_scdt(t, base))))
    assert drift > 1.0


def test_battery_is_deterministic():
    first = [(c.name, c.observed) for c in run_transport_battery()]
    second = [(c.name, c.observed) for c in run_transport_battery()]
    assert first == second
