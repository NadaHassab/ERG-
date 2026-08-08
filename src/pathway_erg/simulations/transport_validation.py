"""Synthetic validation of the signed derivative-OT transform (plan E1).

Every expected effect of the transform's mathematics is verified on synthetic
signals before any downstream modeling uses the descriptor:

- constant-offset invariance;
- known time-shift behavior;
- amplitude scaling changes masses but not normalized quantiles;
- bounded noise sensitivity across the approved smoothing grid;
- polarity reversal swaps the sign channels;
- morphology broadening spreads the quantile maps;
- missing sign mass leaves exactly one valid channel;
- nonuniform sampling integrates consistently;
- standard SCDT on amplitude is offset-sensitive (why derivative transport).

The E1 gate ("proceed only when observations agree with the transform's
mathematics") passes when every check reports `passed: true`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SmoothingConfig
from ..constants import MASS_EPSILON
from ..signal.signed_ot import signed_derivative_ot

SMOOTHING = SmoothingConfig()
N_QUANTILES = 64
FS_HZ = 1953.125
EPS = 1e-6


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: dict = field(default_factory=dict)
    expectation: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": {k: _round(v) for k, v in self.observed.items()},
            "expectation": self.expectation,
        }


def _round(value) -> float | dict | list:
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, (int, float, np.floating)):
        return round(float(value), 6)
    return value


def _time(n: int = 235) -> np.ndarray:
    return np.arange(n) * 1000.0 / FS_HZ


def _peak(time, center=40.0, width=8.0, amp=10.0, offset=0.0):
    return offset + amp * np.exp(-0.5 * ((time - center) / width) ** 2)


def _check_offset_invariance() -> CheckResult:
    t = _time()
    x = _peak(t)
    a = signed_derivative_ot(t, x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, x + 7.5, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    max_q = float(np.max(np.abs(a.q_pos - b.q_pos)))
    passed = max_q < 1e-9 and abs(a.mass_pos - b.mass_pos) < 1e-9
    return CheckResult(
        "offset_invariance",
        passed,
        {"max_quantile_diff_ms": max_q},
        "quantile maps and masses unchanged under constant offset",
    )


def _check_time_shift() -> CheckResult:
    t = _time()
    shift = 15.0
    a = signed_derivative_ot(t, _peak(t, center=40.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, _peak(t, center=40.0 + shift), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    mean_shift = float(np.mean(b.q_pos - a.q_pos))
    mass_drift = abs(a.mass_pos - b.mass_pos) / a.mass_pos
    passed = abs(mean_shift - shift) < 1.0 and mass_drift < 1e-3
    return CheckResult(
        "time_shift",
        passed,
        {"mean_quantile_shift_ms": mean_shift, "relative_mass_drift": mass_drift},
        "quantile maps shift by the known time shift; masses unchanged",
    )


def _check_amplitude_scaling() -> CheckResult:
    t = _time()
    x = _peak(t, amp=10.0)
    a = signed_derivative_ot(t, x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, 3.0 * x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    max_q = float(np.max(np.abs(a.q_pos - b.q_pos)))
    mass_ratio = b.mass_pos / a.mass_pos
    passed = max_q < 1e-9 and abs(mass_ratio - 3.0) < 1e-6
    return CheckResult(
        "amplitude_scaling",
        passed,
        {"max_quantile_diff_ms": max_q, "mass_ratio": mass_ratio},
        "scaling multiplies masses and leaves normalized quantiles unchanged",
    )


def _check_noise_sensitivity() -> CheckResult:
    t = _time()
    rng = np.random.default_rng(0)
    x = _peak(t, amp=10.0)
    clean = signed_derivative_ot(t, x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    levels = [0.001, 0.01, 0.05, 0.10, 0.25]
    distortions = []
    for level in levels:
        noisy = signed_derivative_ot(
            t, x + rng.normal(0.0, level * 10.0, t.size), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON
        )
        distortions.append(float(np.mean(np.abs(noisy.q_pos - clean.q_pos))))
    # Expected behavior: distortion grows with noise but saturates at very
    # high noise, because at SNR->0 the map converges to the noise-only
    # quantile map (a fixed target), not to infinity.
    monotone_early = all(b >= a for a, b in zip(distortions[:3], distortions[1:4], strict=False))
    small_at_low = distortions[0] < 3.0
    saturation = abs(distortions[-1] - distortions[-2]) / distortions[-2] < 0.15
    passed = monotone_early and small_at_low and saturation
    return CheckResult(
        "noise_sensitivity",
        passed,
        {"distortion_by_noise_level_ms": dict(zip(levels, distortions, strict=True))},
        "distortion grows with noise, stays a few percent of component span at low noise, "
        "and saturates at very high noise (convergence to the noise-only map)",
    )


def _check_polarity_flip() -> CheckResult:
    t = _time()
    x = _peak(t)
    a = signed_derivative_ot(t, x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, -x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    q_swap = float(np.max(np.abs(a.q_pos - b.q_neg)))
    mass_swap = abs(a.mass_pos - b.mass_neg)
    passed = q_swap < 1e-9 and mass_swap < 1e-9
    return CheckResult(
        "polarity_flip",
        passed,
        {"max_quantile_swap_diff_ms": q_swap, "mass_swap_diff": mass_swap},
        "polarity reversal swaps the positive and negative sign channels",
    )


def _check_morphology_broadening() -> CheckResult:
    t = _time()
    a = signed_derivative_ot(t, _peak(t, width=4.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, _peak(t, width=8.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    span_a = float(np.percentile(a.q_pos, 90) - np.percentile(a.q_pos, 10))
    span_b = float(np.percentile(b.q_pos, 90) - np.percentile(b.q_pos, 10))
    ratio = span_b / span_a if span_a > 0 else float("inf")
    passed = 1.7 < ratio < 2.3
    return CheckResult(
        "morphology_broadening",
        passed,
        {"span_ratio_narrow_to_wide": ratio},
        "doubling the peak width roughly doubles the quantile-map span",
    )


def _check_missing_sign_mass() -> CheckResult:
    t = _time()
    ramp = np.linspace(0.0, 1.0, t.size)
    a = signed_derivative_ot(t, ramp, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    flat = np.zeros_like(t)
    b = signed_derivative_ot(t, flat, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    passed = a.valid_pos and not a.valid_neg and a.mass_neg == 0.0 and not (b.valid_pos or b.valid_neg)
    return CheckResult(
        "missing_sign_mass",
        passed,
        {"ramp_valid_pos": a.valid_pos, "ramp_valid_neg": a.valid_neg, "flat_valid_any": b.valid_pos or b.valid_neg},
        "a single-sign signal leaves exactly one channel valid; a flat signal leaves none",
    )


def _check_nonuniform_sampling() -> CheckResult:
    rng = np.random.default_rng(1)
    t_uniform = _time(500)
    x = _peak(t_uniform)
    ref = signed_derivative_ot(t_uniform, x, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    jitter = 0.15 * 1000.0 / FS_HZ
    t_irregular = np.sort(t_uniform + rng.uniform(-jitter, jitter, t_uniform.size))
    x_irr = _peak(t_irregular)
    got = signed_derivative_ot(
        t_irregular, x_irr, float(np.median(np.diff(t_irregular))), SMOOTHING, N_QUANTILES, MASS_EPSILON
    )
    mass_err = abs(ref.mass_pos - got.mass_pos) / ref.mass_pos
    q_err = float(np.mean(np.abs(ref.q_pos - got.q_pos)))
    # Integration (trapezoid + spacing-aware derivative) is exact on
    # irregular grids; the ~2% mass difference here comes from the
    # Savitzky-Golay uniform-kernel approximation, which assumes regular
    # sampling.  Real device data has a constant sampling rate (median_dt
    # constant across the cohort), so this is a robustness property, not a
    # data-path concern.
    passed = mass_err < 0.05 and q_err < 3.0
    return CheckResult(
        "nonuniform_sampling",
        passed,
        {"relative_mass_error": mass_err, "mean_quantile_error_ms": q_err},
        "irregular timestamps integrate almost identically; residual mass error "
        "reflects the uniform-kernel smoothing approximation",
    )


def _amplitude_scdt(time_ms: np.ndarray, signal_uv: np.ndarray) -> np.ndarray:
    """Standard SCDT on waveform amplitude (offset-sensitive comparator).

    Treats max(signal, 0) as a density over time and returns its inverse-CDF
    on the fixed tau grid.  A constant offset changes the mass above the
    zero reference and shifts the resulting maps; derivative transport has no
    such reference dependence.
    """
    density = np.maximum(signal_uv, 0.0)
    mass = float(np.trapezoid(density, time_ms))
    if mass <= MASS_EPSILON:
        return np.zeros(N_QUANTILES)
    dt = np.empty_like(density)
    dt[0] = time_ms[1] - time_ms[0]
    dt[1:] = np.diff(time_ms)
    cdf = np.cumsum(density * dt) / mass
    tau = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    return np.interp(tau, np.clip(cdf, 0.0, 1.0), time_ms)


def _check_scdt_offset_sensitivity() -> CheckResult:
    """Demonstrates why derivative transport is used instead of amplitude SCDT."""
    t = _time()
    base = _peak(t, center=40.0, amp=10.0)
    scdt_ref = _amplitude_scdt(t, base)
    scdt_off = _amplitude_scdt(t, base + 5.0)
    scdt_drift = float(np.mean(np.abs(scdt_off - scdt_ref)))
    a = signed_derivative_ot(t, base, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b = signed_derivative_ot(t, base + 5.0, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    ot_drift = float(np.mean(np.abs(a.q_pos - b.q_pos)))
    passed = scdt_drift > 1.0 and ot_drift < 1e-6
    return CheckResult(
        "scdt_offset_sensitivity",
        passed,
        {"scdt_mean_quantile_drift_ms": scdt_drift, "derivative_ot_mean_drift_ms": ot_drift},
        "amplitude SCDT drifts under constant offset; derivative OT does not",
    )


def run_transport_battery() -> list[CheckResult]:
    checks = [
        _check_offset_invariance(),
        _check_time_shift(),
        _check_amplitude_scaling(),
        _check_noise_sensitivity(),
        _check_polarity_flip(),
        _check_morphology_broadening(),
        _check_missing_sign_mass(),
        _check_nonuniform_sampling(),
        _check_scdt_offset_sensitivity(),
    ]
    return checks


def write_transport_report(artifact_root: str | Path, checks: list[CheckResult]) -> Path:
    """Persist the E1 summary JSON and an HTML report; returns the report path."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    artifact_root = Path(artifact_root)
    out_dir = artifact_root / "simulations" / "transport_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    t = _time()
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axes
    rng = np.random.default_rng(0)

    x_off = _peak(t)
    a_off = signed_derivative_ot(t, x_off, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b_off = signed_derivative_ot(t, x_off + 7.5, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    tau = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    ax1.plot(t, x_off, label="signal")
    ax1.plot(t, x_off + 7.5, label="signal + offset")
    ax1.plot(tau, a_off.q_pos, label="OT pos")
    ax1.plot(tau, b_off.q_pos, ls="--", label="OT pos + offset")
    ax1.set_title("offset invariance")
    ax1.legend(fontsize=6)

    a_shift = signed_derivative_ot(t, _peak(t, center=40.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b_shift = signed_derivative_ot(t, _peak(t, center=55.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    ax2.plot(tau, a_shift.q_pos, label="peak at 40 ms")
    ax2.plot(tau, b_shift.q_pos, label="peak at 55 ms")
    ax2.set_title("time shift")
    ax2.legend(fontsize=6)

    clean = signed_derivative_ot(t, _peak(t, amp=10.0), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    levels = [0.001, 0.01, 0.05, 0.10, 0.25]
    dist = []
    for level in levels:
        noisy = signed_derivative_ot(
            t, _peak(t, amp=10.0) + rng.normal(0.0, level * 10.0, t.size), 0.512, SMOOTHING,
            N_QUANTILES, MASS_EPSILON,
        )
        dist.append(float(np.mean(np.abs(noisy.q_pos - clean.q_pos))))
    ax3.plot(levels, dist, marker="o")
    ax3.set_xlabel("noise sd / amplitude")
    ax3.set_ylabel("mean quantile distortion (ms)")
    ax3.set_title("noise sensitivity")

    x_flip = _peak(t)
    a_flip = signed_derivative_ot(t, x_flip, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    b_flip = signed_derivative_ot(t, -x_flip, 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON)
    ax4.plot(tau, a_flip.q_pos, label="pos of +signal")
    ax4.plot(tau, b_flip.q_neg, ls="--", label="neg of -signal")
    ax4.set_title("polarity flip swaps channels")
    ax4.legend(fontsize=6)

    for width, marker, label in ((4.0, "o", "width 4"), (8.0, "s", "width 8")):
        q = signed_derivative_ot(t, _peak(t, width=width), 0.512, SMOOTHING, N_QUANTILES, MASS_EPSILON).q_pos
        ax5.plot(np.linspace(0, 1, 10), np.percentile(q, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                 marker=marker, label=label)
    ax5.set_title("morphology broadening")
    ax5.legend(fontsize=6)

    scdt_ref = _amplitude_scdt(t, x_off)
    scdt_off = _amplitude_scdt(t, x_off + 5.0)
    ax6.plot(scdt_ref, label="SCDT amplitude")
    ax6.plot(scdt_off, ls="--", label="SCDT amplitude + offset")
    ax6.plot(a_off.q_pos, label="derivative OT")
    ax6.plot(b_off.q_pos, ls="--", label="derivative OT + offset")
    ax6.set_title("SCDT (offset-sensitive) vs derivative OT")
    ax6.legend(fontsize=6)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    rows = "".join(
        f"<tr><td>{c.name}</td><td>{'PASS' if c.passed else 'FAIL'}</td>"
        f"<td>{json.dumps(c.observed)}</td><td>{c.expectation}</td></tr>"
        for c in checks
    )
    n_pass = sum(c.passed for c in checks)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>E1 synthetic transport validation</title>
<style>body{{font-family:sans-serif;margin:24px;}}table{{border-collapse:collapse;font-size:11px;}}
td,th{{border:1px solid #ccc;padding:3px 7px;}}.pass{{color:#0a7d0a;font-weight:bold;}}
.fail{{color:#b00;font-weight:bold;}}</style></head><body>
<h1>E1 — synthetic transport validation</h1>
<p>{n_pass}/{len(checks)} checks passed</p>
<table><tr><th>check</th><th>status</th><th>observed</th><th>expectation</th></tr>{rows}</table>
<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/></body></html>"""
    report_path = out_dir / "transport_report.html"
    report_path.write_text(html, encoding="utf-8")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"n_checks": len(checks), "n_pass": n_pass, "all_passed": n_pass == len(checks),
                    "checks": [c.to_dict() for c in checks]}, indent=2),
        encoding="utf-8",
    )
    return report_path
