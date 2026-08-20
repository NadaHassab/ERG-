#!/usr/bin/env python3
"""Compare frozen vs unfrozen SSL-init results side by side."""
import json
from pathlib import Path

RESULTS = Path("artifacts/results")


def load(path):
    return json.load(open(path)) if path.exists() else None


def row(name, path):
    d = load(path)
    if not d:
        return None
    return {
        "name": name,
        "LEOP": f"{d['LEOP']['roc_auc']:.3f} [{d['LEOP']['roc_auc_ci_low']:.3f}, {d['LEOP']['roc_auc_ci_high']:.3f}]",
        "PERG": f"{d['PERG']['roc_auc']:.3f} [{d['PERG']['roc_auc_ci_low']:.3f}, {d['PERG']['roc_auc_ci_high']:.3f}]",
    }


models = [
    ("separate baseline (e6)", RESULTS / "separate_raw_ot_hierarchical_v1/metrics.json"),
    ("SSL-init frozen (e7b)", RESULTS / "separate_raw_ot_hierarchical_sslinit_v1/metrics.json"),
    ("SSL-init UNFROZEN (e7c)", RESULTS / "separate_raw_ot_hierarchical_sslinit_unfrozen_v1/metrics.json"),
    ("SSL 4-domain ext (e9)", RESULTS / "separate_raw_ot_hierarchical_sslinit_external_v1/metrics.json"),
]

print(f"{'Model':<28} {'LEOP':<28} {'PERG':<28}")
print("-" * 84)
for name, path in models:
    r = row(name, path)
    if r is None:
        print(f"{name:<28} {'— (running/pending)':<28} {'—':<28}")
    else:
        print(f"{r['name']:<28} {r['LEOP']:<28} {r['PERG']:<28}")