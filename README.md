# PATH-ERG — Pathway-Aware Transfer for Heterogeneous Electroretinography

Repository implementing the master plan in
`MASTER_PLAN_PATHWAY_AWARE_SIGNED_OT.md`.

Working paper title: *Pathway-Constrained Partial Transfer Across Unpaired
Retinal Electrophysiology Protocols*.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pathway_erg.cli audit --config configs/data/local.yaml
.venv/bin/python -m pathway_erg.cli build-data \
  --data configs/data/local.yaml \
  --preprocessing configs/preprocessing/reference.yaml
```

## Status

Phase 1 (repository, environment, immutable raw audit) and Phase 2 (identities,
labels, schema, folds) are in progress. See `CHANGELOG.md` and Section 26 of the
master plan for gates.
