# Changelog

## 0.1.0 — 2026-08-01

- Repository skeleton: pyproject, src layout, CLI, configs.
- Immutable raw-file audit with SHA-256 checksums.
- LEOP typed parser (participants, recordings, ERG/OP pairing).
- PERG typed parser (multi-session column triplets, explicit NA handling,
  no session averaging).
- PERG repeat-link identity resolution via union-find.
- Versioned diagnosis mapping and label construction.
- Deterministic canonical data build to Parquet + Zarr.
- Nested grouped outer/inner folds with leakage assertions.
- Hard technical validity and fold-fitted QC scaffolding.
- Offset, smoothing, landmark, segmentation, and resampling pipeline.
- Signed derivative optimal transport descriptor with property tests.
