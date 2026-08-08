# PATH-ERG — Progress Log

**Project:** PATH-ERG — Pathway-Aware Transfer for Heterogeneous Electroretinography
**Working paper:** *Pathway-Constrained Partial Transfer Across Unpaired Retinal Electrophysiology Protocols*
**Master plan:** `MASTER_PLAN_PATHWAY_AWARE_SIGNED_OT.md` (authoritative blueprint — 10 phases, 36 sections)
**Changelog:** `CHANGELOG.md` (release history)
**Last updated:** 2026-08-08

> This file is a plain-language log of what is done, what was changed and why,
> and what the findings mean. When an error appears, read the "Watch list" and
> "How to verify" sections first, then check the plan section for the phase.

---

## 1. What the project is (in one paragraph)

Two unpaired retinal electrophysiology datasets exist:
- **LEOPs** — flash ERG from 253 people (Control / ASD / ASD+ADHD), ~1,953 Hz,
  5,309 flash curves + 4,434 oscillatory potentials (OPs).
- **PERG-IOBA** — pattern ERG from 304 canonical subjects, 336 visits,
  677 bilateral sessions, 1,354 eye curves, 1,700 Hz (Normal / Abnormal).

They measure partially overlapping retinal biology with **different protocols,
different labels, and no shared patients**. The core scientific question: can a
model share only the *late inner-retinal (RGC-enriched)* waveform region between
the two protocols ("pathway-constrained partial sharing"), instead of sharing
everything (naive) or nothing (separate models)? Supporting math: a signed
derivative optimal-transport (sOT) descriptor; supporting statistics: strict
participant-level evaluation so thousands of repeated curves are never counted
as independent people.

---

## 2. Phase status vs. master plan (Section 26)

| Phase | What it covers | Status |
|---|---|---|
| 1 | Repository, environment, immutable raw audit | **DONE** |
| 2 | Identities, labels, schema, nested folds | **DONE** |
| 3 | Preprocessing, landmarks, components, QC | **DONE (pending review of findings below)** |
| 4 | Signed OT + synthetic simulations (E1, E2) | **DONE** |
| 5 | Simple classical baselines (E0/E4) + VMD | **PARTIAL** — classical baselines done; VMD not yet implemented |
| 6 | Separate hierarchical neural models | NOT STARTED |
| 7 | Joint SSL + pathway routing | NOT STARTED |
| 8 | Graph controls + label efficiency | NOT STARTED |
| 9 | Robustness, statistics, interpretation | NOT STARTED |
| 10 | Paper + release | NOT STARTED |

---

## 3. What has been done (in order)

### 3.1 Repository skeleton (Phase 1)
- `pyproject.toml` with pinned deps, `src/pathway_erg` package layout,
  typed CLI (`src/pathway_erg/cli.py`), config files under `configs/`.
- **Why:** master plan Section 20 requires all authoritative code in `src`, no
  notebook-driven pipeline, no mutable global preprocessing state.

### 3.2 Immutable raw-file audit (Phase 1)
- Recursively hashed every raw source file (SHA-256) → `raw_files.parquet`,
  `raw_audit.json`, `license_report.md`.
- **Why:** reproducibility — proves raw data are untouched and versions known.

### 3.3 LEOP typed parser (Phase 2)
- Parses participant JSONs into typed `SubjectRecord` / `WaveformRecord`.
- ERG ↔ OP pairing, protocol (9-step / 2-step / LA3), stimulus, eye, electrode,
  supplied features kept.
- Handles the known trap: 5,309 rows but only 5,243 unique `wave_id` values —
  global IDs are built to be collision-safe, `wave_id` alone is never the key.

### 3.4 PERG typed parser (Phase 2)
- Parses 336 visits, multi-session column triplets `TIME_k / RE_k / LE_k`,
  explicit NA handling (never defaulting unknown diagnoses to a class),
  **no session averaging** (old prototype code averaged — that was an error
  being fixed, see master plan Section 16.5).
- Timestamps converted to elapsed ms; sampling ~1,700 Hz retained natively.

### 3.5 PERG identity resolution (Phase 2)
- `rep_record` fields can contain **multiple** record IDs → resolved as an
  undirected graph via union-find → **304 canonical subjects**
  (281 singles, 19 pairs, 1 triple, 1 quadruple, 2 groups of 5).
- **Why:** repeat-linked visits placed in different folds would be direct
  leakage. This is one of the highest-impact fixes in the project.

### 3.6 Versioned label mapping (Phase 2)
- LEOP: Control vs ASD primary; ASD+ADHD excluded from primary training.
- PERG: `diagnosis1 == "Normal"` → 0, every other nonempty diagnosis → 1;
  null diagnosis → ineligible. Every raw label explicitly mapped in
  `artifacts/data/manifests/diagnosis_mapping_perg.csv` (+ `.meta.json`).
- **Why:** label mapping is protocol, not hidden loader logic; no fallback class.

### 3.7 Deterministic canonical data build (Phase 2)
- One `build-data` command → canonical tables (participants, visits, sessions,
  recordings, components, labels) + Zarr arrays (`raw_curves.zarr`).
- All expected counts verified: 253 / 5,309 / 4,434 / 304 / 336 / 677 / 1,354 —
  **every check in the build manifest matches the plan** (`build_manifest.json`).

### 3.8 Nested grouped folds (Phase 2)
- 5 outer folds grouped by LEOP participant / PERG canonical subject;
  4 grouped inner folds; deterministic + stratified.
- Automated leakage assertions: no subject in two partitions, no checksum
  duplicates, ERG/OP pairs together, repeat-linked visits together.
- Fold-safe scalers fitted on outer-training people only (`scalers_fold*.json`).

### 3.9 Validity and QC (Phase 3)
- Hard technical validity: rejects only structurally unusable curves
  (empty arrays, time–amplitude mismatch, <95 % finite, broken linkage, etc.).
- Fold-fitted QC thresholds (never fit on test folds) →
  `qc_thresholds_by_fold.json`, `population_masks.parquet`, three populations:
  all-valid (589), high-QC (406), complete (509).
- **Why:** low amplitude / flat morphology can be disease biology, so QC flags
  rather than deletes.

### 3.10 Signal pipeline (Phase 3)
- Offset policy (LEOP: pre-stimulus median; PERG: none — sOT is offset-invariant),
  Savitzky–Golay smoothing copy kept separate from raw, landmark detection
  (LEOP a/b/late-trough; PERG N35/P50/N95) with fallbacks + confidence,
  component segmentation into 6 biological regions, PCHIP resampling to
  128 points with valid masks, **no extrapolation**.
- Output: `components.parquet` (23,069 components), `component_curves.zarr`,
  visual QA report (`artifacts/qa/qa_report.html`, 419 visual samples across
  151 strata cells).

### 3.11 Signed derivative optimal transport (Phase 4)
- Pure deterministic transform: derivative split into positive/negative
  variation, masses + 64 quantiles per sign + physical features (Section 7 of
  plan). Cached in `signed_ot.zarr`.
- **Why:** ERGs are signed signals; classical SCDT/OT assume nonnegative mass
  and drift under constant offsets — the derivative version does not.

### 3.12 Simulations (Phase 4)
- **E1 transport validation** — 9/9 checks pass (see findings).
- **E2 partial-sharing bias-variance simulation** — reproduces the theory:
  full sharing wins at zero mismatch, oracle partial sharing wins at moderate
  mismatch, wrong graph causes negative transfer, learned gates track the
  oracle level.

### 3.13 Classical baselines (Phase 5, E0/E4)
- Full suite ran on locked folds with cluster-bootstrap CIs
  (`artifacts/results/baselines/metrics.json` + `predictions.parquet`):
  prevalence, metadata, availability/missingness, quality/QC, clinical
  features, FPCA, SCDT, derivative-OT, raw RBF, wavelet scattering — each with
  logistic regression / RBF-SVM / gradient boosting.

### 3.14 Quality gates currently passing
- All **69 tests** pass (`pytest`).
- E1: 9/9 transport checks. Extrapolation audit: no problems.
- Build, split, and QA manifests all complete.

### 3.15 Phase 0 — legacy freeze and v2 pipeline kickoff (2026-08-02)
- **Why:** the pipeline needs deep correctness fixes (participant-level splits
  for LEOP, slot-based features, flag semantics, real spectral bands, valid
  SCDT, leakage-free nested CV) without ever breaking or silently changing the
  already-published numbers. The pre-freeze behavior is therefore frozen and
  kept runnable; all corrected work happens in parallel in `baselines.py`
  ("v2").
- Frozen snapshot: `src/pathway_erg/models/legacy_baselines.py` (verbatim copy
  of `baselines.py`, only its output directory is parameterized).
  Dispatch: `legacy: true` configs route there; `legacy: false` (default)
  routes to the evolving v2 code.
- Output contract: legacy runs write only to
  `artifacts/results/baselines_legacy_v1/`; v2 writes to
  `artifacts/results/baselines_v2/`; the original
  `artifacts/results/baselines/` directory is never touched.
- Configs: `configs/experiments/e4_baselines_legacy.yaml` (legacy) and
  `configs/experiments/e4_baselines.yaml` (renamed to experiment
  `e4_baselines_v2`, corrected pipeline).
- Regression tests: `tests/models/test_legacy_freeze.py` — dispatch,
  determinism, versioned output paths, original dir untouched, snapshot
  self-containment, and a bit-exact fixture (LEOP fold 0: clinical_logreg
  AUROC 0.645833…, prevalence 0.5) locking the frozen code forever.
- Pre-freeze artifact fingerprint (the numbers v2 must eventually beat fairly):
  60 methods, 17,040 prediction rows, `metrics.json` sha256
  `e91bce45c51dae83`.

### 3.16 LEOP cohort experiments (Phase 1 of v2 plan, 2026-08-02)
- New `leop_cohort` config field + `src/pathway_erg/models/leop_cohorts.py`
  (pure mask helpers, no hard-coded counts — cohort sizes are computed from
  the recordings table and logged).
- **`primary_nine_step`** (config `e4_baselines_v2_leop_primary.yaml`):
  Control/ASD only, nine-step recordings only, one prediction per participant.
  Real cohort: **160 subjects (72 ASD + 88 Control)** — the other 72
  supervised subjects have no nine-step recording and are excluded with
  logged counts; 2-step/LA3 rows never enter any feature family (curves,
  valid-mask, sOT and components tables are all filtered consistently).
  Availability/protocol-count shortcuts are **forbidden** (pipeline raises).
- **`secondary_all_protocols`** (config `e4_baselines_v2_leop_secondary.yaml`):
  all 232 eligible subjects, protocols used explicitly (per-protocol counts
  logged; slot representation arrives in Phase 2), plus the
  protocol/availability/QC shortcut baselines for confound quantification.
- Metrics are namespaced `LEOP_primary_nine_step/...` and
  `LEOP_secondary_all_protocols/...`; predictions carry a `cohort` column;
  PERG keys/predictions are untouched by any LEOP cohort.
- Results (patient-level AUROC, 5 outer folds):
  - Primary: clinical+demog logreg 0.721 (sex confound, see 4.5), clinical
    logreg 0.675, quality histgb 0.669, scattering 0.659, scdt_logreg 0.461,
    prevalence 0.448 (n=160, 72 pos).
  - Secondary: **availability histgb 0.782** — the protocol/QC shortcut still
    dominates biology on the v2 pipeline, exactly the confound the plan says
    to quantify before any neural modeling.
- Regression tests: `tests/models/test_leop_cohorts.py` (8 tests) — unit
  masks, nine-step-only filtering, availability ban, namespaced metrics,
  PERG isolation, key-shape regression.

### 3.17 Fixed slot features (Phase 2 of v2 plan, 2026-08-02)
- New `src/pathway_erg/models/slot_features.py`: the fixed slot grid
  `component_type x eye x intensity x protocol` (component_type =
  L_EARLY_A/L_A_TO_B/L_LATE/L_OP/P_EARLY/P_LATE; intensity quantized to one
  decimal, 113.04 -> 113.0; protocol explicit). Grid sizes: 72 for the LEOP
  9-step primary cohort, 100+ for the all-protocol cohort, 4 for PERG
  (P_EARLY/P_LATE x RE/LE).
- Every feature is strictly within-slot: median and MAD of the 13 component
  physical features + per-slot n and flagged rate (28 values/slot). No
  averaging across components/eyes/intensities/protocols — verified by
  synthetic tests. Missing slots stay NaN; imputation happens inside the CV
  loops only (`_fit_transform_features`), never in the builder.
- Method `slot` registered as `slot_logreg` (grid size forced the narrow
  model set); runs inside the cohort pipelines with cohort-filtered tables.
- Results (patient-level AUROC, 5 outer folds):
  - Primary 9-step cohort (n=160): slot_logreg **0.666** (vs clinical 0.675,
    clinical+demog 0.721).
  - Secondary all-protocols (n=232): slot_logreg **0.727** (vs availability
    shortcut 0.782 — the protocol shortcut still wins).
- Tests: `tests/models/test_slot_features.py` (7 tests) — intensity
  quantization, no cross-slot averaging, exact median/MAD, missing-slot NaN,
  grid-vs-domain, end-to-end run, PERG slot shape.

### 3.18 GPU backend + faster inner selection (2026-08-02)
- cuML 26.06.00 installed (`pip install cuml-cu12`, NVIDIA wheels) for the
  RTX 3050. New `use_gpu: bool` config flag: logreg/SVM fits (inner-fold grid
  search + refits) run on cuML; histgb stays scikit-learn. Graceful fallback
  to sklearn per estimator when cuML is absent, recorded in the methodology
  note (`gpu_backend`).
- GPU SVM uses the same deterministic single-model
  `CalibratedClassifierCV(ensemble=False)` pattern as the CPU path; GPU logreg
  maps l1_ratio to penalty (l2/l1/elasticnet) with the QN solver. A GPU/CPU
  benchmark on the 2016-column slot grid selects identical parameters and
  runs ~2x faster at this data size (tiny matrices; transfers dominate).
- Inner-fold hyperparameter selection is now parallelized across parameter
  combinations (joblib threads; deterministic — same selected params as a
  sequential scan, verified by test). The slot logreg grid (12 combos x 4
  inner folds, saga on 2016 columns) was the wall-clock bottleneck (40+ min);
  parallel + GPU paths bring it into minutes.
- Configs: `e4_baselines.yaml` (v2) runs with `use_gpu: true`; the two cohort
  configs carry `use_gpu: false` so the recorded CPU numbers stay exactly
  reproducible.

### 3.19 Relative-phase cache fix (Phase 3 of v2 plan, 2026-08-03)
- **Bug fixed**: `process_recording` (component cache) re-resampled the
  *physical* `time_ms/signal_uv` trace for every segment, so relative-phase
  components (L_LATE, P_LATE) were cached on a physical-time grid instead of
  their canonical relative-phase grid — the phase alignment computed by
  `canonicalize_relative_phase` was silently destroyed. Absolute segments
  (L_EARLY_A, L_A_TO_B, P_EARLY, L_OP) are unchanged.
- **Fix**: for segments with `canonicalization_type == "relative_phase"` the
  cache now stores `seg.canonical_time/canonical_signal` directly (grid
  already on the relative-phase range, clipped to observed support);
  physical resampling remains the absolute-segment path.
- **Schema versioning**: caches are now versioned. Schema 1 keeps the legacy
  file names (`component_curves.zarr`, `signed_ot.zarr`, `components.parquet`)
  read only by the frozen snapshot; schema 2 writes
  `component_curves_v2.zarr`, `signed_ot_v2.zarr`, `components_v2.parquet` +
  `component_cache_manifest_v2.json` with `schema_version` recorded. New
  `cache_paths()` / `load_cache_manifest()` helpers; the v2 pipeline
  (baselines, fit-scalers CLI, QA report) validates the manifest and raises
  on missing/stale caches instead of silently reusing them.
- Cache rebuilt: 11097 recordings / 23069 components (unchanged counts;
  canonical grids differ for the 6663 relative-phase components).
- Tests: `tests/signal/test_relative_phase_cache.py` (7 tests) — L_LATE
  cached in phase units (not ms), cached signal != physical resample,
  absolute components keep the physical domain, alignment changes with the
  relative-phase range, schema rejection/acceptance, v2 vs v1 path names.
- Full suite green (104 tests) + ruff clean.
- **Corrected reruns** (cohort configs, CPU, `use_gpu: false`): primary
  (n=160, 21 methods) — clinical_demog_logreg 0.7206, clinical_logreg 0.6746,
  clinical_svm_rbf 0.6869, slot_logreg 0.6656, quality_histgb 0.6688,
  prevalence 0.4478. Secondary (n=232, 24 methods) — availability_histgb
  0.7823 (shortcut), clinical_demog_logreg 0.7567, slot_logreg 0.7265,
  clinical_logreg 0.7349, scdt_logreg 0.6031. Relative-phase fix moved
  biology-based numbers by <0.01; shortcuts unaffected.

### 3.20 Categorized QC flag semantics (Phase 4 of v2 plan, 2026-08-03)
- **Problem**: `_flagged` (any non-empty `component_qc_flags` = drop) removed
  5437 of 23069 components from clinical features — including **5290 of 5309
  L_LATE** rows whose only flag is `truncated-low` (relative-phase grid
  clipped to observed support). Late-waveform physiology was silently deleted.
- **Categories** (new `src/pathway_erg/models/qc_flags.py`, applied alike to
  clinical/curve/spectral/transport features):
  - `hard_invalid` (excluded everywhere): no-samples-in-window,
    late-support-too-short, fallback-window, late-landmark-invalid — geometry
    not landmark-driven.
  - `low_confidence` (kept): no-prominence-peak, supplied-only.
  - `truncated_or_limited_support` (kept): truncated-low, truncated-high,
    boundary-extreme.
  - `informational_qc` (kept, counted only in QC-rate features):
    disagrees-with-supplied.
- Unknown/new flags degrade to `informational_qc` (never dropped by default);
  test locks the full pipeline flag vocabulary against the category table.
- Wiring: `_clean_pairs` (LEOP+PERG clinical) uses `is_hard_invalid` instead
  of `_flagged`; `e4_curve_features` and `e4_derot_features` exclude
  hard-invalid rows (curve NaN-mask exclusion unchanged, added notes
  `n_components_excluded_qc` to derot). E0 availability/quality keep raw
  any-flag counts (they measure QC burden, not exclusion); slot `_flagged_rate`
  unchanged.
- In the current cache **no component is hard_invalid** — all 5437 previously
  dropped components are now kept (L_LATE 5309, L_A_TO_B 65, P_EARLY 34,
  L_EARLY_A 28, P_LATE 20 flagged rows).
- Tests: `tests/models/test_qc_flags.py` (9 tests) — vocabulary coverage,
  membership, unknown-flag safety, truncated-kept, real-cache check (no
  truncated L_LATE dropped), clean-pairs/curve/derot behavior incl. NaN-mask
  regression. Full suite green (113 tests) + ruff clean.
- **Reruns** (corrected semantics, CPU): primary (n=160) — clinical_demog
  logreg 0.6935, derot_rbf_svm_rbf 0.6892, derot_lr_logreg 0.6870,
  clinical_svm_rbf 0.6739, slot_logreg 0.6656, clinical_logreg 0.6591,
  clinical_histgb 0.6016, prevalence 0.4478. Secondary (n=232) —
  availability_histgb 0.7823 (shortcut dominates), clinical_demog_logreg
  0.7552, clinical_logreg 0.7547, clinical_demog_svm_rbf 0.7400,
  clinical_svm_rbf 0.7349, clinical_histgb 0.7119, slot_logreg 0.7265,
  scdt_logreg 0.6031. Shortcuts unchanged (deterministic lock); clinical
  numbers moved as the L_LATE rows now contribute.

### 3.21 Real multiscale spectral features (Phase 5 of v2 plan, 2026-08-03)
- **Problem**: the only frequency-domain features were `scattering` (wavelet
  scattering on *canonical* arrays) — the canonical curves are not uniform in
  time for relative-phase segments, so FFT-domain features on them are not
  interpretable; no physiological bands were defined.
- **New** `signal/spectral.py`: Hann-windowed periodogram on the *physical*
  component window (uniform sampling at `median_dt_ms`; LEOP 1953.125 Hz,
  PERG 1666.7 Hz). Per component: per-band log energy + relative energy,
  normalized spectral entropy, dominant frequency (search range
  0.5-250 Hz). All deterministic; NaN input -> all-NaN vector.
- **Bands** are explicit in the new `SpectralConfig` (PreprocessingConfig
  field, defaults): slow 0.5-20, mid 20-80, **op 80-300** (explicit OP band),
  fast 300-500 Hz (below both Nyquist rates). Feature names:
  `<band>_logenergy`, `<band>_rel_energy`, `spectral_entropy`,
  `dominant_freq_hz` (2*n_bands+2 features).
- **Cache schema 3**: `spectral_features_v3.zarr` (rows 1:1 with
  components.parquet), manifest records `spectral_feature_names`; `_v3`
  files replace the `_v2` set (same contents + spectral vectors); rebuilt
  (11097 recordings / 23069 components). v2 readers reject schema 2 as
  stale (tests updated to be version-agnostic).
- New `spectral` method (logreg only, like scattering) via
  `e4_spectral_features` (unit means, hard_invalid excluded, notes include
  `n_components_excluded_qc`); added to all three experiment configs.
- Tests: `tests/signal/test_spectral.py` (12 tests) — known-frequency
  behavior (150 Hz sine -> dominant freq ~150, OP band dominates; 5 Hz sine
  -> slow band dominates), entropy (sine low, noise high), relative-energy
  bounds, NaN, determinism, band/name table, explicit OP band default,
  process_recording emits vectors, cache alignment, builder hard_invalid
  exclusion. Full suite green (125 tests) + ruff clean.
- **Results** (CPU, cohorts as §3.20 plus spectral):
  primary spectral_logreg 0.6416 [0.5518, 0.7295] bal_acc 0.6521;
  secondary spectral_logreg 0.6606 [0.5912, 0.7263] bal_acc 0.5930.
  All other numbers bit-identical to §3.20 (deterministic lock).
  Spectral sits below scattering on both cohorts (scattering 0.6624 /
  0.6469); neither beats clinical/derot.

---

### 3.22 Confound audit (E0/E11 gates) — 2026-08-03

New module `src/pathway_erg/evaluation/confound_audit.py`; artifacts in
`artifacts/results/confounds/confound_audit.{json,md}` (primary cohort,
n=160, GPU logreg, locked folds / LOSO):

- **Sex confound:** sex-only AUROC 0.634 (primary) / 0.689 (legacy n=232);
  clinical+demog 0.693 overall but only 0.630 sex-adjusted (< clinical-only)
  → the demographics gain was sex exploitation. Report sex-adjusted AUROC as
  the primary LEOP metric. PERG clean (sex-only 0.515, balanced strata).
- **Availability shortcut:** 0.765–0.782 full cohort, still 0.676 inside the
  primary cohort; availability≈quality (OOF corr 0.91); stacking clinical on
  availability adds nothing (0.779≈0.782). Partly site-driven (site 1:
  ~23 rec/person, site 2: ~70; count→label sign flips between sites).
- **Slot-count ablation:** removing per-slot `_n` changes nothing
  (0.696→0.700); removing `_flagged_rate` too drops to 0.653/0.624
  sex-adjusted → flag rates carry ~0.04 of slot performance (the
  fallback/QC artifact channel; consistent with the 0.857 fallback-mask
  classifier). Decision: `_n` columns are not a leak (safe to drop for
  hygiene); `_flagged_rate` is a confound channel, pending fallback review.
- **LOSO (leave-one-site-out), sex-adjusted:** derot **0.631/0.624**,
  slot_no_counts 0.660/0.598, clinical 0.582/0.571. **derot is the only
  family robust to site shift** — clinical landmark features collapse toward
  chance. This is a direct robustness argument for the sOT representation.
- **First spectral result context (Phase 5, §3.21):** `spectral_logreg`
  0.642 (CI 0.552–0.730) on the primary cohort — below derot (0.687–0.689)
  and slot (0.666). Cause: `e4_spectral_features` unit-averages spectral
  vectors across all components, diluting band×component specificity
  (literature says band features are component-specific, e.g. PERG 1–5 Hz).
  Per-slot spectral is the natural fix (aligns with Phase 6's per-slot
  philosophy).

---

### 3.23 Phase 6 — per-slot signed SCDT/sOT (implementation) — 2026-08-03

Replaces the invalid unit-mean `scdt` baseline (quantiles of a curve averaged
across component types/domains; recorded AUROC 0.41–0.49, i.e. below chance).
New `src/pathway_erg/models/slot_sot_features.py` aggregates the per-component
signed derivative-OT descriptor **strictly within each fixed slot**, mirroring
`slot_features` (elementwise median/MAD, `_n`, `_flagged_rate`; NaN for a
missing slot; nothing averaged across components/eyes/intensities/protocols;
imputation only inside CV). Descriptor = 1D SCDT of each normalized sign
measure against the **declared uniform reference** on the probability grid,
+/− channels kept separate, amplitudes retained as log-masses (+ normalized
`mass_pos_frac`, total/net variation) so scaling survives normalization;
`infer_n_quantiles` reads the grid from the 135-dim v2 cache vector layout
(2·64 q + 7 tail scalars) automatically. Hard-invalid components are excluded
per Phase 4 (`n_components_excluded_qc` in notes).

Tests `tests/models/test_slot_sot_features.py` (8, all green), written before
code per the v2 rule: no cross-slot averaging; exact elementwise median/MAD;
missing slot → NaN; slot grid matches `e4_slot_features`; translation −5 ms →
quantiles shift by 5 ms with masses unchanged (verified end-to-end via
`signed_derivative_ot` with smoothing disabled — `SmoothingConfig(method="none")`);
amplitude ×3 → log-masses +log 3, quantiles unchanged; sign flip → +/− channel
swap; deterministic builder. Registered in `baselines.py` (METHOD_MODELS →
("logreg",)) and both v2 cohort configs (kept out of the main LEOP+PERG
`e4_baselines.yaml`, matching the `slot` convention).

Reconciliation 2026-08-04: the slot-OT feature was unified on the full-descriptor
builder above; the redundant mass-scalars-only variant (`e4_slot_ot_features`,
`SLOT_OT_MASS_COLUMNS`, `SignedOTResult.mass_vector`, `signed_ot_mass_v4.zarr`)
was removed. `SOT_TAIL_FEATURES` now includes `mass_pos_frac` matching
`to_vector`; cache schema 4 was rebuilt (v4 files were not on disk; rebuild
11 097 recordings / 23 069 components, no hard-invalid components). Full suite
137 tests green, ruff clean.

**Primary-cohort rerun (v2 signed-OT, §3.20 config + slot_sot):**
`slot_sot_logreg` **AUROC 0.6572** [0.564, 0.735] (bal_acc 0.6256), between
`slot_logreg` 0.6656 and `scattering_logreg` 0.6624 — above `spectral_logreg`
0.6416 and the chance-level `scdt` (0.4056–0.4891). All other primary numbers
bit-identical to §3.20 except the two derot methods, which gained the
`mass_pos_frac` scalar in the descriptor: `derot_lr_logreg` 0.6870 → 0.6821,
`derot_rbf_svm_rbf` 0.6892 → 0.6877.

**Secondary-cohort rerun (v2 signed-OT):** `slot_sot_logreg` **AUROC 0.7336**
[0.662, 0.802], the best per-slot/transport E4 feature on the secondary cohort
(slot_logreg 0.7265, spectral_logreg 0.6606, scattering_logreg 0.6469,
derot_lr 0.6470, scdt_logreg 0.6031; scdt_svm 0.4882 stays at chance). All
other secondary numbers bit-identical to §3.20 except the two derot methods
(descriptor gained `mass_pos_frac`): `derot_lr_logreg` 0.6470 and
`derot_rbf_svm_rbf` 0.6266 — both below slot_sot.

### 3.24 Phase 7 — nested-CV leakage removal (implementation) — 2026-08-05

Prior to Phase 7 the preprocessing transform (all-NaN/zero-variance pruning,
median impute + missing indicator, standard scale, PCA-variance selection) was
fitted once on the *whole outer training fold* and reused across the inner
CV, so inner-validation samples leaked into the features used for
hyperparameter selection; PCA dimensionality was picked by an outer-train
variance threshold; SVM Platt calibration was not nested.

Fix: every preprocessing + estimator step now lives in a single sklearn
``Pipeline`` (`build_pipeline`) that is refitted on each inner-training slice:
``col_drop -> imputer -> scaler -> [PCA] -> estimator``. Column pruning is a
fit-time transformer (`DropDegenerateColumns`) that uses only the slice it
receives; the PCA dimension for ``pca_fpca`` is now selected on the inner
folds (`_pipeline_param_grid`, dims 2..min(max_pca_components, n_features))
instead of an outer-train variance threshold; svm_rbf stays Platt-calibrated
inside the pipeline; no decision threshold is tuned anywhere (thresholds
locked). `_fit_transform_features` was removed; `select_and_fit`/
`_inner_fold_scores` take ``(kind, method, dataset, ...)`` and return/score
Pipelines.

Regression-first tests `tests/models/test_leakfree_inner_cv.py` (12 before
implementing; all were red): pipeline step layout, col_drop prunes by the
slice it is fit on, inner scoring fits the whole pipeline per fold, PCA dim
is grid-selected (respecting the feature-dimension cap), no threshold keys in
any parameter grid, parallel == sequential selection, determinism. Updated
`tests/models/test_baselines.py` to the pipeline API (col-drop/impute/PCA
behaviour now tested through fitted pipelines). Full suite 153 tests green,
ruff clean.

**Run-time progress log:** `status.log` in the results dir now logs
`FIT_START/FIT_DONE` per (fold, method, model) with a `fit=i/N` counter,
per-fit `elapsed_s`, and a moving `remaining_est_s` estimate, plus
`RUN_DONE` — so an estimate of when a cohort run finishes can be tailed
live (this also replaces the earlier silent-run problem).

**Cohort reruns (Phase 6 configs cloned to `*_p7` subdirs so the Phase-6
numbers stay archived):** both cohorts completed 2026-08-06;
`artifacts/results/baselines_v2_leop_primary_nine_step_p7_no_confound/` and
`baselines_v2_leop_secondary_all_protocols_p7_no_confound/`.

**Confound-column removal (Phase 7 part 2):** `_drop_confound_columns()` in
`baselines.py` strips per-slot `_n` (recording-count) and `_flagged_rate` (QC)
columns from the `slot`/`slot_sot` feature sets (and their `_demog` variants)
before training, because those channels are confounded with recording
site/protocol and carry no retinal biology. Result: on BOTH cohorts the
confound channels were worth only ~0.015 AUROC to `slot_sot` — `slot_sot_logreg`
primary 0.6572 → 0.6417, secondary 0.7336 → 0.7190 — while `slot_logreg`
*improved* (+0.019/+0.020) once the count columns were dropped. The per-slot
shape/quantile signal is real and does not live in the counting/QC columns.

**Primary-cohort Phase-7 results (2026-08-06; run completed in ~1h44m CPU):**
biggest movers vs Phase 6 (preprocessing re-fit per inner split, PCA dimension
inner-selected instead of an outer-train variance threshold): `pca_fpca_logreg`
0.6597 → **0.6143**, `slot_logreg` 0.6656 → **0.6848** (best E4 feature),
`clinical_demog_logreg` 0.6935 → **0.7023**, `clinical_demog_svm_rbf` 0.6288 →
0.6537, `metadata_svm_rbf` 0.5477 → 0.5641, `slot_sot_logreg` 0.6572 → 0.6417,
`clinical_logreg` 0.6591 → 0.6536, `scattering_logreg` 0.6624 → 0.6618,
`spectral_logreg` 0.6416 → 0.6424, `scdt_logreg` 0.4891 → 0.4869.
derot_lr/derot_rbf/raw_rbf/quality/prevalence/scdt_svm bit-identical. The flat
pca_fpca move is the expected correction from removing the outer-train
PCA-variance leak; the slot_logreg gain shows inner-split-pipeline selection
helps slot features.

**Secondary-cohort Phase-7 results (2026-08-06; GPU run completed in 23 min,
4h23m on CPU):** `slot_sot_logreg` 0.7336 → **0.7190** [0.6485, 0.7847],
`slot_logreg` 0.7265 → **0.7466** [0.6776, 0.8052] (best E4 feature),
`clinical_logreg` 0.7547 untchanged (0.7546), availability_histgb 0.7823
unchanged, derot_lr 0.647 unchanged. GPU (cuML) vs CPU (sklearn) agreed within
±0.001 for every non-slot method, so the solver change (cuML `qn` vs sklearn
`saga`) does not distort biology numbers.

**GPU enabling (Phase 7 part 3):** cuML 26.06.00 detected; RTX 3050 Laptop has
only 6 GB, so `Parallel(n_jobs=-1)` on the inner parameter grid OOM'd the card.
Fix in `baselines.py`: `n_jobs=1 if use_gpu else -1` — serialize GPU solvers.
Effect: secondary run 4h6m → 23 min (~10.5×); slot saga grid ~20 min → ~80 s.
`torch` is still the CPU build (`2.13.0+cpu`); nightly build needed before any
Phase-6 neural work can use the GPU. Run note: a duplicate second process was
started mid-run; it was killed — only a single survivor writes results.

### 3.25 Phase 8 — PERG logMAR acuity + sensitivity ablations (implementation) — 2026-08-08

PERG logMAR acuity is parsed directly from the raw participants CSV
(`parse_perg_acuity()` in `data/perg.py`: `va_re_logmar`, `va_le_logmar`,
`acuity_missing`, `acuity_n_eyes`, keyed by 4-digit `source_record_id`) — a
standalone parser, no schema/`VisitRecord` rebuild. The runner
(`evaluation/perg_sensitivity.py`, `PergSensitivityConfig`,
`run_perg_sensitivity`) reuses the leak-free `select_and_fit`/`_run_units`
path, five ablations over the same 336 PERG visits / 304 subjects, and a
`clinical_acuity` method variant (clinical features + the two eye logMar
columns) emitted as its own method name (never a silent mutation of
`clinical`). CLI: `run-perg-sensitivity`. Tests:
`tests/evaluation/test_perg_acuity.py` (4) + `test_perg_sensitivity.py` (4),
all green.

**Design notes fixed during the run (3 crash-debug cycles, 2026-08-07):**
- Sex strata: PERG `sex_standardized` is strings ("Female"/"Male"), not 0/1
  (LEOP is 0/1); strata now derived from the actual unique values.
- `one_visit_per_subject`: `visit_date` lives on `visits.parquet`, not the
  units table — merged via `global_visit_id` before first-visit dedup.
- Diagnosis families: PERG's target is abnormal-vs-normal, and every
  diagnosis family is 100% abnormal (macula 52, RP 47), so within-family
  AUROC is undefined. Per user decision, the ablation is **family vs healthy
  controls** (family patients ∪ the 106 Normal) with `min_family_n` applied
  to the *family* size (not the combined mask). Tiny families (<20) skipped.
- `_ClampedPCA` clamps `n_components` to `min(n_samples, n_features)` in both
  `fit` and `fit_transform` (Pipeline calls `fit_transform`), fixing
  `n_components=32 > 8 samples` crashes on the n=19 acuity-missing subset.
- GPU was unavailable the whole day: the NVIDIA driver died after a laptop
  suspend (`cudaErrorNoDevice`, no sudo to reload), so the full run was CPU.

**Results (CPU run, `artifacts/results/perg_sensitivity_v1/metrics.json`, AUROC):
- Baseline (n=336, pos=230): `clinical_logreg` **0.7537** [0.699, 0.807]
  (best), `clinical_acuity_logreg` 0.7380, `clinical_histgb` 0.7273,
  `pca_fpca_logreg` 0.7200, `derot_lr_logreg` 0.6904, `metadata_histgb`
  0.6650, prevalence 0.489 (chance). The logMar acuity columns add ≤0.002
  (clinical vs clinical_acuity): acuity is **not a confound** and not
  informative beyond the clinical features.
- one_visit_per_subject (n=304): `clinical_logreg` 0.7628 — no degradation vs
  baseline, so repeat visits (23 subjects) do not inflate the signal.
- Family vs healthy: macula (n=158) `clinical_logreg` 0.7442; RP (n=153)
  0.7692 (`clinical_svm_rbf` 0.7818, `pca_fpca_logreg` 0.7800) — both major
  families are detectable against healthy controls.
- Age strata: under-18 (n=64) hardest — `clinical_logreg` 0.6079 vs adult
  0.7332; under-18 metadata_logreg 0.6897 (age/sex metadata carry more for
  kids).
- Sex strata: female (n=176) `clinical_logreg` 0.7481 vs male (n=160) 0.6848.
- Acuity missingness: with-acuity (n=317) 0.7650; without-acuity (n=19)
  chance-level across the board (noise, as expected at n=19).

## 4. Key findings so far (numbers to remember)

### 4.1 Transport math behaves correctly (E1)
- Constant offset → quantiles/masses unchanged (plain SCDT drifts ~20.6 ms; sOT drifts 0 ms).
- Time shift → quantiles shift by exactly the shift (15.0 ms); masses unchanged.
- Amplitude ×3 → masses ×3, normalized quantiles unchanged.
- Noise: distortion grows with noise but stays a few % of span at low noise.
- Polarity flip → +/− channels swap exactly.
- Flat signals → both sign channels invalid (flagged, not fabricated).

### 4.2 Sharing theory is confirmed in simulation (E2)
- Zero mismatch → full sharing is best. High mismatch → separate is best.
- **Moderate mismatch → pathway partial sharing is best** — the central
  hypothesis of the paper is supported in the synthetic world.

### 4.3 Baseline performance (patient-level AUROC, outer folds)
| Endpoint | Best classical model | AUROC | Note |
|---|---|---|---|
| LEOP (232 people, 75 ASD) | slot_logreg (no confound cols, Phase 7) | 0.747 | best E4 biology feature; confound cols removed |
| LEOP | slot_sot_logreg (no confound cols, Phase 7) | 0.719 | per-slot shape signal, ~0.015 from confound cols only |
| LEOP (232 people, 75 ASD) | availability (gradient boosting) | 0.782 | missingness/QC are predictive — shortcut risk |
| LEOP | quality (gradient boosting) | 0.753 | QC features alone beat most biology |
| LEOP | clinical + demographics (logreg) | 0.757 | mostly sex-driven — see 4.5 |
| LEOP | clinical features (logreg) | 0.735 | |
| LEOP | scattering + demographics (logreg) | 0.735 | demographics add +0.06 |
| LEOP | prevalence (sanity floor) | 0.496 | |
| PERG (336 visits, 304 subjects, 230 abnormal) | FPCA + demographics (logreg) | 0.750 | demographics ~neutral |
| PERG | FPCA | 0.746 | |
| PERG | SCDT logreg | 0.724 | |
| PERG | raw RBF | 0.718 | |
| PERG | clinical features (histgb) | 0.716 | |
| PERG | prevalence (sanity floor) | 0.489 | |

**Interpretation:** simple models work; for LEOP the waveform-quality /
availability shortcut is dangerously close to (or above) signal-driven
performance — this must be addressed before neural models (plan E0 gate and
Section 29 confound strategy).

### 4.5 Demographics experiment (age + sex appended to every E4 method)
- **LEOP: demographics add +0.02 to +0.16 AUROC across methods**
  (clinical 0.735 → 0.757; scattering 0.672 → 0.735; scdt_svm 0.455 → 0.614;
  derot +0.06 to +0.07). **BUT the gain is mostly sex imbalance: controls are
  62 % male while the ASD group is only 24 % male** (76 % female ASD group —
  recruitment artifact, opposite of typical ASD epidemiology; ages identical,
  12.8 vs 12.9 y). A model can "succeed" by learning female → ASD. Must be
  handled by sex-stratified / sex-controlled analyses (plan E11) before any
  biological claim on LEOP.
- **PERG: demographics are ~neutral** (clinical 0.734 → 0.728, FPCA 0.746 →
  0.750). Disease group is older (38.3 vs 32.8 y); sexes balanced. Age signal
  is real but marginal for the disease mixture.
- **vs literature:** PERG FPCA+demographics 0.750 ≈ Koca 2026 (0.76 AUC,
  71.4 % acc, same PERG-IOBA data, patient-level CV) — we match the honest
  state of the art on this dataset without using visual acuity.

### 4.4 Confound / shortcut signals (must be resolved before Phase 6)
- **QC flag rate is imbalanced by class:** 4.9 % (class 0) vs 6.5 % (class 1).
- **OP missingness differs by class:** 82 % of control sessions have OP vs
  72 % of ASD sessions; missingness-only LEOP AUC = 0.607.
- **Fallback rate:** 195 / 23,069 = 0.85 %, but a fallback-only classifier
  reaches CV AUC 0.857 — fallbacks are predictable (mostly small mass /
  extreme slopes / duration), which needs the planned "no unexplained severe
  class imbalance" review (plan Section 8.6 gate).
- PERG prevalence: 230/336 = 68 % abnormal — class imbalance is structural.

---

## 5. Watch list — where errors are most likely to show up

1. **Leakage:** any code that fits scalers/QC/thresholds on test folds, or any
   split that puts a repeat-linked PERG subject in two folds. Guard rails:
   leakage assertions in `data/splits.py` + `assert_no_leakage` tests.
2. **Counts drift:** build must reproduce 253/5,309/4,434/304/336/677/1,354/11,097.
   First thing to check if the audit manifest changes: did raw files change?
3. **Determinism:** any rebuild with identical inputs must produce identical
   hashes (`data_hash` in `build_manifest.json`).
4. **Fallback classifier:** AUC 0.857 means landmark fallbacks are systematic —
   do not start neural training until this review is documented.
5. **Shortcut models:** LEOP availability/quality AUC ≈ 0.75–0.78 nearly match
   biological signal; the E0 decision rule (plan Section 17) applies.
6. **sOT edge cases:** near-flat traces (both masses zero), truncated late
   support, negative-time LEOP baseline — must be flagged, never fabricated.
7. **Not yet existing (do not expect to find):** `signal/vmd.py`,
   `models/raw_stem.py`, `ot_stem.py`, `pathway_router.py`, `aggregators.py`,
   `heads.py`, `path_erg.py`, `training/*`, `evaluation/bootstrap.py`,
   `calibration.py`, `data/datasets.py`, `collate.py`, `signal/augment.py`.

---

## 6. What changed recently and why (summary of decisions)

| Change | Why |
|---|---|
| New package `pathway_erg` instead of extending old PERG prototype | Plan Section 20.1 — old code has NA parsing, identity, session-averaging, and normalization errors; kept only as reference |
| PERG sessions never averaged before modeling | Averaging destroys session-level information and inflates sample counts |
| Repeat-link union-find → 304 subjects | Prevents the single worst leakage path |
| Explicit NA + no default diagnosis class | Unknown diagnoses must be ineligible, never silently a class |
| Derivative sOT instead of amplitude SCDT | Constant offsets (LEOP has pre-stimulus support, PERG does not) corrupt amplitude SCDT; derivative version is invariant |
| Fold-fitted QC + scalers | Global/threshold fitting on test data is leakage |
| 5/4 nested grouped folds, one locked split v1 | Every model must consume the same frozen splits |
| Raw vs smoothed copies kept separate | Differentiation amplifies noise; raw morphology must survive for the raw encoder |
| VMD deferred to a comparator (not implemented yet) | Plan Sections 1.4/15 — modes are unstable and are not named retinal generators |

---

## 7. What is next (immediate execution order, plan Section 35)

The v2 correctness plan runs in 9 phases (user-approved); every phase adds
regression tests *before* touching code, and nothing promises higher accuracy —
correctness first, then a fair old-vs-new comparison:

- **Phase 1 — LEOP cohorts:** `leop_primary_nine_step` (Control/ASD only,
  nine-step only, expected ~72 ASD / ~88 controls — log counts, never
  hard-code) and `leop_secondary_all_protocols` (all ~232 eligible, protocols
  explicit, never merged by averaging), plus protocol/availability/quality-only
  shortcut baselines; participant-level splits.
- **Phase 2 — slot features:** fixed slots component_type × eye × intensity ×
  protocol; median/variability within a slot; never averaging across
  components/eyes/intensities/protocols; imputation inside CV only.
- **Phase 3 — relative-phase cache: DONE (2026-08-03, §3.19):** relative-phase
  segments cached on their canonical grid; caches schema-versioned (v1 names
  stay frozen for the legacy snapshot; v2 pipeline reads `*_v2` files and
  rejects stale manifests); cache rebuilt; cohort numbers re-recorded.
- **Phase 4 — flag semantics: DONE (2026-08-03, §3.20):** four categories
  (hard_invalid / low_confidence / truncated_or_limited_support /
  informational_qc); only hard_invalid excluded, applied alike to
  clinical/curve/spectral/transport; unknown flags degrade to informational;
  the 5290 truncated L_LATE rows are now kept in features; cohorts rerun.
- **Phase 5 — real multiscale spectral features: DONE (2026-08-03, §3.21):**
  physical-domain periodogram (never on canonical arrays), explicit bands
  incl. the OP band (80-300 Hz), band/relative energy, spectral entropy,
  dominant frequency; schema-3 cache stores per-component vectors; known-
  frequency tests; cohorts rerun.
- **Phase 6 — valid signed SCDT/sOT: DONE (2026-08-04, §3.23):** per-slot only
  (never averaged across component types, eyes, intensities or protocols); +/−
  measures kept separate, masses kept as log-masses (+ `mass_pos_frac`,
  amplitude survives normalization), declared uniform reference on the
  probability grid; descriptor = 1D SCDT per normalized sign measure
  (2×64 quantiles + 7 tail features, schema-4 cache). New module
  `src/pathway_erg/models/slot_sot_features.py` (mirrors `slot_features`);
  regression tests `tests/models/test_slot_sot_features.py` (8 green: no
  cross-slot averaging, exact elementwise median/MAD, missing slot NaN, grid
  matches slot_features, translation→quantile shift, amplitude→log-mass only,
  sign flip→channel swap, determinism). Registered in `baselines.py`
  METHOD_MODELS→("logreg",) + dispatch and in both v2 cohort configs.
  Primary-cohort result: `slot_sot_logreg` 0.6572; secondary-cohort result:
  0.7336 (best per-slot/transport E4 feature; slot_logreg 0.6656/0.7265,
  spectral 0.6416/0.6606; scdt stays at chance).
- **Phase 7 — nested-CV leakage removal: DONE (2026-08-06, §3.24):** sklearn
  pipelines fitted inside every inner split; PCA dim inner-selected; per-slot
  `_n`/`_flagged_rate` confound columns dropped (`_drop_confound_columns`);
  GPU enabled (cuML, `n_jobs=1` to avoid 6 GB OOM). Primary `slot_logreg`
  0.685 / `slot_sot_logreg` 0.642; secondary `slot_logreg` 0.747 /
  `slot_sot_logreg` 0.719 (23 min GPU vs 4h23m CPU).
- **Phase 8 — PERG logMAR acuity: DONE (2026-08-08, §3.25):** parser
  (`parse_perg_acuity`, `data/perg.py`) + `perg_sensitivity.py` runner + CLI
  `run-perg-sensitivity` + 5 ablations (age strata, sex strata,
  one-visit-per-subject, diagnosis families, acuity missingness) + clinical
  acuity variant; `evaluation/` has 8 new tests (all green). Baseline
  `clinical_logreg` 0.754; acuity adds ≤0.002 (0.738 with acuity);
  one-visit-per-subject 0.763 (no repeat-visit inflation); family vs healthy:
  macula 0.744, RP 0.769; under-18 hardest (0.608); acuity-missing (n=19)
  chance-level. Runs took 3 crash-debug cycles (GPU driver died via suspend;
  sex labels are strings not 0/1; family ablation is family-vs-healthy since
  families are all-abnormal). `_ClampedPCA` now also clamps to n_samples.
- **Phase 9 — reporting/acceptance:** balanced acc, sens/spec/precision/F1,
  confusion matrix, clustered CIs, non-empty config/data/split/label hashes,
  label-permutation ≈ chance, acceptance checklist.
- Then: the corrected VMD baseline (`signal/vmd.py`, plan Section 15), Phase 6+
  neural work as previously planned.

---

## 8. How to verify everything (commands)

```bash
# Full test suite (must stay green)
.venv/bin/python -m pytest tests/ -q

# Legacy (frozen) baselines -> artifacts/results/baselines_legacy_v1/
.venv/bin/python -m pathway_erg.cli run-baselines \
  --experiment configs/experiments/e4_baselines_legacy.yaml

# Corrected v2 baselines -> artifacts/results/baselines_v2/
.venv/bin/python -m pathway_erg.cli run-baselines \
  --experiment configs/experiments/e4_baselines.yaml

# Immutable raw audit (checksums must be unchanged)
.venv/bin/python -m pathway_erg.cli audit --config configs/data/local.yaml

# Rebuild canonical data (must reproduce identical hashes/counts)
.venv/bin/python -m pathway_erg.cli build-data \
  --data configs/data/local.yaml \
  --preprocessing configs/preprocessing/reference.yaml

# Folds / QC / components / transport / simulations
.venv/bin/python -m pathway_erg.cli make-splits \
  --build-manifest artifacts/data/manifests/build_manifest.json --version v1
.venv/bin/python -m pathway_erg.cli run-qc --config configs/data/folds.yaml
.venv/bin/python -m pathway_erg.cli cache-components --fold all \
  --config configs/preprocessing/reference.yaml
.venv/bin/python -m pathway_erg.cli validate-transport --config configs/preprocessing/reference.yaml
.venv/bin/python -m pathway_erg.cli simulate-sharing
.venv/bin/python -m pathway_erg.cli run-baselines \
  --experiment configs/experiments/e4_baselines.yaml
```

**After any failure:** check (a) the failing manifest under `artifacts/*/manifest.json`,
(b) whether raw inputs changed (`raw_audit.json`), (c) whether the failing code
touches test-fold data, (d) the corresponding phase gate in the master plan.
If the failure is a test failure, run `pytest -x -v tests/` for the exact test.
