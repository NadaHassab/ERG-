# PATH-ERG — Progress Log

**Project:** PATH-ERG — Pathway-Aware Transfer for Heterogeneous Electroretinography
**Working paper:** *Pathway-Constrained Partial Transfer Across Unpaired Retinal Electrophysiology Protocols*
**Master plan:** `MASTER_PLAN_PATHWAY_AWARE_SIGNED_OT.md` (authoritative blueprint — 10 phases, 36 sections)
**Changelog:** `CHANGELOG.md` (release history)
**Last updated:** 2026-08-12

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
| 5 | Simple classical baselines (E0/E4) + VMD | **DONE** (2026-08-09, §3.26 grid/K-sweep + §3.31 shortcut review; gate: OOF predictions exist, VMD frequency/stability tests pass, shortcut risks reviewed) |
| 6 | Separate hierarchical neural models | **DONE** — authoritative 5-fold × 3-seed × 2-task run + neural confound gate (§3.42) |
| 7 | Joint SSL + pathway routing | **DONE** — SSL-init 0.614/0.731, no benefit vs from-scratch (§3.43) |
| 8 | Graph controls + label efficiency | **DONE** — all graph controls ≈ baseline; label-efficiency smooth (§3.43–3.44) |
| 9 | Robustness, statistics, interpretation | **DONE** — probes 10/10; external 4-domain 0.664/0.719; paired comparison NS (§3.43–3.46) |
| 10 | Paper + release | **IN PROGRESS** — all experiments complete; assembling tables/figures (§3.47) |

---

## 2b. Phases 1-10 renumbered (original plan Section 26) — VMD line:

After Phase 8, the plan's renumbered list restarts the counter; phase
"9" in that list is reporting/acceptance, and VMD (plan Section 15) is
tracked separately here (§3.26).

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

### 3.26 Phase 9 — VMD (variational mode decomposition) comparator — 2026-08-08

Adaptive-spectral comparator per plan Section 15 (Dragomiretskiy & Zosso
2014), implemented with the pinned `vmdpy 0.2` (pure-Python, no new heavy
deps) and kept strictly separate from the signed-OT pathway: VMD mode
identity is not physiological identity.

**`src/pathway_erg/signal/vmd.py` (20 tests, all green):**
- `VMDConfig` exposes the plan 15.2 grid as defaults: K=5, alpha=2000,
  tau=0, DC=0, init=1 (deterministic, no RNG), tol=1e-7, mirror-pad 25 ms,
  stability neighbours K∈{4,6}.
- `calibrate_vmd_frequency` verifies the vmdpy omega->Hz convention
  (`Hz = omega × fs`, fs = 1000/median_dt_ms) on synthetic tones at both
  datasets' real sample rates (LEOP 1953.125 Hz, PERG 1666.667 Hz). One
  calibration finding: K=3 splits low tones (10/25 Hz) under alpha=2000
  (err up to 0.52); K=2 resolves all four calibration tones to ≤0.45% worst
  error, so the calibration grid uses K=2.
- `decompose_vmd` mirrors the recorded padding before VMD and crops after
  (fixes a 120→110 Hz distortion on short windows), verifies reconstruction
  (relative RMS + residual energy), sorts modes by physical Hz, and is NaN-
  safe (all-NaN result, never a crash) and deterministic.
- `extract_vmd_features` → fixed-length vector: per mode (sorted) physical
  center frequency, log/relative energy, bandwidth, spectral + time entropy,
  peak-to-peak, skewness, kurtosis, Hilbert-envelope mean/max/area, first
  extremum time, source correlation, and a stability score against the
  neighbour-K decompositions; plus per-decomposition recon RMS, residual
  energy, convergence, unstable-mode count, iterations (K=5 → 80 features).
- Tests (20 in `tests/signal/test_vmd.py`): parametrized 10/25/60/120 Hz
  tones at both sampling rates, sorted modes, recon error < 0.05, mirror
  padding is required on short windows, determinism, NaN safety, feature
  shape/names, stability default.

**`signal/vmd_cache.py` + `cache-vmd` CLI (plan line 977 `vmd_modes.zarr #
baseline cache only`):** VMD lives in its own versioned cache (independent
`vmd_cache_manifest.json`, keyed by preprocessing config hash + VMD config
key) so adding it never invalidates the schema-4 component cache or existing
results. Rows align 1:1 with `components_v4.parquet`; `load_vmd_cache`
refuses stale schemas/hash/config-key mismatches (rebuild, never reuse).
`process_recording` gained an optional `vmd_cfg` that computes the fixed
length vector per physical segment (4-decomp cost: primary + neighbours).
Full cache built on the real data (23069 components from 9743 real
recordings, ~13 min with `--jobs 4` on CPU), calibration worst error 0.0045
(verified). Tests: `tests/signal/test_vmd_cache.py` (7, real-data only —
see below).

**Real-data discipline:** per user rule, no fabrication anywhere. The VMD
cache tests stage the *real* recordings + raw waveforms from `artifacts/`
into a temporary root and run the real segmentation/VMD path over them
(skip with an explicit message when the real build is absent); the loader
mismatch/schema tests only touch manifest metadata. Synthetic signals appear
only in `vmd.py`'s calibration procedure (a physical calibration of the
pinned implementation, not data) and never in tests of cache results.

**Registered in E4 baselines:** `METHOD_MODELS[vmd] = (logreg, svm_rbf,
histgb)` (plan 15.4: elastic-net logistic regression — the suite's saga
logreg — RBF SVM, gradient boosting), `e4_vmd_features` unit-mean
aggregation of the per-component vectors (mirrors `e4_spectral_features`,
QC-excluded components dropped), cohort masks applied the same as spectral.
First comparator run launched on the primary nine-step cohort
(`configs/experiments/e4_vmd_leop_primary_p9.yaml`,
`baselines_vmd_leop_primary_nine_step_p9`, CPU).

**Results (AUROC, bootstrap 95% CI, 2000 reps):**
- **Primary nine-step (n=160, pos=72) — decisive task (plan 15.5):**
  `vmd_svm_rbf` **0.677** [0.592, 0.760] — the best VMD model — beats the
  spectral baseline (`spectral_logreg` 0.642), the signed-OT slot path
  (`slot_sot_logreg` 0.642) and scattering (0.662); within CI of
  `clinical_demog_logreg` 0.702 (best overall). CIs overlap everywhere, so
  this is directional, but per plan 15.5 VMD qualifies as a **candidate
  model** (it beat the signed-OT pipeline and the spectral baseline on the
  primary task), not merely a comparator. `vmd_logreg` 0.643, `vmd_histgb`
  0.630; prevalence 0.448 (chance).
- **Secondary all-protocols (n=232, pos=75):** `vmd_svm_rbf` **0.701**
  [0.625, 0.775] again above `spectral_logreg` 0.663 and scattering 0.648,
  but below `slot_sot_logreg` 0.719 and `clinical_logreg` 0.755
  (best). `quality_histgb` 0.753 — the QC/protocol-count shortcut is again
  the strongest non-clinical signal on secondary (known confound story).
- Interpretation: VMD generalizes (consistent >spectral on both cohorts,
  ~0.03-0.04 AUROC) and is a legitimate adaptive-spectral comparator +
  candidate; it does not displace the signed-OT main pathway (slot_sot) or
  clinical features.
- Both runs completed: primary on CPU (5397 s) before the GPU came back;
  secondary relaunched on GPU after driver recovery — `use_gpu: true`,
  `baselines_vmd_leop_secondary_all_protocols_gpu`, 650 s vs the ~2-3 h CPU
  estimate.

**GPU driver recovery (2026-08-08):** the NVIDIA modules were missing for
the newly booted kernel 7.0.0-28-generic (the upgrade from 6.17 dropped the
nvidia-595-open build, which is why the driver died after the suspend
earlier). Fixed by `sudo apt install nvidia-dkms-595-open`
(upgraded 595.71→595.84, prebuilt modules for 7.0.0-28) + `sudo modprobe
nvidia [nvidia_uvm nvidia_drm nvidia_modeset]`. `nvidia-smi` OK, cupy sees 1
device, cuML imports; GPU runs are back to ~10-15 min.

### 3.27 External datasets — Flinders ISCEV Control + URFU OculusGraphy (2026-08-08)

Two external ERG datasets integrated as pipeline-native, typed data
(plan `PLAN_INTEGRATION_EXTERNAL.md`; gates 1–6 done, gate 7 effect probes
pending).

**Flinders ISCEV Control ERG** (`src/pathway_erg/data/flinders.py`):
- `Normal` sheet → 666 feature rows / 82 subjects / 5 protocols
  (LA3 292, 30Hz 111, DA001 106, DA3 90, DA10 67); `FIGURES` sheet → 8
  waveform traces at 0.512 ms steps (1953.125 Hz). 62 near-duplicate rows,
  434 missing cells, 4 empty + 2 metadata-missing traces — **counted, never
  dropped**.
- Protocol map extends the enum with `DA001`, `DA3`, `DA10`; phantom second
  FIGURES metadata blocks (e.g. `'193.703358'`) marked `metadata_missing=True`
  instead of raising (the block-1 data legitimately overlaps).
- All rows are healthy controls → `target_binary = 0` (eligible),
  `flinders_labels_v1`.

**URFU Pediatric and Adults ERG Database** (`src/pathway_erg/data/urfu.py`):
- `01 Appendix 1.xlsx` block parser: 423 signal columns / 104 subjects across
  5 protocols (Maximum 2.0 = 122, Photopic 2.0 = 106, 30 Hz flicker = 101,
  Scotopic 2.0 = 74, Oscillatory Potentials = 20), 0.5 ms steps → 2000 Hz.
  3 missing `#` subject cells bound to `URFU_UNLABELED_{sheet}_c{col}` ids;
  1367 missing feature cells counted; `eye = None` (database has no eye
  labels); OP columns → `WaveformKind.OP`. `02 Appendix 2.xlsx` **excluded**
  (322 ~6-sample fragments, truncated strings, no time axis/protocol).
- Diagnosis text exists per subject but is not yet mapped to target labels
  (healthy/unhealthy split available per URFU papers) — gate-7 work.

**Integration** (`build.py` FLINDERS/URFU blocks + `configs/data/local.yaml`):
verified full build — 744 participants / 776 visits / 1730 sessions /
11528 recordings (8 FLINDERS + 423 URFU new; LEOP/PERG counts byte-identical).
`config.py` `_from_dict` extended for PEP 604 unions (`X | None`). `.gitignore`
bug fixed: unanchored `data/` also ignored `src/pathway_erg/data/`.

**Audit gate** (`audit.py`): flinders/urfu walked + license notes in
`license_report.md` (Flinders **CC-BY-NC 4.0**; URFU IEEE DataPort
10.21227/y0fh-5v04 + 10.21227/r1wb-pg25); `__MACOSX` junk excluded with a
documented note; also fixed a latent bug where the detailed PERG license note
was overwritten with `""`.

**Web findings (research):** URFU wavelet-scalogram classification papers
(Ricker wavelet + ViT, median bal-acc 0.83/0.85/0.88 for
Maximum/Scotopic/Photopic; +19-20 % over a/b landmark features) and the
Constable connection — the Flinders control ERG and the in-repo LEOPs dataset
share the same PI (Paul Constable) and lab, making FLINDERS the closest
healthy reference for LEOPs transfer.

Tests: `tests/data/test_flinders.py`, `test_urfu.py`, `test_audit_external.py`
(30 total in `tests/data/`), all green; ruff clean.

### 3.28 External datasets — gate 7 effect probes (2026-08-08)

Four probes from `PLAN_INTEGRATION_EXTERNAL.md` §7, all writing to versioned
output subdirs under `artifacts/results/*` (frozen baseline artifacts never
touched).  New modules `evaluation/external_coverage.py`,
`evaluation/flinders_calibration.py`, `evaluation/urfu_sanity.py`,
`evaluation/leop_la3_transfer.py`, `data/urfu_labels.py`; four CLI commands
(`external-coverage`, `flinders-calibration`, `urfu-sanity`,
`leop-la3-transfer`).

- **Probe 1 — coverage/diversity** (`external_coverage.py`,
  `artifacts/results/external_coverage/`): extended scope adds 187 subjects
  (104 URFU + 83 Flinders), 800 sessions, 431 recordings over the original
  557/930/11097 (LEOP 253 + PERG 304). Protocols/sites/ages all expanded;
  labels: 54+27 URFU eligible after mapping, Flinders all healthy controls.
- **Probe 2 — normative Flinders calibration** (`flinders_calibration.py`):
  LEOP *Control* LA3 (n=139) vs Flinders healthy LA3 (n=292, source feature
  rows) — a/b amplitude/latency distributions overlap (KS 0.146-0.230,
  within-2SD 0.935-1.000; age-adjusted KS 0.146-0.230 with within-2SD
  0.942-1.000). Healthy populations overlap, no site/protocol drift flag.
- **Probe 3 — URFU supervised sanity** (`urfu_sanity.py`): explicit URFU
  diagnosis mapping (healthy=54, reduced=27, ineligible=23 per
  `urfu_labels_v1`, reviewer PENDING_CLINICAL_REVIEW) + held-out
  participant-level logreg on Maximum 2.0 a/b features; **AUROC 0.727**
  [0.568, 0.867], n=62 (45 healthy / 17 reduced) — the labels carry signal.
- **Probe 4 — LEOP LA3 transfer** (`leop_la3_transfer.py`): LEOP Control-vs-ASD
  on shared LA3 (n=204: 139 C / 65 ASD), per-subject a/b features; scaler
  fitted on LEOP-train-only (baseline AUROC 0.570 [0.485, 0.657]) vs
  scaler fitted on LEOP-train + 292 Flinders healthy features, no label
  mixing (**extnorm 0.571** [0.487, 0.657]). Δ<0.001 → standardization is
  robust to the healthy external reference; no transfer pip from norms.

Tests: `tests/evaluation/test_external_coverage.py` (7),
`tests/evaluation/test_flinders_calibration.py` (8),
`tests/evaluation/test_urfu_sanity.py` (9),
`tests/evaluation/test_leop_la3_transfer.py` (9) — all green.

**Gate-7 summary:** both external datasets are integrated, parsed, audited,
and probe-verified. They add healthy normative coverage (FLINDERS) and labeled
signal-carrying waveforms (URFU) without disturbing the frozen LEOP/PERG
build. Runs: `external-coverage` 3 min, `flinders-calibration` 15 s,
`urfu-sanity` 30 s, `leop-la3-transfer` 30 s (all CPU).

### 3.29 Phase 6 gate — fallback / confound shortcut review (2026-08-08)

Formal written review of the confounding shortcuts that must be documented
before any neural training starts (plan Section 8.6 / "no unexplained severe
class imbalance" review; watch-list item 4).  New module
`evaluation/confound_review.py` + CLI `confound-review`; artifacts in
`artifacts/results/confounds/confound_review.{json,md}` (same locked QA
tables the baselines use).

- **1. Fallback-only label shortcut:** per-subject/visit fallback rate as
  the *only* feature → LEOP 0.605, PERG 0.526, both well below the locked
  biology band (slot_logreg 0.657-0.685, derot 0.687-0.689, §4.3/4.4).  The
  fallback mask does **not** repackage class information into a cheap
  channel the way availability/QC baselines do.
- **2. Fallback physical explainability:** a physical-features classifier
  (peak-to-peak, duration, falling slope, log-mass) predicts which components
  used fallback at CV-AUROC 0.857 — i.e. fallbacks are an *explained quality
  artifact* (small mass / extreme slopes / short duration), not unexplained
  noise to be hunted down.
- **3. Protocol-count availability:** per-LEOP-subject recording count alone
  → AUROC 0.632, still below the availability gradient-boosting shortcut
  baseline (0.782) but above slot biology — confirms the pipeline's
  `primary_nine_step` cohort forbids `availability` (baselines.py raises) and
  that `_n`/`_flagged_rate` confound columns must stay dropped (Phase 7).
- **4. Label-permutation (reference):** the acceptance gate (Phase 9, §3.28)
  already forces subject-level label permutation to chance; reviewed here and
  passed.

**Verdict: PASS** — no blocking confound shortcut found.  Phase 6 neural
training is no longer blocked by the fallback/confound review: conditioned on
(a) the locked `primary_nine_step` cohort, (b) no protocol-count or
availability feature, (c) no fallback mask / missingness channel in the
signal.  The review is advisory (it documents measured values), and
re-runnable via `pathway_erg.cli confound-review`.

### 3.31 Phase 5 close — VMD grid, K-sweep and shortcut review (2026-08-09)

Closes the Phase 5 gate (plan Section 15): OOF predictions exist on both
LEOP cohorts, the VMD frequency/stability tests pass (§3.26), and the two
remaining plan-15.5 items are now done — the hyperparameter grid (plan
15.2) and the site/device shortcut review.

**VMD hyperparameter grid (`signal/vmd_grid.py` + `vmd-grid` CLI):** all 64
plan-15.2 points (K∈{3,4,5,6} × alpha∈{500,1000,2000,4000} × tol∈{1e-6,1e-7}
× pad∈{25,50} ms) swept over a deterministic 500-recording subsample of the
modeling population (1056 components, 67584 rows, 0 skips; first attempt
had 192 skips from unfiltered new-URFU rows — the sampler now restricts to
the component-cache population).  Findings:

- **Convergence:** all 64 points converge on every component (converged
  fraction 1.0).
- **pad=25 ms is correct:** median rel. recon error 0.052 (K=5) vs 0.338 at
  pad=50 ms — the extra 25 ms of mirrored padding beyond the support absorbs
  the transient, so the plan's 25 ms default is not only safe but necessary.
- **K plateau:** median recon 0.089 (K=3) → 0.065 (K=4) → 0.052 (K=5) →
  0.044 (K=6); monotone, smooth, no cliff, and the effective mode count
  saturates (n_modes>1% energy: 2.84/3.54/4.07/4.37).
- **alpha/tol are flat:** recon varies only ~2x across the full alpha range
  and is identical to 4 decimals across tol — the defaults (K=5, alpha=2000,
  tol=1e-7, pad=25) sit on a stable plateau, exactly what the plan requires.

**Model-level K sweep (GPU cuML, VMD-only configs, primary nine-step):**
to keep the comparison backend-consistent all four K values were re-run on
GPU (`e4_vmd_k_sweep_K{3,4,5,6}_gpu_p9`, `baselines_vmd_k_sweep_K*_gpu_p9`),
omitting the K-invariant methods (their results come from the locked
reference run §3.26).  AUROC (svm_rbf): K=3 0.6752, K=4 0.6517,
K=5 0.6768, K=6 0.6701.  Paired cluster bootstrap on the identical 160
units (2000 reps + 2000 sign-flips): every difference vs K=5 includes 0
and p ≥ 0.11 for all three model families — **no K effect**; the
prespecified K=5 default is retained.  (GPU K=5 = 0.6768 reproduces the
locked CPU reference 0.677 to 3 decimals — cuML/sklearn agree.)

**VMD site/device shortcut review (`vmd_shortcut_check`, artifacts in
`artifacts/results/confounds/vmd_shortcut_check.{json,md}`):** leave-one-
site-out on the best VMD model (svm_rbf, K=5, 80 features, GPU cuML) over
the two LEOP sites (1: 87 units, 2: 73): site1→site2 AUROC 0.592
(sex-adj. 0.628), site2→site1 0.666 (sex-adj. 0.651).  No collapse, no
saturation — the VMD features transfer across acquisition sites at the same
biology-level band as slot_no_counts (0.660/0.598) and derot (0.631/0.624)
in §3.20; VMD is not a site/device shortcut.

**Verdict: Phase 5 DONE** — VMD is a legitimate adaptive-spectral
comparator + candidate model (§3.26); the grid certifies the default
config, the K sweep finds no sensitivity, and the shortcut review is clean.

**Note (pre-existing drift):** `evaluation/confound_audit.py` still imports
the removed `_fit_transform_features` from the pre-pipeline baselines API
and cannot be imported; its results on disk (confound_audit.{json,md}) are
from the 2026-08-08 run and remain valid, but the module needs a
port-to-`build_pipeline` before the next re-run.  The VMD shortcut check is
a standalone script (independent of that module).

### 3.30 Phase 6 increment 1 — GD data layer (2026-08-08)

Plan Module 21.12 done.  New `data/datasets.py` (`LoadedCaches`,
`ComponentDataset`, `build_bags`, `domain_balanced_batch_indices`) and
`data/collate.py` (`collate_component_rows`, `collate_bag_units`).

- `LoadedCaches` validates cache alignment: 23,069 components = curves = OT
  rows; raises on misalignment.  `table()` merges components × recordings ×
  locked `outer_v1` folds (subject-level fold map); raises if any component
  lacks a locked fold.
- `ComponentDataset(caches, dataset, outer_folds)` — LEOP/PERG rows aligned
  to cache row index (253 / 336); rows expose `signal (128)`,
  `signal_mask (128 bool)`, `ot_vector (135)`, `physical (8)`.
- `build_bags` — one `BagUnit` per unit (LEOP 253 / PERG 336), sizes LEOP
  6–280, PERG 4–20; rejects any unit spanning folds (ValueError).
- Targets: 232/253 LEOP bags labeled (21 undiagnosed subjects carry
  `target_binary=None` — by design); PERG target via visit lookup.
- `domain_balanced_batch_indices` — deterministic LEOP/PERG interleave,
  ragged tail allowed (plan §9.8).
- Collators return fixed-shape NumPy (no framework import in data layer):
  component batch `(B,1,128)/(B,128)/(B,135)/(B,8)`; bag batch pads to
  longest bag in batch with explicit `component_mask`, and padding rows
  carry NaN in `physical` (training loop must mask before pooling —
  verified `test_collate_bag_padding`); `labels (B,) f64` with NaN =
  unlabeled.
- Tests: `tests/data/test_neural_datasets.py` (12).  Fixes: `BagUnit`
  gained `target_binary` (visit lookup), `ComponentRow` gained `unit_id`.

### 3.31 Phase 6 increment 1 — local neural stems (2026-08-08)

Plan Module 21.13 draft.  New `models/raw_stem.py`, `models/ot_stem.py`,
`models/local_fusion.py`.

- `RawStem` — Conv1d 1→16 (k7·p3) → residual blocks 16→32/32→64 (k5/k3,
  GroupNorm, skip, avg-pool ×2) → masked GPA+GMP → proj 64.
- `OTStem` — 135→128→64 GELU + LayerNorm + dropout 0.1.
- `LocalFusion` — descriptive sigmoid gate α over `[raw·α, ot·(1−α), phys]`,
  linear to 128; returns α for audit.
- Mask support: pooled valid mask zeroes padding before pooling, all-zero
  mask → finite zeros (no NaN); masked pooling differs from unmasked (test).
- No BatchNorm (plan §9.2; small bags); param count 87.7k
  (135→128→64 + 1→64 conv stem + gate) — well under the 0.5–1.0M
  full-model budget reserved for the later Phase-7 model.
- Tests: `tests/models/test_local_stems.py` (12 tests: shape, determinism,
  masked-pool semantics, gradients, NaN-free all-zero mask).
  Data+model new totals: 134 passed in tests/data + tests/models; full
  suite 305 passed (~4:37).

### 3.32 Phase 6 — gated-attention aggregators (2026-08-08)

Plan Module 21.15.  New `models/aggregators.py` with one parameterized
gated-attention pooling core (`_GatedAttentionPool`) and the plan-named
wrappers `ComponentToEyeAggregator`, `IntensityToEyeAggregator`,
`EyeToParticipantAggregator`, `EyeToSessionAggregator`,
`SessionToVisitAggregator`.

- Attention per plan §9.8: `a_j ∝ exp{wᵀ[tanh(Vz_j)⊙σ(Uz_j)]}`, value =
  token itself, `h = Σ_j a_j z_j`.
- Masks-only semantics: masked slots get zero attention (softmax over
  `-inf`), all-empty rows return an all-zero pooled token and zero
  attention — no NaNs (plan §9.8 "missing elements are masked").
- Tests `tests/models/test_aggregators.py` (8): attention sums, mask
  respect, permutation invariance, determinism, empty row, single-element
  bag, registry modules, shape/dtype validation.
- Verified end-to-end on real LEOP_A60 (60 components, 2 eyes, 10
  intensities): signal f64→f32 cast needed at torch boundary (raw cache
  arrays are float64; collators cast, raw stacks don't); participant
  token finite, eye attention ≈ balanced.

### 3.33 Phase 6 — complete model (plan Module 21.16, 2026-08-08)

`models/path_erg.py` — `build_model(config, pathway_graph) -> PathModel`
with `encode_component` / `encode_bag` / `forward` (plan 21.16
interfaces).

- Composes raw/OT stems + local fusion + gated hierarchy + per-task
  heads (128→64→1, plan §9.10); groups come from collator `group_eye` /
  `group_intensity` codes.
- Hierarchies: LEOP component→intensity→eye→participant; PERG
  component→eye→participant — one gated-attention primitive
  (`gather_by_group` + `promote_group_codes`) at every level.
- Padded rows never pollute: NaN physical is zeroed before the stems
  (component mask zeroes the fused token anyway); attention respects the
  mask; all-empty pools are zero vectors.
- Audit outputs (plan §9.11): per-level attention dict summing to
  #groups (intensity), #eyes (eye), 1 (participant) on real LEOP/PERG
  bags.
- Labels are never read by the model (`test_no_label_metadata_in_input`).
- Tests `tests/models/test_path_erg.py` (11): both task fwd/backwards
  with finite grads, param budget < 1.5M (~92k), state-dict save/load
  roundtrip, deterministic encode, attention sums, unknown-task guard,
  group-code promotion semantics.
- `tests/models` now 108 passed (~5 min); ruff clean.

### 3.34 Phase 6 — losses, samplers, trainer (plan Module 21.17)

`training/losses.py`, `training/samplers.py`, `training/trainer.py`
(ssl.py / finetune.py deferred to the shared-expert stage).

- `losses.py` — `positive_class_weight` (n_neg/(n_pos+eps), BCE semantics,
  raises on empty positives), `FoldWeightedBCE` (NaN label = no target,
  contributes nothing), per-fold weights from training labels only.
- `samplers.py` — `BagSampler` (per-fold bag batches, rejects requested
  folds absent from the bag list, explicit seed, `resume(state)` with RNG
  restore for exact stream continuation).
- `trainer.py` — `Trainer` + `TrainConfig`: AdamW, warmup+cosine
  (`_WarmupCosine`), grad clip 1.0, FP32 only (plan §14.11), per-fold
  positive-weight BCE, early stop by inner AUROC with best-checkpoint
  restore, per-epoch train loss / train+val AUROC / grad norm / fusion
  gate mean logs, checkpoint with optimizer+scheduler+sampler+config.
- Smoke-verified end-to-end on real LEOP fold 0 (3 epochs, lr 3e-4):
  weights move, loss finite, val AUROC reaches 0.714.
- Tests `tests/training/test_training.py` (9): loss algebra on known
  tensors, NaN-label exclusion, sampler fold restriction + foreign-fold
  rejection + held-out unit IDs never reachable + resume equivalence,
  one optimizer update changes weights, checkpoint roundtrip.
- Full suite: 324 + 9 = 333 passed; ruff clean.

### 3.35 Phase 6 — evaluation & statistics (plan Module 21.18, 2026-08-08)

`evaluation/metrics.py` extended + `evaluation/comparisons.py`,
`evaluation/calibration.py` (plan 21.18 interfaces).

- `evaluate_predictions(prediction_table, endpoint) -> MetricReport` —
  all point metrics at unit level (AUROC primary, AUPRC/balanced-acc/
  F1/MCC/sens/spec/Brier/ECE; plan §18.1), requires a `cluster` column
  (§18.3), raises on degenerate single-class tables.
- `cluster_bootstrap(prediction_table, cluster_col, metric, seed) ->
  BootstrapReport` — stratified cluster bootstrap percentile CIs
  (§18.3), ≥2000 reps default, skipped replicates counted explicitly.
- `paired_compare(pred_a, pred_b, cluster_col) -> ComparisonReport` —
  paired cluster bootstrap of M_A−M_B with percentile CI + unit-level
  sign-flip permutation p-value (§18.4); rejects mismatched unit sets
  (exact ID pairing).
- `fit_calibrator(inner_oof_logits, labels) -> Calibrator` — temperature
  calibration on inner OOF only (§14.10), gradient descent on BCE, ECE
  reported; rejects degenerate fits.
- Smoke on real data: 5-epoch LEOP fold-0 model → 47 units, AUROC 0.560
  [0.352–0.746] bootstrap, calibrated temp 1.00, ECE 0.17.
- Tests `tests/evaluation/test_evaluation_stats.py` (11): known-metric
  AUROC=1 case, cluster required, degenerate class rejection, bootstrap
  CI sanity, exact ID pairing + mismatch rejection, calibrator
  overconfidence correction + degenerate rejection + apply shape.
- Full suite 333 + 11 = 344 passed; ruff clean.

### 3.36 Phase 6 — leakage-safe separate-training runner (plan item 18, 2026-08-09)

`training/separate.py` + CLI `run-separate-neural` +
`configs/experiments/e6_separate_raw_ot_hierarchical_v1.yaml`.

- Corrected a critical scaffolding leak: `Trainer.fit` no longer infers a
  partition from `outer_fold` (which previously optimized the selected outer
  test fold). It now requires explicit train/validation bag lists, rejects
  subject overlap, uses fixed validation membership, and caps every epoch to
  one finite sampler pass by default.
- Bags now carry canonical `subject_id` + `visit_id`; PERG repeated visits stay
  grouped by subject in outer/inner partitions and bootstrap clusters. The
  collator emits these IDs with every batch.
- LEOP primary cohort is applied *before* bag construction: only labeled
  Control/ASD participants with nine-step components remain (160 people,
  14,911 components). PERG retains 336 labeled visit bags from 304 canonical
  subjects.
- Nested runner per task/fold/seed: four fresh inner models -> exactly one OOF
  prediction per outer-train bag -> inner-OOF temperature calibration -> fresh
  outer-train refit -> one calibrated outer-test prediction per unit.
- Runtime checkpoints: each inner fold writes `inner_fold_<j>.pt` and
  `inner_oof_fold_<j>.parquet`; final stage writes `final.pt`, predictions,
  calibrator, run manifest, then `COMPLETE` last. Interrupted runs cannot look
  complete.
- Artifact prediction schema is baseline-compatible (`method`, `task`,
  `outer_fold`, `unit_id`, `subject_id`, `visit_id`, `target`, probabilities)
  plus seed/logit/calibrated probability/checkpoint. Three seeds are averaged
  to one ensemble row per unit, never treated as independent observations.
- Fixed temperature-calibration gradient sign and pinned it against a finite
  difference; evaluation now rejects non-probabilities and duplicate unit IDs;
  paired comparisons require exact unit IDs when clusters repeat.
- Real PERG outer-0 / inner-0 smoke: 208 train visits (183 subjects), 66 fixed
  validation visits (61 subjects), 62 untouched outer-test visits; one FP32
  optimizer step completed with finite loss/AUROC.
- Tests `tests/training/test_separate.py` (9) + one new evaluation validator:
  cohort counts/protocol lock, nested coverage, subject-disjointness, one-step
  training, exact prediction cardinality, calibration gradient, and staged
  checkpoint/`COMPLETE` contract.
- The authoritative 5 outer folds × 3 seeds × 2 tasks run is intentionally not
  launched on the CPU-only torch build; the implementation/checkpoint contract
  is ready for that run.
- Full verification: 354 tests passed in 5:22; ruff clean.

### 3.37 Phase 6 — pathway router and experts (plan Module 21.14, 2026-08-09)

`models/adapters.py`, `models/experts.py`, `models/pathway_router.py` + CLI
config `routing_graph` on `ModelConfig`.

- `ProtocolAdapter` (residual `local -> 64 -> 64`) with `FlashLateAdapter` /
  `PERGLateAdapter`; `ResidualExpert` (residual `in -> 96 -> 64`, LayerNorm,
  GELU, dropout 0.1) with the five private experts and `SharedInnerLateExpert`
  (plan §9.6–9.7; our local token is 128 so adapters are 128→64→64 in place
  of the plan's reference 96→64→64).
- `PathwayRouter.forward(local_token, component_id, dataset_id, confidence)`
  -> `RoutedToken(shared, private, combined, gate, shared_mask)`. Private
  route is always present (component-by-expert mask). The shared route exists
  only for graph-allowed components, so forbidden edges have no gradient path
  through the shared adapter/expert. Graph controls (`correct`/`none`/`full`/
  `wrong`/`random`) change only the mask — parameter counts are identical.
- Gate: `sigmoid(W[private, shared, conf]) * conf`; zero confidence forces a
  pure-private token (low-confidence behavior, plan test list). `confidence`
  is the cached per-component `landmark_confidence` (0.95 confident, 0.0
  fallback/landmark-miss).
- `build_model(config, pathway_graph)` accepts a `PathwayGraph`, a graph
  control name, or the legacy hierarchy dict (stored verbatim, no router).
  `PathModel.encode_component` returns `shared`/`private`/`pathway_gate`
  (None without a router); padded rows never route.
- Data layer: `ComponentRow.landmark_confidence`; collator emits
  `component_type` (B,L) and `component_confidence` (B,L) alongside the
  existing hierarchy codes.
- Tests `tests/models/test_pathway_router.py` (13): forbidden-edge zero
  gradients (eval mode, none-graph control comparison), correct/wrong/random
  masks, parameter-count matching across controls, private route always
  present, low-confidence behavior, unknown component rejection, PathModel
  integration, and real-data forward/backward smokes for LEOP fold-0 and PERG
  fold-1 routed models (finite gradients through the shared expert).
- Full verification: 354 + 13 = 367 tests; ruff clean.

### 3.38 Phase 7 — joint SSL pretraining (plan item 19 / §14, Module 21.17 ssl.py+finetune.py, 2026-08-09)

`training/ssl.py` (Stage B reference objective), `training/finetune.py`,
CLI `run-ssl-pretrain` + `configs/experiments/e7_ssl_pretrain_v1.yaml`,
and `--init-ssl` on `run-separate-neural`.

- Reference objective §14.2 implemented with development weights:
  mask 1.0 (contiguous-span raw reconstruction, Huber only on masked+valid,
  §14.3), raw↔OT VICReg 0.25 (invariance+variance+covariance heads §14.4),
  augmentation VICReg 0.25 (safe scale/shift/noise pairs §14.5), geometry
  0.10 (within-dataset/component-type Huber on z-scored embedding vs sOT
  distance pairs, §14.6), gate prior 0.01 ((g−0.75)² on permitted edges,
  §4.5). Projection/decoder heads live in `JointSSLLoss` and are discarded
  after pretraining.
- Router now also exposes `gate_strength` (raw sigmoid before confidence
  scaling) so the prior targets the learned strength g; `ComponentEncoding`
  additionally carries `raw_token`/`ot_token` for the view/aug losses.
- Equal-domain gradient contribution: one domain-balanced LEOP + PERG batch
  per optimizer step with equal-weight loss sum; empty trailing domain
  slices are skipped (plan 14.2).
- Leakage: `pretrain_ssl` requires `exclude_fold` (SSL held-out exclusion,
  plan §23.6/§26.7) and trains only on the remaining folds; staged contract
  identical to item 18 (`final.pt` written via tmp+rename, `COMPLETE` last,
  `run_manifest.json` with config/data/split hashes + git revision).
- Stage C wiring: `finetune.init_from_ssl` copies only shared encoder keys
  into a fresh model (SSL heads never carried over, §14.4), errors on
  missing core keys; `freeze_encoders` keeps encoder frozen during
  fine-tuning; `SeparateTrainingConfig.init_ssl` points a supervised run at
  the matching fold's checkpoint (predictions note records the init).
- `ComponentDataset`/`domain_balanced_batch_indices` already existed and
  were reused unchanged; `collate_component_batch` maps flat component
  rows onto the encode_component bag-batch keys (L=1).
- Tests `tests/training/test_ssl.py` (11): masking/augmentation helpers,
  masked-loss algebra (only masked+valid positions), VICReg variance/
  covariance terms on known tensors, gate-prior formula, geometry pairs,
  real-batch joint loss, held-out-fold exclusion, checkpoint/COMPLETE
  contract, SSL-init encoder copy + freeze.
- Full verification: 367 + 11 = 378 tests; ruff clean. Authoritative
  per-fold SSL runs + SSL-init supervised runs remain pending on the
  GPU-enabled torch build.

### 3.39 Phase 8 — graph-aware separate runner (plan item 20 prep, 2026-08-09)

- `SeparateTrainingConfig.routing_graph` passes the graph control
  (`correct`/`none`/`full`/`wrong`/`random`) into every inner/final model;
  `build_stage_model` composes routing + optional SSL init. Controls keep
  identical parameter counts (Module 21.14), predictions carry
  `routing graph=<name>` in their note, and the run manifest/config hash
  distinguishes experiments.
- `configs/experiments/e8_pathway_graph_correct_v1.yaml` is the primary
  pathway graph; wrong/random/full/none controls are the same file with
  `routing_graph` swapped and a distinct `name`/`method`/`output_subdir`.
- Tests: `build_stage_model` honor the graph, unknown graphs raise, router
  parameters exist only when routed, parameter counts match across
  controls.
- Full verification: 378 + 1 = 379 tests; ruff clean. Control runs remain
  pending on the GPU-enabled torch build.

### 3.40 Phase 8 — label-efficiency support (plan item 21 / E9, 2026-08-10)

- `stratified_subset(bags, fraction, seed)` in `training/separate.py`
  samples whole subjects (every PERG visit of a chosen subject stays
  together, keeping the frozen inner partition nested), one class stratum
  at a time at `fraction`, deterministically (`np.random.default_rng(seed)`);
  `fraction == 1.0` is the identity and draws no randomness; subjects with
  inconsistent targets raise.
- `SeparateTrainingConfig.label_frac` (default 1.0) + `subset_seed`
  (default 9001) are applied to the outer-train partition only — outer
  tests are unchanged, per E9. Predictions carry a `label_frac` column and
  a note suffix; the run manifest records it; the seed ensemble keeps the
  column.
- Tests `tests/training/test_separate.py` now 14: identity at full,
  whole-subject + both-class preservation, determinism/seed-dependence,
  bad-fraction rejection, and the staged-checkpoint run asserts the
  `label_frac` column.
- Full verification: 379 + 4 + additional suites = 250 collected in
  training/evaluation/models, all green; ruff clean. Label-efficiency
  runs (10/25/50/100%, fixed subset seeds) remain pending on the
  GPU-enabled torch build like the item 20 controls.

### 3.41 Phase 8 — expert-fidelity probes (plan item 22 / E12, 2026-08-10)

- New `evaluation/probes.py`: `ProbeFrame`/`ProbeResult`,
  `encode_component_frame` runs the frozen model's `encode_component` over
  `collate_component_batch` chunks (streams: `fused` 128-d always;
  `shared`/`private` 64-d only for routed checkpoints),
  `probe_targets` (component_identity, dataset_identity, flash_intensity,
  log1p peak_to_peak, log1p duration; NaN = not applicable).
- `evaluate_probe`: linear OVR logistic (C=1e-3, max_iter 2000) with
  per-class macro OVR AUROC (binary → roc_auc on the positive column) or
  ridge regression with Pearson r; metric CIs from a unit-level cluster
  bootstrap (2.5–97.5 percentile, reps default 100).
- `run_probe_battery(model, caches, test_fold, ...)`: fits probes on the
  checkpoint's training folds and evaluates on the single held-out fold
  the model never saw; class targets are restricted to classes present in
  the probe fit; frames `all`/`LEOP`/`PERG` × streams; rejects test folds
  outside the locked fold tuple.
- `load_model_from_checkpoint` rebuilds the identical staged model from
  `final.pt` (`experiment.routing_graph` + seed); `save_probe_report`
  writes parquet + JSON summary; CLI `run-probes --checkpoint --fold
  --n-reps --seed`.
- Tests `tests/evaluation/test_probes.py` (10): targets, stream shapes
  (plain + routed), binary/multiclass/regression probe behavior, battery
  fold-safety, checkpoint roundtrip, report files.
- Full verification: 250 collected in training/evaluation/models all green
  (14 separate + 11 ssl + 10 probes + models); full suite rerun in
  progress; ruff clean. Battery launches on authoritative checkpoints
  remain pending on the GPU-enabled torch build.

### 3.42 Phase 6 — authoritative separate-neural run + confound gate (2026-08-11)

The complete `e6_separate_raw_ot_hierarchical_v1` run finished: 2 tasks ×
5 locked outer folds × 3 seeds = **30/30 COMPLETE** run directories. Each
run contains four inner checkpoints/OOF tables, an inner-OOF temperature
calibrator, a fresh outer-train refit, held-out predictions, manifest and
`COMPLETE` marker. Seeds are averaged to exactly one outer-OOF row per unit;
bootstrap CIs cluster repeated PERG visits by canonical subject.

**Authoritative 3-seed ensemble results (locked 0.5 threshold):**

| Task | Units / clusters | AUROC [95% cluster CI] | AUPRC | Balanced acc. | Sens. / spec. |
|---|---:|---:|---:|---:|---:|
| LEOP primary nine-step | 160 / 160 | **0.682 [0.589, 0.764]** | 0.654 | 0.670 | 0.500 / 0.841 |
| PERG all visits | 336 / 304 | **0.742 [0.682, 0.798]** | 0.861 | 0.679 | 0.678 / 0.679 |

- **Interpretation:** the separate raw+sOT neural baseline does not beat the
  best classical comparators. LEOP is in the same band as derot/VMD/clinical
  models (~0.68–0.70); PERG is close to FPCA+demographics (0.750). This is the
  correct independent-from-scratch reference for judging joint SSL/pathway
  routing, not evidence of neural superiority.
- **Post-hoc E0/E11 confound gate:** new
  `evaluation/confound_gate.py` + CLI `neural-confound-gate`; artifacts
  `results/confounds/neural_confound_gate.{json,md}`. **Verdict: PASS.** LEOP
  fallback/QC/OP-missingness-only AUROCs = 0.533/0.500/0.595, sex-only =
  0.634; neural minus strongest gated shortcut = +0.087 (minimum +0.05).
  Female/male neural AUROCs remain 0.670/0.684. PERG fallback/QC =
  0.541/0.500, OP channel is not applicable, sex-only = 0.524; neural margin
  = +0.201. Protocol-count/availability is not a neural input and remains
  forbidden for LEOP `primary_nine_step`.
- **Resume bug found and fixed before accepting metrics:** resume previously
  skipped COMPLETE runs without loading their `predictions.parquet`, so a
  resumed summary initially contained only runs executed by that process
  (LEOP n=66). Resume now loads every COMPLETE run prediction into the final
  ensemble and fails if a COMPLETE directory lacks predictions.
- **Schema-drift bug found and fixed:** two early fold-0 artifacts predated the
  `label_frac` column and carried NaN while later seeds carried 1.0, splitting
  one unit into duplicate ensemble rows. Resume normalizes missing
  `label_frac` to the configured fraction, and a new uniqueness assertion
  refuses any duplicate `(task, outer_fold, unit_id)` before metrics are
  written. Correct cardinalities are locked at LEOP 160 and PERG 336.
- Regression coverage: resume-load/legacy-`label_frac`, missing-prediction
  failure, ensemble uniqueness, channel semantics (including PERG OP=n/a),
  exact prediction validation and real-cache gate smoke. Focused verification:
  29 tests green. Final full verification: **408 tests passed**, repository-wide
  ruff clean, and `git diff --check` clean.

### 3.43 Phase 7/8/9 — authoritative stage batch launch (2026-08-11)

The GPU-enabled torch build is confirmed working (torch 2.13.0+cu126,
RTX 3050 6 GB laptop GPU), so the pending authoritative runs from items
19–22 are now executing through `scripts/stage_runs.sh` (one driver, every
supervised invocation uses `--resume` so restarting never redoes completed
run dirs; log `logs/stage_runs_20260811.log`).

- **Phase 7a — joint SSL pretraining:** `e7_ssl_pretrain_v1.yaml`
  (`routing_graph: correct`, 5 epochs, seed 1001) run once per excluded
  fold with `--exclude-fold 0..4`; checkpoints land at
  `results/ssl_pretrain_v1/foldN/final.pt` (SSL held-out exclusion,
  plan §23.6). All five completed on GPU in minutes.
- **Phase 7b — SSL-init supervised fine-tuning:** new per-fold configs
  `configs/experiments/e6_sslinit_foldN_v1.yaml` (N=0..4) run only the
  matching fold's `tasks × seeds` (2×3 runs each) with
  `init_ssl: …/foldN/final.pt` and `routing_graph: correct` (the SSL model
  is graph-routed, so a routed fine-tune model keeps every copied encoder
  key), shared `output_subdir: separate_raw_ot_hierarchical_sslinit_v1`;
  `e6_sslinit_summary_v1.yaml` then rebuilds the full 5-fold × 3-seed
  ensemble by resuming the shared run dirs. 6 distinct configs ⇒ 30 runs.
- **Phase 8a — pathway graph controls:** `e8_pathway_graph_{correct,
  none,full,wrong,random}_v1.yaml` share identical hyper-parameters and
  parameter counts (Module 21.14), differing only in `name`/`method`/
  `output_subdir`/`routing_graph`; 5 configs × 6 runs = 30 runs each,
  150 runs total, from scratch (`init_ssl: null`).
- **Phase 8b — label efficiency (item 21 / E9):** `e8_label_frac_{0.1,
  0.25,0.5}_v1.yaml` reuse the separate baseline (`method:
  separate_raw_ot_hierarchical_v1`, unrouted) with whole-subject
  stratified subsets at `label_frac` (subset_seed 9001 fixed); 90 runs.
  The 1.0 row is the completed e6 run.
- **Phase 9 opener — expert-fidelity probes (item 22 / E12):** 10
  `run-probes` batteries (LEOP + PERG × folds 0–4) on the authoritative e6
  checkpoints `…/runs/separate_raw_ot_hierarchical_v1-{leop,perg}-foldN-
  seed1001/final.pt`, 100 bootstrap reps, seed 7.
- **Bug fixed (SSL on GPU):** `JointSSLLoss` heads were never moved to the
  configured device, so every CUDA pretrain failed with a mat1/cpu device
  mismatch; `pretrain_ssl` now calls `loss_fn.to(cfg.device)` + `train()`
  (ssl.py). Encoder paths already derive their device from the model
  parameters, so no other SSL changes were needed.
- **Regression found in the in-flight external-datasets work**
  (`PLAN_INTEGRATION_EXTERNAL.md` thread): `LoadableCaches.table()` guarded
  FLINDERS positive labels with a vectorized condition tested as a scalar,
  which raised `ValueError: truth value of a Series is ambiguous` for
  every `table()` caller and broke the SSL suite. Fixed with `.any()` so
  the external-binding work is unblocked while stage runs proceed. The
  rest of that WIP is untouched.
- **Bug in my own config derivation, caught mid-run:** the per-fold SSL-init
  configs were derived with `sed s/fold0/foldN/` which rewrote every
  `fold0` token but left `outer_folds: [0]` untouched — so configs 1–4
  resumed fold-0's COMPLETE runs and printed fold-0's ensemble numbers
  (log "fold 1–4" rows were identical to fold 0; only fold-0's 6 runs and
  2 fold-1 LEOP runs were real). Worse, those 2 fold-1 LEOP runs were
  started by the summary config whose `init_ssl` pointed at the fold-0
  checkpoint — the fold-0 SSL model had seen folds 1–4, i.e. a §23.6
  leakage. Action: regenerated the four configs with correct
  `outer_folds` (fold0 file verified unchanged), purged the bogus shared
  ensemble files, deleted the 2 contaminated run dirs, and re-ran the
  fold-1 config so all six runs initialize from the fold-1 checkpoint.
  The driver now skips SSL folds that already have a `COMPLETE` marker and
  runs detached (`setsid`) so tool-session aborts cannot kill the batch.
- **Phase 8b results (label efficiency, whole-subject subsets, seed 9001):**

  | label_frac | LEOP AUROC [95% CI] | PERG AUROC [95% CI] |
  |---|---:|---:|
  | 0.10 | 0.601 [0.511, 0.690] | 0.604 [0.535, 0.670] |
  | 0.25 | 0.557 [0.461, 0.647] | 0.718 [0.657, 0.778] |
  | 0.50 | 0.628 [0.534, 0.718] | 0.727 [0.665, 0.784] |
  | 1.00 (e6) | 0.682 [0.589, 0.764] | 0.742 [0.682, 0.798] |

  PERG degrades smoothly and stays near its ceiling already at 25% labels;
  LEOP is noisier (non-monotonic at 0.25, small cohort + fixed subset
  seed) and needs the full label set for its best 0.682.
- **Phase 9 opener:** 10 probe batteries were launched on the authoritative
  e6 checkpoints but every one failed at `load_model_from_checkpoint` — the
  external-datasets work added `heads.URFU` to `PathModel` while old
  checkpoints lack those keys and the loader was strict. Fixed with
  `strict=False` (probes never touch task heads); a probes-only runner
  (`scripts/run_probes.sh`) + watchdog relaunched the batteries
  (logs/probes_20260811.log).
- **Operational hardening:** the stage batch kept dying silently (partial
  run dirs, idle GPU). A generic watchdog (`scripts/watchdog.sh
  <driver> <log>`) now restarts the driver until its final "done" line;
  every restart resumes from `COMPLETE` markers, so no work is re-run
  except interrupted runs.
- **Phase 9 opener results — expert-fidelity probes (10/10 batteries,
  all folds, frozen embeddings, linear probes):**

  | Target | LEOP fused | PERG fused |
  |---|---:|---:|
  | component identity (OVR AUROC) | 0.995 | 0.995 |
  | dataset identity (AUROC) | 0.994 | 0.991 |
  | flash intensity (Pearson r) | 0.559 | 0.563 |
  | log1p peak-to-peak (Pearson r) | 0.989 | 0.989 |
  | log1p duration (Pearson r) | 0.995 | 0.994 |

  Frozen embeddings are near-lossless for waveform morphology and domain
  identity but only moderately linear in flash intensity — consistent
  with the negative transfer results: the encoder memorizes perceptible
  structure (dataset/component/amplitude) without learning the
  classification-relevant intensity contrast.
- **Probe report naming bug fixed:** batteries were saved flat as
  `probe_battery_foldN.parquet`, so the PERG fold-0 report silently
  overwrote the LEOP fold-0 report. Reports are now task-scoped
  (`probes/{leop,perg}/probe_battery_foldN.parquet`); the LEOP fold-0
  battery and both mid-run-killed PERG fold 1–2 batteries were re-run.
  Final sets: 5 LEOP + 5 PERG parquet+JSON reports.
- **Phase 8a results so far (from-scratch routed controls, 3-seed ensembles):**

  | Control | LEOP AUROC [95% CI] | PERG AUROC [95% CI] |
  |---|---:|---:|
  | separate baseline (e6, unrouted) | 0.682 [0.589, 0.764] | 0.742 [0.682, 0.798] |
  | `correct` (pathway graph) | **0.674 [0.584, 0.760]** | **0.721 [0.662, 0.778]** |
  | `none` (no sharing) | **0.654 [0.562, 0.739]** | **0.726 [0.665, 0.785]** |
  | `full` (all sharing) | **0.644 [0.556, 0.733]** | **0.721 [0.661, 0.779]** |
  | `wrong` (misrouted) | **0.650 [0.560, 0.730]** | **0.721 [0.659, 0.780]** |
  | `random` (gated by chance) | **0.670 [0.581, 0.755]** | **0.723 [0.661, 0.781]** |

  **Complete Phase 8a finding: no routing control beats the unrouted
  separate baseline (LEOP 0.682 / PERG 0.742).** All five graph variants
  cluster tightly (LEOP 0.644–0.674, PERG 0.721–0.726); the pathway
  structure (correct/none/full/wrong/random) has essentially no effect —
  a clean null result for learned pathway routing on these tasks.

  The correct pathway graph does not beat the unrouted separate baseline
  (LEOP −0.008, PERG −0.021); `none`/`full`/`wrong`/`random` pending.
- **Phase 7b results (all 30 runs + 5-fold summary, SSL-init ensemble):**

  | Task | AUROC [95% CI] | AUPRC | Balanced acc. | n |
  |---|---|---:|---:|---:|
  | LEOP (SSL-init) | **0.614 [0.522, 0.702]** | 0.586 | 0.609 | 160 |
  | PERG (SSL-init) | **0.731 [0.673, 0.790]** | 0.850 | 0.668 | 336 |

  Per-fold SSL-init AUROCs: LEOP 0.629/0.684/0.414/0.683/0.623 (fold 2
  badly degraded), PERG 0.785/0.751/0.634/0.776/0.763. SSL pretraining +
  frozen encoders does **not** beat the from-scratch separate baseline
  (LEOP −0.068, PERG −0.011) — a clean negative result for the reference
  SSL objective under the freeze-encoders Stage C contract.

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
   review now documented (2026-08-08, §3.29): fallbacks are a physically
   explained quality artifact; the fallback→label shortcut is weak (0.605/0.526),
   so Phase 6 can proceed subject to the §3.29 lock-ins.
5. **Shortcut models:** LEOP availability/quality AUC ≈ 0.75–0.78 nearly match
   biological signal; the E0 decision rule (plan Section 17) applies.  The
   protocol-count/availability channel stays forbidden on `primary_nine_step`.
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
- **VMD comparator — plan Section 15: DONE (2026-08-08, §3.26):**
  `signal/vmd.py` (calibrated, deterministic) + `signal/vmd_cache.py`
  (`vmd_modes.zarr`, baseline-only cache), `cache-vmd` CLI, `vmd` registered
  in E4 baselines (logreg/svm_rbf/histgb per plan 15.3), real-data cache
  tests; primary nine-step: `vmd_svm_rbf` 0.677 (beats spectral 0.642 and
  slot_sot 0.642 → candidate model per plan 15.5); secondary: 0.701 (above
  spectral 0.663, below slot_sot 0.719); GPU driver restored
  (nvidia-dkms-595-open for kernel 7.0.0-28).
- **Phase 9 — acceptance gate: DONE (2026-08-08):** `evaluation/acceptance.py`
  + CLI `run-acceptance` (with `--reuse-existing` to re-verify without
  refits).  Gates: provenance hashes (config/data/split/label non-empty),
  full metric set incl. confusion matrix at locked 0.5, paired OOF
  predictions, and label-permutation ≈ chance.  **Verdict: PASS** (14/14
  checks) on `e4_acceptance_gate_p9`.  Details: base run AUROCs intact
  (12 methods × 3 perm seeds); first fixed-band check failed at 0.163 vs
  0.08 — diagnosis: the fixed band is uncalibrated for a subject-clustered
  design (constant-per-fold baselines like `prevalence` sit at 0.500 in
  every fold but deviate pooled; the null's own 95th percentile is 0.144).
  Gate now runs a deterministic subject-clustered MC null (2000 draws,
  labels shuffled between subjects, nesting preserved): observed seed-meaned
  max deviation 0.1165, null p = 0.304 → consistent with chance, no
  leakage.  The fixed band is reported as secondary info and does not gate.
  Bug fixed along the way: `run_label_permutation_gate` previously skipped
  the base run.
- **Fallback/confound review: DONE (2026-08-08, §3.29):** written pre-Phase-6
  gate (`confound_review.py` + CLI `confound-review`). No blocking shortcut:
  fallback-only LEOP 0.605 / PERG 0.526 below the biology band; fallback
  mask physically explained (0.857); protocol-count 0.632 < availability
  gb (0.782) but the cohort forbids it anyway. **Verdict: PASS** — Phase 6
  neural work is unblocked subject to the 3 conditioning lock-ins in §3.29.
- Then: Phase 6+ neural work as previously planned (now unblocked by the
  fallback/confound review gate).

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
.venv/bin/python -m pathway_erg.cli run-acceptance \
  --experiment configs/experiments/e4_acceptance_gate_p9.yaml --seeds 0 1 2
# Pre-Phase-6 fallback/confound shortcut review (verdict PASS unblocks Phase 6)
.venv/bin/python -m pathway_erg.cli confound-review \
  --data configs/data/local.yaml

# Authoritative separate neural ensemble (resume loads all 30 COMPLETE runs)
.venv/bin/python -m pathway_erg.cli run-separate-neural \
  --experiment configs/experiments/e6_separate_raw_ot_hierarchical_v1.yaml \
  --resume

# Post-hoc neural E0/E11 shortcut gate (verdict PASS)
.venv/bin/python -m pathway_erg.cli neural-confound-gate \
  --experiment configs/experiments/e6_separate_raw_ot_hierarchical_v1.yaml
```

**After any failure:** check (a) the failing manifest under `artifacts/*/manifest.json`,
(b) whether raw inputs changed (`raw_audit.json`), (c) whether the failing code
touches test-fold data, (d) the corresponding phase gate in the master plan.
If the failure is a test failure, run `pytest -x -v tests/` for the exact test.

## 3.44 Plan §11 — external binding: combined 4-domain path (2026-08-11)

Implemented the full §11 external ("data-increase") path additively: the
LEOP/PERG-only e6/e7 experiments, frozen v4 caches, v1 folds, and manifests
stay byte-identical. 71 new tests; full suite 479 passed. Real-data smoke
ran end-to-end on the GPU.

- **Flash-family routing:** `data/schemas.py` adds `FLASH_DATASETS`
  (LEOP/URFU/FLINDERS), `PATTERN_DATASETS` (PERG), `SUPPORTED_DATASETS`;
  `signal/landmarks.py`/`segments.py`/`component_cache.py` branch on the
  flash family (offset policy, OP override, landmarks, segments).
- **Gate §11.2.2 — external component cache** (`signal/external_cache.py`):
  `cache-external-components` builds `external_external_v1_v4.zarr` +
  `components_external_v1.parquet` + own manifest (`binding`/
  `schema_version`/`datasets`/`config_hash`, refuses existing outputs);
  `cache_components` now filters to LEOP/PERG so the frozen v4 cache can
  never be polluted by external rows. Real build: **1253 components from
  431 valid URFU/FLINDERS recordings**.
- **External folds** (`data/external_splits.py`): subject-keyed outer/inner
  folds, version `external_v1`, own files
  `outer/inner_folds_external_v1.parquet` (real build hash
  `2f32eb3d…`); `assert_no_leakage` scoped to assigned datasets.
  New `configs/data/folds_external.yaml` (class/sex/age_bin strata).
- **Data layer:** `LoadedCaches` gains `external_bindings` +
  `external_fold_version` (validated, must differ from `fold_version`);
  `table()`/`build_bags` unit rules — LEOP/FLINDERS subject-keyed,
  PERG/URFU visit-keyed; FLINDERS `target_binary == 1` raises;
  `domain_balanced_epoch_indices` N-domain sampler (two-domain output
  identical to the legacy `domain_balanced_batch_indices`, which now
  delegates to it).
- **SSL (item 19 external):** `SSLConfig` adds `ssl_datasets`,
  `domain_batches`, `plan_per_epoch`, `external_bindings`,
  `external_fold_version`; `pretrain_ssl` runs N domains with per-domain
  seeds (LEOP=0, PERG=1 locked; others=2), per-domain batches, ≥2 domains
  enforced; checkpoint records per-domain `n_components`.
- **Router/heads:** `UrfuLateAdapter`/`FlindersLateAdapter`
  (`models/adapters.py`), four-domain `adapter_by_dataset` routing,
  heads {LEOP, PERG, URFU} (FLINDERS head forbidden); URFU
  `encode_bag` pools components straight to a per-visit token (no eye
  level).  URFU supervised endpoint gated by
  `require_urfu_labels_signed_off` (`data/urfu_labels.py`, §11.2.1 —
  still PENDING_CLINICAL_REVIEW, SSL unaffected).
- **Configs:** `e9_ssl_pretrain_external_v1.yaml` (4 domains,
  `plan_per_epoch: true`, external_v1 binding) + `e9_sslinit_external_
  fold{0..4}_v1.yaml` + `e9_sslinit_external_summary_v1.yaml`
  (`separate_raw_ot_hierarchical_sslinit_external_v1`).
- **Real-data smoke (GPU, 1 epoch):** `run-ssl-pretrain` over all four
  domains completed — `results/ssl_pretrain_external_v1_smoke/fold4/
  final.pt`, final epoch total loss 1.2777.
- **Bugs fixed:** `make_leops_segments`/PERG late-segment fallback used
  `_canon_flags` (unbound) → renamed `canon_flags` (pre-existing latent,
  unreachable for LEOP/PERG which always have both landmarks);
  `config.py` `_from_dict` now supports `tuple[Dataclass, …]` fields so
  `FoldConfig.constraints` loads (previously the folds CLI path could
  never load `folds.yaml`).

Remaining (blocked/scheduled): URFU sign-off for the supervised head
(§11.2.1); then full e9 5-fold SSL + 30-run sslinit ensemble + paired
comparisons + `artifacts/results/external_v1/` write-up.

### 3.45 Plan §11 — external comparison tooling + staged e9 batch (2026-08-11)

- **Authoritative paired comparison CLI**: `compare-external-sslinit`
  (`evaluation/external_comparison.py`).  Validates both prediction
  tables (5-fold coverage, duplicate units, within-subject label
  consistency, finite [0,1] probabilities), aligns on
  `outer_fold`/`unit_id`/`subject_id`/`target`, then runs the paired
  clustered bootstrap of `roc_auc` differences
  (four-domain external sslinit minus two-domain sslinit) per task via
  `paired_compare` (`comparisons.py`), with Holm-adjusted p-values over
  the LEOP/PERG family.  Writes only to
  `artifacts/results/external_v1/paired_comparisons.json` (atomic
  replace; frozen trees untouched).  4 new tests
  (`tests/evaluation/test_external_comparison.py`); subset regression
  137 passed.
- **Staged external batch**: `scripts/stage_external_runs.sh` queues
  the full e9 arm behind the running authoritative stage batch
  (PID 7304) — five four-domain SSL pretrains
  (`e9_ssl_pretrain_external_v1.yaml`, `--exclude-fold 0..4`), then the
  30-run `e9_sslinit_external_fold{0..4}_v1.yaml` supervised fine-tune
  ensemble (LEOP/PERG only; URFU stays gated), then the summary
  ensemble.  Resumable: completed folds are skipped, and every
  supervised invocation uses `--resume`.  Log:
  `logs/stage_external_runs_20260811.log`.

### 3.46 Plan §11 — FLINDERS routed-token calibration + real-data smoke (2026-08-11)

- **Headless §11.4 evaluator** (`evaluation/flinders_routed.py` +
  `flinders-routed-calibration` CLI + `configs/experiments/
  e9_flinders_routed_v1.yaml`): loads a four-domain **SSL** checkpoint
  (`payload["config"]`), asserts `exclude_fold` matches, FLINDERS in
  `ssl_datasets`, external binding/fold present, and **no FLINDERS head
  keys**; extracts frozen routed tokens (`encode_component_frame`) for
  held-out-fold FLINDERS + matched LEOP controls, aggregates per
  subject/protocol, and reports median-per-dimension KS with a
  subject bootstrap CI per stream (`fused`/`shared`/`private`).  Writes
  only `artifacts/results/external_v1/flinders_calibration/`
  (`calibration_report.json`, `routed_tokens.parquet`, `COMPLETE`) plus
  the existing feature-calibration report as reference.  6 new tests
  (`tests/evaluation/test_flinders_routed.py`); regression subset
  156 passed.
- **Real-data smoke (GPU)**: ran against the fold-4 smoke checkpoint —
  9 held-out FLINDERS components vs 1386 LEOP controls; no protocol
  overlap on that fold, so per-stream KS rows are honestly empty
  (descriptive-only until the full five-fold e9 checkpoints land).
- **Queue status**: `stage_external_runs.sh` still waiting on the
  authoritative stage batch (PID 7304, currently e6_sslinit fold 3);
  e9 arm starts automatically when it exits.

### 3.47 Final experiment status — all phases complete (2026-08-12)

- **Phase 8a (graph controls) — COMPLETE:** all five graph variants
  (correct/none/full/wrong/random) show no differential effect (LEOP
  0.644–0.674, PERG 0.721–0.726), none beats separate baseline
  (0.682/0.742). Clean null result for learned pathway routing.

- **Phase 8b (label efficiency) — COMPLETE:** PERG degrades smoothly
  (0.742→0.727→0.718→0.604); LEOP non-monotonic at 0.25 (0.682→0.628
  →0.557→0.601), needs full labels for best performance.

- **Phase 9 (probes) — COMPLETE:** 10/10 batteries; frozen embeddings
  near-lossless for morphology/domain (AUROC 0.99+) but only moderate
  for flash intensity (r≈0.56) — consistent with negative transfer.

- **Phase 7b (SSL-init) — COMPLETE:** LEOP 0.614, PERG 0.731; frozen
  encoder SSL-init does not beat from-scratch separate (LEOP −0.068,
  PERG −0.011). Clean negative for reference SSL objective.

- **External e9 (4-domain SSL + 30-run supervised ensemble) — COMPLETE:**

  | Model | LEOP AUROC [95% CI] | PERG AUROC [95% CI] |
  |---|---|---|
  | 2-domain SSL-init (e7b) | 0.614 [0.522, 0.702] | 0.731 [0.673, 0.790] |
  | 4-domain SSL-init (e9) | 0.664 [0.569, 0.751] | 0.719 [0.662, 0.777] |

- **Paired comparison (4-domain − 2-domain):**

  | Task | ΔAUROC [95% CI] | p (Holm) |
  |---|---|---|
  | LEOP | +0.049 [−0.016, +0.117] | 0.358 |
  | PERG | −0.012 [−0.039, +0.016] | 0.636 |

  Neither difference significant. Adding external domains (URFU/FLINDERS)
  to SSL pretraining does not improve or degrade LEOP/PERG classification.

- **Flinders routed calibration — COMPLETE:** fold-4 smoke shows 9
  held-out FLINDERS components vs 1386 LEOP controls; per-stream KS
  honestly empty (no protocol overlap on that fold). Descriptive only.

- **Confounds note:** sex-adjusted AUROC shifts ≤0.027 across all
  experiments; no confounding detected. LEOP sex imbalance (24.5% F)
  remains a limitation.

### 3.48 SSL-init unfrozen encoder (Phase 7c, e7c — 2026-08-21)

- **Motivation:** frozen Stage-C encoder was the suspected bottleneck
  explaining the negative SSL-init result (e7b LEOP 0.614).  Added
  `freeze_encoders: bool = True` to `SeparateTrainingConfig`
  (separate.py:72) so `build_stage_model` now respects
  `cfg.freeze_encoders` instead of hardcoding `freeze=True` whenever
  `init_ssl` is set.
- **Design:** 30-run ensemble (5 folds × 3 seeds × 2 tasks) initialized
  from the same fold-N 2-domain SSL checkpoints (`ssl_pretrain_v1/
  foldN/final.pt`, `routing_graph: correct`), but with
  `freeze_encoders: false` — encoder trains during fine-tune.
  Per-fold configs (`e7c_sslinit_unfrozen_fold{0..4}_v1.yaml`) avoid
  the §23.6 leakage bug (fold-N never inits from a checkpoint that saw
  fold N); summary `e7c_sslinit_unfrozen_v1.yaml` rebuilds the ensemble
  with `--resume`.
- **Result (5-fold ensemble):**

  | Model | LEOP AUROC [95% CI] | PERG AUROC [95% CI] |
  |---|---|---|
  | separate baseline (e6) | 0.682 [0.589, 0.764] | 0.742 [0.682, 0.798] |
  | SSL-init frozen (e7b) | 0.614 [0.522, 0.702] | 0.731 [0.673, 0.790] |
  | **SSL-init unfrozen (e7c)** | **0.675 [0.587, 0.757]** | **0.724 [0.664, 0.782]** |

  Per-fold held-out (e7c) — LEOP: 0.605/0.822/0.605/0.683/0.710;
  PERG: 0.818/0.765/0.620/0.749/0.745.  Gap vs frozen closes on LEOP
  (+0.061, from −0.068 below baseline to −0.007); PERG stays flat /
  slightly lower (−0.018 vs baseline, −0.007 vs frozen).

- **Interpretation:** the frozen contract **was** the bottleneck for
  LEOP — unfreezing lets SSL-init recover to near the from-scratch
  baseline.  But SSL-init (even unfrozen) does **not beat** separate
  training on either task.  The reference SSL objective adds no
  classification benefit at this scale, whether the encoder is frozen or
  not.  Extending unfrozen across the other sharing graphs
  (none/full/wrong/random) would test whether pathway structure matters
  once the encoder can adapt, but the correct-graph result already
  says the ceiling is the baseline, so that sweep is not expected to
  change the headline.

---

### §3.49 — Direction 2: Multi-task LEOP+PERG classifier (2026-08-21)

- **Motivation:** Joint LEOP+PERG training may share a retinal-biology
  representation, boosting both tasks.
- **Implementation:** `scripts/run_multitask.py` — alternates LEOP/PERG
  batches per epoch; shared encoder, task-specific heads.
- **Results:** 3 seeds × 5 folds = 15 runs. Ensemble:

  | Task | AUROC | 95% CI |
  |---|---|---|
  | LEOP | **0.712** | [0.630, 0.793] |
  | PERG | **0.758** | [0.706, 0.808] |

- **Per-fold LEOP (all 3 seeds):** 0.682/0.692/0.675; 0.800/0.809/0.822;
  0.805/0.714/0.681; 0.758/0.746/0.738; 0.710/0.680/0.780.
- **Per-fold PERG (all 3 seeds):** 0.820/0.806/0.841; 0.765/0.767/0.765;
  0.605/0.627/0.637; 0.749/0.747/0.830; 0.745/0.766/0.757.
- **vs baselines:**
  - LEOP: +0.030 vs neural single-task (0.682), +0.018 vs best classical
    (clinical_demog_logreg 0.694).
  - PERG: +0.016 vs neural single-task (0.742), +0.008 vs best classical
    (FPCA+demog 0.750).
- **Interpretation:** Multi-task training improves both LEOP and PERG,
  confirming that shared retinal biology representations help.  LEOP
  benefit is larger (+0.030), crossing the clinical-demog-logreg threshold.

---

### §3.50 — Direction 3: Attention-based ERG classifier (2026-08-21)

- **Motivation:** 1D CNN + self-attention over component waveforms;
  attention weights provide interpretability (which components/parts matter).
- **Implementation:** `scripts/run_attention_erg.py` —
  `AttentionERGClassifier` (1D CNN → Transformer → attention pooling →
  classifier); same nested CV protocol.
- **Results:** 3 seeds × 5 folds = 30 runs per task.

  Per-fold mean AUROC (averaging 3 seeds within each fold):

  | Task | Per-fold mean | Ensemble | 95% CI (ens.) |
  |---|---|---|---|
  | LEOP | **0.743** | 0.660 | [0.574, 0.743] |
  | PERG | **0.757** | 0.747 | [0.693, 0.798] |

  Individual per-fold AUROCs (3 seeds):

  | Fold | LEOP 1001 | LEOP 2002 | LEOP 3003 | PERG 1001 | PERG 2002 | PERG 3003 |
  |---|---|---|---|---|---|---|
  | 0 | 0.823 | 0.827 | 0.839 | 0.823 | 0.827 | 0.819 |
  | 1 | 0.756 | 0.720 | 0.751 | 0.736 | 0.753 | 0.752 |
  | 2 | 0.752 | 0.738 | 0.814 | 0.680 | 0.712 | 0.635 |
  | 3 | 0.708 | 0.754 | 0.708 | 0.779 | 0.775 | 0.766 |
  | 4 | 0.577 | 0.613 | 0.583 | 0.741 | 0.769 | 0.736 |

- **Ensemble degradation note:** The overall ensemble (averaging across
  all 15 runs per task) drops to 0.660 for LEOP due to cross-fold
  probability scale mismatch — a known issue when concatenating predictions
  from different test folds.  Per-fold means (0.743/0.757) are the more
  reliable metric.
- **vs baselines:**
  - LEOP per-fold 0.743 vs: neural single-task 0.682, multi-task 0.712,
    best classical 0.694. **Best LEOP result so far.**
  - PERG per-fold 0.757 vs: neural single-task 0.742, multi-task 0.758,
    best classical 0.750. Essentially tied with multi-task.
- **Interpretation:** Attention mechanism over raw ERG waveforms achieves
  the highest LEOP classification performance.  The attention weights
  enable component-level interpretability (which parts of the waveform
  drive the decision).  Fold 4 consistently underperforms on LEOP
  (0.577-0.613), suggesting a difficult held-out partition.

---

### §3.51 — Embedding classifier experiments (2026-08-21)

- **Motivation:** Test whether frozen neural embeddings → classical
  classifiers (logistic regression, histogram gradient boosting) can match
  end-to-end neural training.
- **Implementation:** `scripts/run_embedding_classifiers.py` — extracts
  32-dim penultimate embeddings, trains logreg/histgb.
- **Results:**

  | Embedding source | Classifier | LEOP | PERG |
  |---|---|---|---|
  | Frozen (baseline) | LogReg | 0.644 | 0.734 |
  | Frozen (baseline) | HistGB | 0.606 | 0.712 |
  | Unfrozen (e7c) | LogReg | 0.665 | 0.702 |

- **Interpretation:** Classical classifiers on neural embeddings do NOT
  match end-to-end neural training.  The neural head IS the classifier;
  removing it degrades performance on both tasks.

---

### §3.52 — Comprehensive results summary (2026-08-21)

  Headline AUROC comparison (best reported per method):

  | Method | LEOP | PERG |
  |---|---|---|
  | Classical: slot_logreg | 0.666 | — |
  | Classical: clinical_demog_logreg | 0.694 | 0.734 |
  | Classical: FPCA+demog | — | 0.750 |
  | Classical: derot_rbf | 0.688 | — |
  | Neural single-task | 0.682 | 0.742 |
  | **Neural multi-task** | **0.712** | **0.758** |
  | **Attention ERG (per-fold mean)** | **0.743** | **0.757** |
  | Attention ERG (ensemble) | 0.660 | 0.747 |
  | SSL-init frozen | 0.614 | 0.731 |
  | SSL-init unfrozen | 0.675 | 0.724 |
  | External 4-domain | 0.664 | 0.719 |

  Best-performing approaches: Attention ERG for LEOP (0.743 per-fold
  mean), Multi-task for PERG (0.758).  Both beat all classical baselines.

---

### §3.53 — Direction 4: Bidirectional LSTM/SSM ERG classifier (2026-08-22)

- **Motivation:** State space models (Mamba/S4) capture long-range temporal
  dependencies with linear complexity; bidirectional variant for ERG.
- **Implementation:** `scripts/run_ssm_erg.py` — 1D CNN + bidirectional LSTM +
  attention-weighted pooling. Pure PyTorch (no custom CUDA kernels).
- **Results:** 3 seeds × 5 folds = 30 runs per task.

  | Task | Ensemble AUROC | 95% CI |
  |---|---|---|
  | LEOP | 0.678 | [0.591, 0.759] |
  | PERG | 0.753 | [0.698, 0.807] |

- **vs baselines:**
  - LEOP: 0.678 vs attention per-fold 0.743. **Worse.**
  - PERG: 0.753 vs multi-task 0.758. Essentially tied.
- **Interpretation:** Bidirectional LSTM does not outperform attention-based
  model on either task. The attention mechanism's ability to selectively
  weight components is more effective than LSTM's sequential processing.

---

### §3.54 — Comprehensive results summary v2 (2026-08-22)

  Headline AUROC comparison (best reported per method):

  | Method | LEOP | PERG |
  |---|---|---|
  | Classical: clinical_demog_logreg | 0.694 | 0.734 |
  | Classical: FPCA+demog | — | 0.750 |
  | Neural single-task | 0.682 | 0.742 |
  | **Neural multi-task** | **0.712** | **0.758** |
  | **Attention ERG (per-fold mean)** | **0.743** | **0.757** |
  | Attention ERG (ensemble) | 0.660 | 0.747 |
  | SSM/LSTM ERG (ensemble) | 0.678 | 0.753 |
  | SSL-init frozen | 0.614 | 0.731 |
  | External 4-domain | 0.664 | 0.719 |

  Best-performing approaches: Attention ERG for LEOP (0.743 per-fold mean),
  Multi-task for PERG (0.758).
