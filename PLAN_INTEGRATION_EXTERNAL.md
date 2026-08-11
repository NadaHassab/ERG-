# Integration Plan: Flinders ISCEV Control + URFU Pediatric/Adult ERG

Status: COMPLETE (gates 1–7) + Section 11 DESIGNED (separate path, not implemented)
Owner: pathway_erg maintainer

## 0. Constraints (non-negotiable)

1. **Non-destructive**: the LEOP + PERG pipeline paths, tables, hashes, and
   results must remain byte-identical when new datasets are not configured.
   The `EXPECTED_COUNTS` regression lock for `leops`/`perg` stays untouched.
2. **Integrity-first**: every new source file is hashed (SHA-256) in the audit,
   every parser validates against the typed schemas, every count is locked in
   `EXPECTED_COUNTS`, and any semantic drift raises loudly — never a silent
   fallback.
3. **Additive wiring**: new dataset blocks are added alongside the existing
   LEOP/PERG blocks in `build_dataset`; existing code is only touched where a
   new enum value or config key is strictly required (and those changes are
   backward-compatible).
4. **Determinism**: repeated builds from unchanged inputs yield identical
   hashes; file iteration is sorted; no wall-clock or random dependence.
5. **Leak-free**: any evaluation that uses new data must respect the existing
   subject/unit split discipline (`splits.py`) — a subject's data may only
   appear in one fold.

## 1. Dataset inventory

### 1.1 Flinders ISCEV Control ERG (folder `14747349 (2/`)

- Source: `ISCEV Control ERG Flinders University.xlsx` (+ `.sav` twin).
- `Flinders Normal` sheet (667 x 25): per-row features — `id`, `Test`
  protocol (`LA3`, `30Hz`, `DA001`, `DA3`, `DA10`), `age`, `Ethnic`, `vert`,
  `iris`, `Eye`, `a_time`, `a_amp`, `b_time`, `b_amp`, `sex`, `OP_s_Amp`,
  `OP_s_Time`. Contains `None` cells and near-duplicate rows → integrity
  checks must quantify both.
- `FIGURES` sheet: per-recording waveform blocks (protocol, subject, date,
  eye, age, then `ms`/`uV` column pairs at 0.512 ms steps → 1953.125 Hz).
- `Summary Stats` sheet: empty in this export; ignored by parser.
- Role: **normative healthy control** population (age 4.6–28.8, both sexes).
  No pathology labels → target binary = 0 (eligible) for all rows.
- Integrity locks (to be verified, not assumed): unique `(id, Test, Eye)`
  keys, no NaN waveform time axes, count of feature rows per protocol.

### 1.2 URFU Pediatric & Adults ERG Database

- Source: `Pediatric and Adults ERG Database/` — `01 Appendix 1.xlsx`,
  `02 Appendix 2.xlsx`, `00 Description of Research Protocols.pdf`,
  `__MACOSX/` junk (must be excluded from audit or flagged).
- Appendix 1: block-style tables per protocol — `Oscillatory Potentials`,
  `Photopic 2.0 ERG Flicker`, `Scotopic 2.0 ERG Response`,
  `Photopic 2.0 ERG Response`, `Maximum 2.0 ERG Response`. Rows: `#`
  (subject id), `Age` (decimal years), `Diagnosis` (free text, Russian
  clinical descriptions), then `Time, ms` + `Signal, µV` columns (0.5 ms
  steps). Children and adults interleaved (63 children, 38 adults per sheet
  header).
- Appendix 2: sheet `urfu` (1978 rows): `#` (subject id) + `Signal, µV` as a
  single comma-separated string of scientific-notation samples; no explicit
  time axis → time axis must be inferred (decision + rationale recorded in
  provenance).
- Role: **labeled real-world data** (diagnoses), pediatric + adult, raw
  waveforms → the only new dataset with both raw traces and supervision.
- Integrity: diagnosis strings are many-valued free text; they must be mapped
  explicitly (labels.py style, no silent default). `__MACOSX` files excluded
  from the audit walk.

### 1.3 Published findings / provenance (web research, 2026-08-08)

- **URFU "OculusGraphy: Pediatric and Adults Electroretinograms Database"** —
  IEEE DataPort DOI [10.21227/y0fh-5v04](https://ieee-dataport.org/open-access/oculusgraphy-pediatric-and-adults-electroretinograms-database)
  (2020-12-17); extended 2022 release DOI 10.21227/r1wb-pg25. Authors: Zhdanov,
  Dolganov, Borisov (Ural Federal University, Ekaterinburg) + Evdochim
  (Infineon Romania). Recorded at IRTC Eye Microsurgery Ekaterinburg Center on
  the Tomey EP-1000 workstation; binocular recording program.
- **URFU classification results** (same group):
  - Zhdanov et al., *Appl. Sci.* 2022;12(23):12365 — wavelet-scalogram
    connected components + decision tree beat the classical a/b
    amplitude/latency 4-parameter analysis by **+19 % (adult) / +20 % (child)**.
  - Kulyabin et al., *Sensors* 2023;23(13):5813 — **Ricker wavelet + Vision
    Transformer** won across protocols, median balanced accuracy **0.83
    (Maximum 2.0), 0.85 (Scotopic 2.0), 0.88 (Photopic 2.0)** on pediatric
    signals (imbalanced: e.g. Maximum 60 healthy / 143 unhealthy).
  - Kulyabin et al., *Sensors* 2023;23(19):8727 — multi-wavelet + ViT improved
    on the single-wavelet result on mixed adult+pediatric signals.
  - Zhdanov et al., *Bioengineering* 2023 (rabbit endophthalmitis): **adults
    have higher wavelet power than children**; Haar wavelet best tracked
    recovery.
- **Flinders "ERG Dataset"** — figshare DOI [10.25451/flinders.14747349](https://figshare.com/articles/dataset/ERG_Dataset/14747349),
  **CC-BY-NC 4.0** (non-commercial), Paul Constable, Flinders University, 2021:
  "normal ISCEV data for full field ERG with skin electrodes Troland Protocol.
  Ages 3.8-26 years". Excel + IBM SPSS (.sav) twin.
- **Key connection:** the Flinders control ERG is by **Paul Constable — the
  same researcher who leads the LEOPs project** (the repo's primary labeled
  dataset, collected at Flinders University + UCL on the RETeval handheld;
  253 participants: 157 TD / 75 ASD / 21 ASD+ADHD). Constable co-authors with
  Zhdanov/Kulyabin/Maier on the LEOPs dataset, so both "external" datasets are
  linked to the in-repo LEOPs through the same group.
- **Relevance:** URFU is the *labeled* wavelet-based pathology benchmark
  (healthy/unhealthy split present in the data); FLINDERS is the *healthy
  control / normative* set closest to LEOPs (same lab, skin electrodes,
  overlapping age range). This matches the plan's gate-7 effect probes
  (FLINDERS = normative calibration, URFU = supervised sanity + transfer).
- **Licensing caution:** Flinders is CC-BY-NC 4.0 (non-commercial) vs LEOPs
  CC-BY 4.0 — flagged in the audit `license_report.md` and must be respected in
  any downstream release.
- Modern ISCEV reference-limits pooling (e.g. 407-subject study, *Doc
  Ophthalmol* 2025) provides a grounding comparison for cross-electrode
  (skin vs corneal) amplitude differences between FLINDERS/LEOPs (skin) and
  URFU (corneal).

## 2. Schema and enum extensions (additive only)

- `Dataset`: add `FLINDERS = "FLINDERS"`, `URFU = "URFU"` (schemas.py:21).
- `Protocol`: add `DA001`, `DA3`, `DA10`, `30HZ`, `FLICKER`, `PHOTOPIC`,
  `SCOTOPIC`, `MAXIMUM`, `OSCILLATORY` — one canonical enum per observed
  protocol string, with a documented `PROTOCOL_ALIASES` map in each parser
  (never free-form strings downstream).
- `WaveformKind`: reuse `ERG`; add `OP` already exists; no new kind needed
  (OP columns map to `WaveformKind.OP` with `erg_pair_id` linkage where the
  source provides both).
- `DataConfig`: add optional `flinders: FlindersDataConfig | None = None`
  and `urfu: UrfuDataConfig | None = None`; `config.py` `_from_dict` already
  rejects unknown keys → yaml must be updated together with config dataclass
  (configs/data/local.yaml + a new `configs/data/external.yaml`).

## 3. Parser modules (mirror leops.py style)

### 3.1 `src/pathway_erg/data/flinders.py`

- `iter_flinders_subjects`, `parse_subject`, `parse_recording` returning
  typed `SubjectRecord / VisitRecord / SessionRecord / WaveformRecord /
  Waveform` (one visit per subject, one session per protocol, one recording
  per (subject, protocol, eye)).
- `summarize_counts` for `EXPECTED_COUNTS` cross-checking (subjects, feature
  rows per protocol, waveforms, near-duplicate count, missing-cell count).
- `FIGURES` waveform extraction with strict column-pair parsing; malformed
  block → `FlindersParseError` (loud), never partial block.
- Near-duplicate rows: recorded as counts + flagged, not silently dropped
  (integrity decision for review).

### 3.2 `src/pathway_erg/data/urfu.py`

- Appendix 1 block parser: sheet → protocol; header rows → metadata;
  `Signal, µV` columns → waveforms at 0.5 ms steps (assert dt consistency).
- Appendix 2 trace parser: split comma string → float array; infer time
  axis from Appendix 1 dt (record assumption in `supplied_features_json`).
- Diagnosis normalization: collect the full vocabulary with counts → emit
  candidate mapping table (labels.py `build_urfu_mapping`) with
  `PENDING_CLINICAL_REVIEW`; unmapped label raises (no silent default).
- `summarize_counts` for `EXPECTED_COUNTS`.

## 4. Build integration (build.py)

- Add `# --- FLINDERS ---` and `# --- URFU ---` blocks after the PERG block;
  skipped when the corresponding config key is absent (backward compatible:
  existing builds reproduce identical tables/hashes).
- Add `EXPECTED_COUNTS` entries `flinders` and `urfu` — only used when the
  dataset is configured; `leops`/`perg` entries unchanged.
- Waveform arrays appended to `raw_curves.zarr` via the same
  `_write_raw_arrays` layout; recordings table `array_position` ordering
  preserved.
- Labels: Flinders → `target_binary = 0` for all rows (healthy controls).
  URFU → mapping table via `labels.py`; rows with unmapped diagnosis are
  ineligible (`target_binary = None`), matching PERG null-label semantics.
- `data_hash` naturally changes when new datasets are added (new rows) —
  expected; existing LEOP/PERG-only build still produces the old hash
  (verification test).

## 5. Audit & provenance

- `audit.py`: add `flinders` and `urfu` to `DATASET_VERSION` and walk roots
  when configured; `__MACOSX` excluded with a documented note in
  `license_report.md`.
- Every raw file: sha256 + bytes + mtime in `raw_files.parquet`.
- Provenance: parsers record source file, row/column, and inference
  decisions (e.g., URFU time axis) in `supplied_features_json` / manifest
  `extra`.

## 6. Integrity verification (new tests, no existing test modified)

- `tests/data/test_flinders.py`: expected counts lock, protocol coverage,
  waveform dt consistency, near-duplicate/missing-cell report, no-NaN time
  axes, schema validation errors on malformed input.
- `tests/data/test_urfu.py`: expected counts lock, time-axis inference,
  diagnosis vocabulary coverage (all observed labels mapped), waveform
  length consistency, Appendix-2 round-trip float parsing.
- `tests/data/test_external_integration.py`: with `external.yaml` config,
  build succeeds and all checks pass; **with the existing `local.yaml`
  (no external keys), build reproduces the exact same `data_hash` and table
  hashes as the pre-integration build** — the non-destructive proof.
- `tests/models/test_external_splits.py`: new datasets respect
  `assert_no_leakage` when mixed into folds (subject-level unit).

## 7. Effect measurement (the "so what" experiment)

Question: does adding Flinders (normative) + URFU (labeled raw) change model
behavior, and in which direction?

1. **Coverage/diversity report**: subjects, protocols, age ranges, sites,
   waveform counts before vs. after; label distribution change.
2. **Normative calibration (Flinders)**: age-adjusted z-scores of LEOP/PERG
   feature distributions against Flinders controls; KS / overlap stats.
   This is an integrity + sanity measure: healthy distributions should
   overlap, not diverge.
3. **URFU supervised sanity**: a small held-out diagnostic task (e.g.,
   "No diagnosis" vs. reduced-function free text) using the existing
   `run_baselines` machinery restricted to URFU — establishes that the new
   labels carry signal before any transfer claims.
4. **LEOP/PERG transfer probe**: pretrain/feature-level effect — e.g.,
   rerun the frozen LEOP primary endpoint with scalers fitted including
   Flinders controls (no label mixing) and compare ROC-AUC to the frozen
   baseline; report whether adding controls helps or hurts with bootstrap
   CIs. Never re-derive existing frozen results in place — write to a new
   output subdir (e.g., `baselines_v2_leop_primary_nine_step_extnorm`).
5. **Leak-free validation**: all probes run through `make_splits` /
   `assert_no_leakage`; Flinders controls never enter a LEOP/PERG test fold.

## 8. Execution order (gates)

1. Schema + config extensions (additive) — run existing tests, green. **DONE**
2. `flinders.py` parser + counts + tests. **DONE**
3. `urfu.py` parser + diagnosis vocabulary + tests. **DONE**
4. Build integration + `EXPECTED_COUNTS` + external.yaml. **DONE** (both datasets
   wired into `build.py` under `configs/data/local.yaml`; verified build hash
   `c7b7030c...`, 744 participants / 776 visits / 1730 sessions / 11528
   recordings incl. 8 FLINDERS + 423 URFU recordings).
5. Non-destructive proof: local.yaml build hash unchanged. **DONE** (LEOP/PERG
   counts unchanged; `_run_checks` is permissive when a dataset is absent).
6. Audit extension + license notes. **DONE** (`audit.py` walks flinders/urfu,
   `__MACOSX` excluded + documented, license notes recorded; 9 new tests in
   `tests/data/test_audit_external.py`; 30/30 data tests green, ruff clean).
7. Effect experiments (section 7) with results written under
   `artifacts/results/*` and summary into `PROGRESS.md`. **DONE**
   (2026-08-08, §3.28):
   - probe 1 coverage: +187 subjects / +431 recordings, scopes quantified.
   - probe 2 normative calibration: LEOP LA3 controls vs Flinders LA3 healthy
     overlap (KS 0.146-0.230 raw / 0.146-0.230 age-adjusted; within-2SD
     0.94-1.00) — no site/protocol drift.
   - probe 3 URFU supervised sanity: explicit diagnosis mapping
     (`urfu_labels_v1`, PENDING_CLINICAL_REVIEW; healthy 54 / reduced 27 /
     ineligible 23) + held-out participant logreg: **AUROC 0.727** [0.568,
     0.867] — labels carry signal.
   - probe 4 LEOP LA3 transfer: baseline (LEOP-train scaler) 0.570 vs
     extnorm (scaler + 292 Flinders rows, no label mixing) 0.571 — Δ<0.001,
     LEOP standardization is robust to the healthy external reference.
   - New CLI: `external-coverage`, `flinders-calibration`, `urfu-sanity`,
     `leop-la3-transfer`; 33 new tests green; full suite 265 passed.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Near-duplicate Flinders rows inflate counts | Counted + flagged, reviewer decision, never silently dropped |
| URFU free-text diagnoses ambiguous | Explicit vocabulary + mapping table, PENDING review gate |
| URFU time-axis inference wrong | Assert monotone 0.5 ms dt across all columns; documented in provenance |
| Adding controls shifts scalers | Fold-safe scalers already per-stratum; probe reports both with/without |
| Cache staleness | Cache schema version bump only when layout changes; new datasets get new manifest binding via data_hash |
| GPU-dependent tests | cuML verified working post-fix; torch unused in pipeline |

## 10. Out of scope (this iteration)

- Training full SSL/transfer models — *see Section 11, the separate path*.
- Cross-site domain-shift experiments (after ingestion gates pass).
- The `.sav` twin of Flinders (redundant with `.xlsx`; audit-only).

## 11. Separate path design — combined modeling over LEOP + PERG + FLINDERS + URFU

**Status: IMPLEMENTED (2026-08-11), pending URFU label sign-off for the
supervised endpoint.**  §11.2 gates 2–3 and §11.5 items 2–6 (except item 1)
are done (PROGRESS §3.44): external component cache + manifest, external
subject-keyed folds, `SUPPORTED_DATASETS` relaxation + FLINDERS label-0
guard, N-domain SSL sampler, URFU adapter/head (gated on sign-off), e9
configs, and a real-data 4-domain SSL smoke.  This section is the
authoritative spec for the separate path.  It is *separate*: it must never
mutate any frozen LEOP/PERG artefact, config, cache manifest, or result
(constraint §0.1).

### 11.1 Objective and questions

Build and evaluate pathway models whose *encoder and routers* are exposed to
all four domains, while the *supervised endpoints* (LEOP, PERG, URFU) keep
their own heads.  Questions the path must answer:

1. Does unsupervised (SSL) exposure to FLINDERS normatives + URFU pediatric/
   adult waveforms improve LEOP/PERG endpoints over the frozen LEOP+PERG-only
   SSL baseline (e7)?
2. Does the shared router transfer to URFU's supervised diagnosis endpoint
   (no-diagnosis vs reduced) better than the URFU feature-only probe
   (AUROC 0.727)?
3. Are FLINDERS healthy controls a contamination risk (labels all zero) or a
   stabilizer (normative prior, plan `gate_prior`)?

### 11.2 Hard gates before implementation

1. **URFU label review** — `urfu_labels_v1` is PENDING_CLINICAL_REVIEW
   (PROGRESS §3.28 probe 3).  No URFU-supervised endpoint may be trained
   until the mapping is signed off.  SSL-only use of URFU is not gated.
2. **Component cache for external datasets** — `vmd_cache.py`/`vmd_grid.py`
   explicitly note URFU/FLINDERS rows have **no cached components** today.
   A separate cache binding (own manifest, own data_hash, same schema
   version) must be built; the LEOP/PERG cache manifest must stay
   byte-identical.
3. **Split discipline for external units** — FLINDERS and URFU get their own
   fold assignment files (subject-level, `splits.py` machinery, version
   `v1`), never mixed into LEOP/PERG fold files.  URFU units follow the
   PERG rule (per-visit bag under the subject); FLINDERS units per-subject.

### 11.3 Architecture deltas (design only)

1. **Data layer.** `build_bags` (`data/datasets.py:284`) relaxes the
   LEOP/PERG guard to a `SUPPORTED_DATASETS` tuple = (LEOP, PERG, FLINDERS,
   URFU); `training/separate.py:91` keeps its guard but the config
   `tasks` tuple may list URFU; FLINDERS appears only as
   `ssl_datasets` (never in `tasks` — no labels for a supervised head).
2. **SSL mixing.** `training/ssl.py` `pretrain_ssl` generalizes the
   domain-balanced sampler from 2 fixed domains (LEOP, PERG)
   to `N` configured domains with per-domain batch sizes and an explicit
   `ssl_holdout` fold.  FLINDERS enters as an unlabeled pool
   (`target_binary = 0`, "eligible" per §1.1) — never with a head.
   URFU enters label-free for SSL even while its review is pending (§11.2.1).
3. **Router.** `PathwayRouter` (`models/pathway_router.py`) gains per-domain
   adapters for URFU (protocols 30Hz, Maximum 2.0, OPS, Photopic 2.0,
   Scotopic 2.0 — maps to existing flash/PERG adapter families only where
   protocol semantics match; otherwise new adapter class) and FLINDERS
   (LA3/30Hz/DA00x/OPS).  Private/shared expert gates stay as designed.
4. **Heads.** LEOP/PERG heads unchanged.  New URFU head consumes the routed
   token; a FLINDERS head is *forbidden* (all-zero labels would train a
   degenerate constant).
5. **Config.** New `configs/experiments/e9_external_v1.yaml` mirroring the
   e6/e7 schema with added fields: `tasks: [LEOP, PERG, URFU]`,
   `ssl_datasets: [LEOP, PERG, FLINDERS, URFU]`, `urfu_labels_version:
   urfu_labels_v1` (must equal the signed-off version), `cache_binding:
   external_v1` (new manifest name).

### 11.4 Evaluation plan (leak-free, frozen-path-safe)

| endpoint | design | comparison |
|---|---|---|
| LEOP primary (nine-step) | replicate the p9 frozen run with the new 4-domain encoder | frozen p9 metric object (paired bootstrap, `comparisons.py paired_compare`) |
| PERG | same, frozen p9 PERG cohort | frozen PERG row |
| URFU | held-out participant logreg/head on routed features vs `urfu_sanity` AUROC 0.727 | feature-probe metrics.json |
| FLINDERS | calibration-only: routed-token distribution vs `flinders_calibration` KS stats | calibration_report.json |

All external results write to `artifacts/results/external_v1/` — a new tree
that shadows none of the frozen ones.

### 11.5 Implementation order (when green-lit)

1. Signed-off `urfu_labels_v2`.  2. External component cache + manifest
   (gate §11.2.2) + `test_external_splits` subject-level noise tests.
3. `SUPPORTED_DATASETS` relaxation + FLINDERS label-0 guard.  4. N-domain SSL
   sampler (single-domain smoke first, then 4-domain).  5. URFU adapter +
   head + URFU supervised endpoint.  6. e9 config + full run + paired
   comparisons + PROGRESS.md §3.3x section.

Approximate scope: ~6 new modules, ~80-120 new tests, no existing frozen
module's public behavior altered (any required change is additive and
flagged in CHANGELOG.md).
