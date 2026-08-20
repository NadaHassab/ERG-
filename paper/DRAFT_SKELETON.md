# Pathway-Constrained Partial Transfer Across Unpaired Retinal Electrophysiology Protocols

## Abstract (draft)
- Two unpaired retinal electrophysiology datasets (LEOP flash ERG, PERG-IOBA pattern ERG)
- Core question: can pathway-constrained partial sharing outperform full or zero sharing?
- Simulation confirms theory at moderate mismatch; real data at 92k params/n=160–336 shows no benefit
- Contributions: sOT descriptor, pathway-constrained architecture, confound-gated evaluation, honest negative results

## 1. Introduction
- Retinal electrophysiology measures overlapping biology via different protocols
- Transfer learning promise: share knowledge across protocols without shared patients
- Challenge: what to share? Full sharing risks negative transfer; zero sharing wastes data
- Our answer: pathway-constrained partial sharing — share only the biologically overlapping waveform region
- Paper structure: framework, simulation validation, real-data evaluation, honest negative results

## 2. Background & Related Work
- ERG/PERG signal biology
- Transfer learning across clinical domains
- Optimal transport for waveform alignment
- Partial/federated transfer learning

## 3. The sOT Descriptor
- Signed derivative optimal transport: transport between positive and negative derivative signals
- Describes protocol mismatch as a scalar (or vector) distance
- Properties verified (E1): offset, shift, amplitude, noise, polarity invariance
- See `src/pathway_erg/math/signed_ot.py`

## 4. Simulation: E2 Partial Sharing Theory
- Setup: 2 linear tasks in R^6, 2 shared + 2 private dimensions, orthogonal design
- Grid: mismatch {0, 0.25, 1, 4} × noise {0.1, 1} × n {100, 1000} × 300 reps
- Result: full sharing best at zero mismatch; oracle-partial best at moderate/high mismatch; separate best at high mismatch
- Learned gate approximates oracle-partial but not perfectly
- See Figure 1 (heatmap), Table 6

## 5. Methods: Pathway-Constrained Architecture
- Encoder: 92k param MLP with frozen embeddings
- Pathway router: per-component gating based on sOT-derived constraints
- Graph controls: correct / none / full / wrong / random
- SSL pretraining: contrastive loss across domains
- Evaluation: nested cross-validation, patient-level AUROC, clustered bootstrap CI

## 6. Data
- LEOPs: 253 subjects, 5309 flash curves, ASD/ASD+ADHD/Control
- PERG-IOBA: 304 subjects, 336 visits, 1354 eye curves, Normal/Abnormal
- External: URFU (wavelet ERG), FLINDERS (full-field ERG) — 431 recordings, 1253 components
- Preprocessing: landmarks, sOT, VMD, QC gating
- Nested folds: 5-fold patient-level

## 7. Experiments & Results

### 7.1 Baseline Performance (Table 1 context)
- Separate neural model: LEOP 0.682, PERG 0.742
- Classical baselines: FPCA+demographics 0.750 (PERG), slot_logreg 0.747 (LEOP)
- Confound signals: sex imbalance in LEOP (24.5% F), QC shortcuts

### 7.2 Graph Control Ablations (Table 1, Figure 2)
- Five sharing strategies tested on pathway architecture
- All cluster tightly (LEOP 0.644–0.674, PERG 0.721–0.726)
- None beats separate baseline → clean null result
- Interpretation: at 92k params, model cannot exploit pathway structure

### 7.3 Label Efficiency (Table 2, Figure 3)
- PERG degrades smoothly (0.742→0.718→0.604 at 0.1)
- LEOP non-monotonic, needs full labels
- Implications for low-resource clinical deployment

### 7.4 Expert-Fidelity Probes (Table 3, Figure 4)
- Frozen embeddings near-lossless for morphology/domain (AUROC >0.99)
- Only moderate for flash intensity (r≈0.56)
- Explains negative transfer: encoder memorizes perceptible structure without classification-relevant contrast

### 7.5 SSL Pretraining (Section 7b context)
- 5-fold joint SSL + SSL-init fine-tune
- LEOP 0.614, PERG 0.731 — worse than from-scratch
- Frozen-encoder Stage C contract limits benefit

### 7.6 External Domain Transfer (Tables 4–5, Figure 5)
- 4-domain SSL (LEOP+PERG+URFU+FLINDERS) + 30-run supervised ensemble
- 2-domain vs 4-domain: ΔLEOP +0.049 (p=0.36), ΔPERG −0.012 (p=0.64)
- Adding external domains neither helps nor hurts
- Flinders routed calibration: descriptive only (no protocol overlap in held-out fold)

## 8. Discussion
- Theory confirmed in simulation; null in real data
- Why the gap? Model capacity (92k params), sample size (n=160–336), protocol mismatch unknown
- The frozen-encoder contract is the limiting factor
- Honest negative: pathway-constrained partial transfer shows no benefit at this scale

## 9. Limitations
- Small cohorts (LEOP 160, PERG 336 after nesting)
- Unknown real-world protocol mismatch (sOT assumes linear relationship)
- Sex imbalance in LEOP (24.5% F)
- External datasets gated (URFU labels pending clinical review)
- Single model architecture (92k params)

## 10. Conclusion
- Framework + simulation validated; real-data evaluation honest
- Confound-gated evaluation protocol as contribution
- Pathway routing is a promising direction but needs larger cohorts and unfrozen encoders
- Future: larger models, unfrozen Stage C, multi-site data

## References

## Appendix
- A. sOT mathematical properties
- B. Simulation E2 full grid
- C. Probe battery details
- D. Confound gate methodology
- E. External dataset protocols
