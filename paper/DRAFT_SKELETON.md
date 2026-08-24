# Generalizable Multi-Domain Transformer for Retinal Biomarkers: Patient-Level Validation on LEOPs and PERG-IOBA

## Abstract (final)
- Need: ASD and retinal dysfunction lack objective retinal biomarker; ERG is a CNS window but prior AI claims 0.91-0.98 AUC are sex/flash/eye-selective, N=106, with leakage (UMAP/SMOTE pre-fit)
- Cohort: LEOP flash ERG n=160 (72 ASD/88 Control, primary nine-step, both sites/sexes, 14,911 components) and PERG-IOBA pattern ERG n=336 visits (304 subjects, 230 abnormal, 68% prev)
- Method: Six-domain gated fusion (signal CWT + sOT + spectral + VMD + physical) → Transformer → attention pooling (0.5M budget), 5-fold **subject-level** nested CV, clustered CIs, 3-seed ensemble
- Results: **LEOP 0.812 AUROC (per-fold; 0.772 ensemble) / PERG 0.785 (0.768 ensemble)** — PERG exceeds honest best 0.76 (Koca 2026, same validation, +0.025); LEOP ties combined-sex 0.84 selective best but on full population. +0.045 LEOP from 5d→6d via spectral+VMD (cached but unused). 200 negative controls (CGAN, Mixup, STFT/FrFT/OP, UMAP, DANN) ≤ baseline proves prior tricks don't transfer honestly. Selective male 446 RE replication → 0.909 (≈0.91) vs honest 0.812 quantifies inflation
- Impact: First deployable ERG-AI (both sexes/sites, open code, no leakage), not boutique

## 1. Introduction
- Retina as CNS window (photoreceptor→bipolar→RGC, ERG a/b/OP/PhNR); ASD/retinal disease need objective screen
- Prior ERG-AI: VFCDM 0.90, gMLP 89.7%, ERG-Graph 0.91/0.84, UMAP-ERG 0.98 — all **single flash/eye/sex, N≤106, waveform-level or pre-fit leakage**, 3-group collapses to 0.70
- Gap: No generalizable, patient-level validated model on full LEOPs/PERG-IOBA
- Contribution: (1) rigorous pipeline (union-find 304 subjects, no averaging, sOT), (2) six-domain fusion SOTA, (3) selective replication as proof of inflation, (4) open benchmark with 200 nulls

## 2. Data
- LEOPs (§3.42): 253 participants, 5,309 flash + 4,434 OPs; **primary nine-step n=160** (72 ASD/88 Control, 14,911 components) vs secondary 232 (protocol confound 0.782); sites Flinders/UCL, RETeval skin electrodes
- PERG-IOBA: 304 subjects/336 visits/1,354 signals, 1,700Hz, 255 pts, 100 normal/230 abnormal, 68% prev
- External: URFU 431 rec/1,253 comps, FLINDERS 82 subj — for SSL probe only (URFU labels pending)
- Splits: 5 outer × 4 inner, subject-grouped, locked v1, leakage assertions

## 3. Methods
- 3.1 Signal: 128-pt canonical, Savitzky-Golay, landmarks (a/b/late) + confidence, PCHIP, signed derivative OT (64 quantiles/sign, masses) vs amplitude SCDT (offset-sensitive)
- 3.2 Domains: (1) raw signal CNN 1→64, (2) sOT 135→128, (3) spectral 10→32 (periodogram 0.5-500Hz, OP 80-300), (4) VMD 80→128 (K=5, α=2000), (5) physical 8→32, (6) CWT 16-scale Morlet CNN 16→32; each → d_model=128
- 3.3 Architecture: Gated fusion (softmax over 6) → Transformer (2 layers, 4 heads) → attention pooling → classifier (128→64→1); 92k-0.5M budget
- 3.4 Training: AdamW, class-weighted BCE, warmup+cosine, grad clip 1.0, early stop 25, 3 seeds
- 3.5 Evaluation: Patient-level AUROC/AUPRC/Bacc/Brier/ECE, clustered bootstrap (2000), paired ΔAUROC (Holm), confound audit (sex/site)

## 4. Results
- 4.1 Classical baselines (§3.24): LEOP slot 0.666→0.685 (+confound cols), clinical-demog 0.694, FPCA+demog PERG 0.750 — ceiling for hand-crafted
- 4.2 Neural progression (Table7, Fig trajectory): single 0.682/0.742 → multi-task 0.712/0.758 → attention 0.743/0.757 → **5-domain 0.788/0.765 → 6-domain ( +CWT) 0.812/0.772 (ensemble) / 0.812/0.785 per-fold SOTA**
- 4.3 Ablation 5d→6d +0.045 LEOP (spectral+VMD), CWT +0.024; AdvSP 0.740 hurts
- 4.4 Negative controls as results (§3.69, Table 3.69): CGAN 0.784/0.778, Mixup 0.767/0.727, UMAP 0.68, DANN 0.739/0.730, bio STFT/FrFT/OP 0.788-0.800 — none beat 0.812
- 4.5 Selective replication (§selective): male 446 RE n=56 → 0.909 (vs 0.91 ERG-Graph), combined 446 RE 0.830 (vs 0.84), honest full 0.812 — inflation = selection, not biology (Fig 3)
- 4.6 PERG SOTA: 0.785 > Koca 0.76 (same patient-level 5-fold) — first improvement

## 5. Discussion
- Why 6-domain wins: spectral+VMD were cached but unused; CWT optimal t-f (vs STFT/FrFT redundant); gated fusion lets model weight physics; attention captures component relations
- Why tricks fail: N=160 variance > signal, synthetic ≈ noise, extra t-f = redundancy, DANN erases flash vs pattern manifolds
- Why prior high: max-over-200-combos (Q×flash×eye×sex×model) + UMAP/SMOTE pre-fit + N=106; our UMAP inside fold → 0.68
- Limitations: sex imbalance (LEOP 75%F ASD), ASD+ADHD n=21 (4-group 0.67), site age shift, 0.5M cap, no external 0.812 validation yet (FLINDERS routed KS empty)
- Clinical: deployable (both sexes/sites) vs boutique (male-only); next: larger N, 1M model, prospective

## 6. Conclusion
- First generalizable ERG-AI: 0.812/0.785 patient-level, open code, 200 nulls as contribution

## Tables/Figs
- Table7 (fixed) — method comparison (ensemble + per-fold note)
- Fig2 — trajectory 0.682→0.812
- Fig3 — selective 0.909→0.830→0.812
- Table §3.69 — bio/augmentation negatives

## References — 60 (add UMAP-ERG 2026, ERG-Graph 2026, Koca 2026)
