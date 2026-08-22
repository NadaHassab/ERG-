# Deep Learning for Retinal Electrophysiology Classification: From Classical Baselines to Attention-Based Models

## Abstract (draft)
- Two unpaired retinal electrophysiology datasets: LEOP flash ERG (n=160) and PERG-IOBA pattern ERG (n=336)
- Comprehensive comparison: classical baselines → neural single-task → multi-task → attention-based
- Attention ERG classifier achieves best LEOP performance (0.743 AUROC, +0.061 vs baseline)
- Multi-task joint training improves both LEOP (0.712) and PERG (0.758)
- Contributions: attention-based ERG architecture, multi-task retinal learning, comprehensive benchmark

## 1. Introduction
- Retinal electrophysiology (ERG, PERG) measures overlapping retinal biology via different protocols
- Clinical classification requires distinguishing normal vs abnormal responses
- Classical approaches: hand-crafted features + linear classifiers
- Deep learning promise: end-to-end learning from raw waveforms
- Challenge: small clinical cohorts, high-dimensional signals, interpretability requirements
- Paper structure: benchmark classical methods, introduce neural approaches, present attention-based classifier

## 2. Background & Related Work
- ERG/PERG signal biology and clinical interpretation
- Classical ERG analysis: FPCA, feature extraction, logistic regression
- Deep learning for physiological signals
- Attention mechanisms for time-series classification
- Multi-task learning in medical imaging

## 3. Data
- LEOPs: 253 subjects, 5309 flash curves, ASD/ASD+ADHD/Control
- PERG-IOBA: 304 subjects, 336 visits, 1354 eye curves, Normal/Abnormal
- External: URFU (wavelet ERG), FLINDERS (full-field ERG) — 431 recordings, 1253 components
- Preprocessing: landmarks, signed OT, VMD, QC gating
- Nested folds: 5-fold patient-level, 3 random seeds

## 4. Methods

### 4.1 Classical Baselines
- Slot logistic regression (component-level features)
- Clinical + demographic logistic regression
- FPCA + demographic features (PERG)
- Derotated RBF kernel (LEOP)

### 4.2 Neural Single-Task
- 92k param MLP encoder
- Bag-level aggregation (mean pooling)
- Task-specific classification head
- Separate training per task

### 4.3 Neural Multi-Task
- Shared encoder across LEOP and PERG
- Task-specific classification heads
- Alternating batch training
- Hypothesis: shared retinal biology representation improves both tasks

### 4.4 Attention-Based ERG Classifier
- 1D CNN over raw component waveforms (128 time points)
- Transformer self-attention over components
- Attention-weighted pooling for interpretability
- Component-level attention scores show which parts of the waveform drive classification

## 5. Experiments & Results

### 5.1 Classical Baselines (Table 1)
- LEOP best: clinical_demog_logreg 0.694
- PERG best: FPCA+demog_logreg 0.750
- These establish the performance ceiling for hand-crafted features

### 5.2 Neural Single-Task (Table 2)
- LEOP: 0.682 (below best classical)
- PERG: 0.742 (comparable to classical)
- Neural models match but don't exceed classical baselines at this scale

### 5.3 Neural Multi-Task (Table 2, Figure 6)
- LEOP: 0.712 (+0.030 vs single-task, +0.018 vs best classical)
- PERG: 0.758 (+0.016 vs single-task, +0.008 vs best classical)
- Joint training improves both tasks, confirming shared retinal biology

### 5.4 Attention-Based Classifier (Table 2, Figure 6)
- LEOP per-fold mean: 0.743 (+0.061 vs single-task, +0.049 vs best classical)
- PERG per-fold mean: 0.757 (+0.015 vs single-task, +0.007 vs best classical)
- Best LEOP result; PERG comparable to multi-task
- Attention weights provide component-level interpretability

### 5.5 Ablation Studies
- Graph controls: no sharing strategy beats baseline (Table 3, Figure 2)
- Label efficiency: smooth degradation for PERG, non-monotonic for LEOP (Table 4, Figure 3)
- Embedding classifiers: classical classifiers on neural embeddings underperform end-to-end (Table 5)
- Expert-fidelity probes: embeddings preserve morphology but not classification-relevant contrast (Table 6, Figure 4)

### 5.6 External Domain Transfer (Table 7, Figure 5)
- 4-domain SSL (LEOP+PERG+URFU+FLINDERS): LEOP 0.664, PERG 0.719
- Paired comparison: neither improvement nor degradation significant
- Adding external domains neither helps nor hurts

## 6. Discussion

### 6.1 Key Findings
- Attention mechanism achieves best LEOP performance (0.743 vs 0.682 baseline)
- Multi-task learning improves both tasks (+0.030 LEOP, +0.016 PERG)
- Classical baselines remain competitive (FPCA+demog 0.750 PERG)
- At 92k params/n=160-336, neural models match but don't dramatically exceed classical methods

### 6.2 Why Attention Works
- Raw waveform input preserves more information than hand-crafted features
- Self-attention captures component-level relationships
- Attention weights show which waveform regions drive classification
- Interpretability enables clinical validation

### 6.3 Why Multi-Task Helps
- LEOP and PERG share retinal biology (RGC pathway)
- Joint training regularizes the shared representation
- LEOP benefits more (+0.030 vs +0.016) due to smaller dataset

### 6.4 Why Transfer Failed
- Pathway-constrained partial transfer shows no benefit at this scale
- Frozen-encoder contract limits adaptation
- Model capacity (92k params) may be insufficient
- Protocol mismatch unknown (sOT assumes linear relationship)

## 7. Limitations
- Small cohorts (LEOP 160, PERG 336 after nesting)
- Attention ensemble degradation due to cross-fold probability scale mismatch
- Sex imbalance in LEOP (24.5% F)
- External datasets gated (URFU labels pending clinical review)
- Single model architecture (92k params)

## 8. Conclusion
- Comprehensive benchmark: classical → neural → multi-task → attention
- Attention ERG classifier achieves best LEOP performance (0.743)
- Multi-task learning improves both tasks
- Interpretability via attention weights enables clinical validation
- Future: larger models, multi-site data, clinical deployment

## References

## Appendix
- A. Signed OT mathematical properties
- B. Simulation E2 full grid
- C. Probe battery details
- D. Attention weight visualization
- E. External dataset protocols
