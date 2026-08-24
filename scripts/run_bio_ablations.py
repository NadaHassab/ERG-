"""Ablations for bio domains: each new domain added alone to 6-domain baseline."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import stft, butter, filtfilt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_multidomain_fusion import MultidomainERGDataset, load_extra_features, bootstrap_auroc
from scripts.run_bio_extended import compute_stft_features, compute_frft_features, compute_op_features, precompute_bio_features, FS, N_SCALES, FRFT_ORDERS
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine
from pathway_erg.evaluation.metrics import roc_auc_score

SEEDS = [1001, 2002, 3003]
N_SCALES_LOCAL = N_SCALES

ABLATIONS = {
    "stft":  {"dims": {"stft": N_SCALES}, "label": "6+CWT+STFT"},
    "frft":  {"dims": {"frft": 12},       "label": "6+CWT+FrFT"},
    "op":    {"dims": {"op": 8},          "label": "6+CWT+OP"},
}

class AblationModel(nn.Module):
    def __init__(self, extra_keys, d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)
        self.extra_keys = extra_keys
        n_domains = 6 + len(extra_keys)
        self.signal_cnn = nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.signal_proj = nn.Linear(64,d_model)
        self.ot_mlp = nn.Sequential(nn.Linear(135,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,d_model))
        self.spectral_mlp = nn.Sequential(nn.Linear(10,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,d_model))
        self.vmd_mlp = nn.Sequential(nn.Linear(80,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,d_model))
        self.physical_mlp = nn.Sequential(nn.Linear(8,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,d_model))
        self.cwt_cnn = nn.Sequential(nn.Conv1d(N_SCALES_LOCAL,32,5,padding=2),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(32,32,3,padding=1),nn.BatchNorm1d(32),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.cwt_proj = nn.Linear(32,d_model)
        self.extra_mlps = nn.ModuleDict()
        for k in extra_keys:
            dim = {"stft": N_SCALES_LOCAL, "frft": 12, "op": 8}[k]
            self.extra_mlps[k] = nn.Sequential(nn.Linear(dim,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,d_model))
        self.gate = nn.Sequential(nn.Linear(d_model*n_domains, n_domains), nn.Softmax(dim=-1))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_scorer = nn.Linear(d_model,1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,64), nn.GELU(), nn.Linear(64,1))
        self.n_domains = n_domains

    def forward(self, signal, vmask, ot, spectral, vmd, physical, cwt, extra_feats, comp_mask=None):
        B,L = signal.shape[:2]
        sig_feat = self.signal_proj(self.signal_cnn(signal.reshape(B*L,1,-1)).squeeze(-1))
        ot_feat = self.ot_mlp(ot.reshape(B*L,-1))
        spec_feat = self.spectral_mlp(spectral.reshape(B*L,-1))
        vmd_feat = self.vmd_mlp(vmd.reshape(B*L,-1))
        phys_feat = self.physical_mlp(physical.reshape(B*L,-1))
        cwt_feat = self.cwt_proj(self.cwt_cnn(cwt.reshape(B*L,N_SCALES_LOCAL,-1)).squeeze(-1))
        feats = [sig_feat, ot_feat, spec_feat, vmd_feat, phys_feat, cwt_feat]
        for k in self.extra_keys:
            feats.append(self.extra_mlps[k](extra_feats[k].reshape(B*L,-1)))
        concat = torch.cat(feats, dim=-1)
        weights = self.gate(concat)
        fused = sum(weights[:,i:i+1]*feats[i] for i in range(self.n_domains))
        tokens = fused.reshape(B,L,-1)
        if comp_mask is None:
            comp_mask = torch.ones(B,L,dtype=torch.bool,device=tokens.device)
        tokens = self.transformer(tokens)
        scores = self.attn_scorer(tokens).squeeze(-1).masked_fill(~comp_mask, float("-inf"))
        w = F.softmax(scores, dim=-1).masked_fill(~comp_mask, 0.0)
        pooled = (tokens * w.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1), w

def collate_ablation(bags, dataset, scal_cache, bio_caches, extra_keys):
    B = len(bags); L = max(len(b.components) for b in bags)
    signal=np.zeros((B,L,1,128),dtype=np.float32); valid_mask=np.zeros((B,L,128),dtype=bool)
    ot=np.zeros((B,L,135),dtype=np.float32); physical=np.zeros((B,L,8),dtype=np.float32)
    spectral=np.zeros((B,L,10),dtype=np.float32); vmd=np.zeros((B,L,80),dtype=np.float32)
    cwt=np.zeros((B,L,N_SCALES_LOCAL,128),dtype=np.float32); comp_mask=np.zeros((B,L),dtype=bool)
    labels=np.full(B,np.nan,dtype=np.float64)
    extra = {k: np.zeros((B,L,{"stft":N_SCALES_LOCAL,"frft":12,"op":8}[k]),dtype=np.float32) for k in extra_keys}
    stft_c, frft_c, op_c = bio_caches
    cmap = {"stft": stft_c, "frft": frft_c, "op": op_c}
    for i,bag in enumerate(bags):
        labels[i]=bag.target_binary if bag.target_binary is not None else np.nan
        for j,comp in enumerate(bag.components):
            signal[i,j,0,:]=comp.signal; valid_mask[i,j,:]=comp.signal_mask
            ot[i,j,:]=comp.ot_vector; physical[i,j,:]=comp.physical
            spectral[i,j,:]=dataset.get_spectral(comp); vmd[i,j,:]=dataset.get_vmd(comp)
            cid=comp.global_component_id
            if cid in scal_cache: cwt[i,j,:,:]=scal_cache[cid]
            for k in extra_keys:
                if cid in cmap[k]: extra[k][i,j,:]=cmap[k][cid]
            comp_mask[i,j]=True
    out={"signal":torch.as_tensor(signal),"valid_mask":torch.as_tensor(valid_mask),"ot":torch.as_tensor(ot),"physical":torch.as_tensor(physical),"spectral":torch.as_tensor(spectral),"vmd":torch.as_tensor(vmd),"cwt":torch.as_tensor(cwt),"comp_mask":torch.as_tensor(comp_mask),"label":torch.as_tensor(labels)}
    for k in extra_keys: out[k]=torch.as_tensor(extra[k])
    return out

def eval_auc(model, dataset, scal_cache, bio_caches, bags, device, extra_keys):
    model.eval(); yt,yp=[],[]
    for bag in bags:
        if bag.target_binary is None: continue
        batch=collate_ablation([bag],dataset,scal_cache,bio_caches,extra_keys)
        with torch.no_grad():
            ef={k: batch[k].to(device) for k in extra_keys}
            logit,_=model(batch["signal"].to(device),batch["valid_mask"].to(device),batch["ot"].to(device),batch["spectral"].to(device),batch["vmd"].to(device),batch["physical"].to(device),batch["cwt"].to(device),ef,batch["comp_mask"].to(device))
        yt.append(bag.target_binary); yp.append(float(torch.sigmoid(logit[0]).item()))
    return 0.5 if len(yt)<2 or len(set(yt))<2 else float(roc_auc_score(np.array(yt),np.array(yp)))

def train_model(model, train_bags, val_bags, dataset, scal_cache, bio_caches, seed, device, extra_keys, lr=1e-4):
    model.to(device)
    sampler=BagSampler(train_bags,folds={b.outer_fold for b in train_bags},batch_size=8,seed=seed)
    labels=np.asarray([b.target_binary for b in sampler.bags],dtype=float)
    criterion=FoldWeightedBCE(positive_class_weight(labels))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    steps=max(1,len(sampler.bags)//8); total=200*steps; warm=5*steps
    sched=_WarmupCosine(optimizer,warmup_steps=warm,total_steps=total,min_frac=0.05)
    best=-1; best_state=None; patience=0
    model.train()
    for epoch in range(200):
        for step,idx in enumerate(sampler):
            if step>=steps: break
            batch=collate_ablation([sampler.bags[i] for i in idx],dataset,scal_cache,bio_caches,extra_keys)
            labels_b=batch["label"].to(device)
            ef={k: batch[k].to(device) for k in extra_keys}
            logits,_=model(batch["signal"].to(device),batch["valid_mask"].to(device),batch["ot"].to(device),batch["spectral"].to(device),batch["vmd"].to(device),batch["physical"].to(device),batch["cwt"].to(device),ef,batch["comp_mask"].to(device))
            loss=criterion(logits,labels_b)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); sched.step()
        val_auc=eval_auc(model,dataset,scal_cache,bio_caches,val_bags,device,extra_keys)
        if val_auc>best: best=val_auc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=25: break
    if best_state is not None: model.load_state_dict(best_state)
    return {"best_epoch":epoch,"best_val_auc":best}

def predict_model(model,dataset,scal_cache,bio_caches,bags,device,extra_keys):
    model.eval(); rows=[]
    for bag in bags:
        if bag.target_binary is None: continue
        batch=collate_ablation([bag],dataset,scal_cache,bio_caches,extra_keys)
        with torch.no_grad():
            ef={k: batch[k].to(device) for k in extra_keys}
            logit,_=model(batch["signal"].to(device),batch["valid_mask"].to(device),batch["ot"].to(device),batch["spectral"].to(device),batch["vmd"].to(device),batch["physical"].to(device),batch["cwt"].to(device),ef,batch["comp_mask"].to(device))
        rows.append({"unit_id":bag.unit_id,"subject_id":bag.subject_id,"target":int(bag.target_binary),"probability":float(torch.sigmoid(logit[0]).item())})
    return pd.DataFrame(rows)

def main():
    data_cfg=load_config(DataConfig,"configs/data/local.yaml")
    caches=LoadedCaches(data_cfg.artifact_root,fold_version="v1")
    DEVICE="cuda"
    spectral_vecs,vmd_vecs,spectral_names,vmd_names=load_extra_features(data_cfg.artifact_root)
    for ablation, extra_keys in [("stft",["stft"]),("frft",["frft"]),("op",["op"])]:
        OUT_DIR=Path(f"artifacts/results/bio_ablate_{ablation}_v1"); OUT_DIR.mkdir(parents=True,exist_ok=True)
        for task in ["LEOP","PERG"]:
            print(f"\n{'='*60}\n  {task} — Ablation {ABLATIONS[ablation]['label']}\n{'='*60}")
            bags=build_task_bags(caches,task,"primary_nine_step")
            ds=MultidomainERGDataset(bags,spectral_vecs,vmd_vecs,spectral_names,vmd_names)
            from scripts.run_cwt_erg import precompute_scalograms
            scal_cache=precompute_scalograms(bags,N_SCALES_LOCAL)
            bio_caches=precompute_bio_features(bags)
            print(f"  Caches: CWT {len(scal_cache)}, bio {len(bio_caches[0])}")
            for seed in SEEDS:
                print(f"\n  --- seed {seed} ---")
                for outer_fold in range(5):
                    run_dir=OUT_DIR/task.lower()/f"run-fold{outer_fold}-seed{seed}"
                    if (run_dir/"predictions.parquet").exists(): print(f"  fold {outer_fold}: EXISTS (skip)"); continue
                    train_bags,test_bags=outer_partition(bags,outer_fold)
                    model=AblationModel(extra_keys,seed=seed)
                    log=train_model(model,train_bags,test_bags,ds,scal_cache,bio_caches,seed,DEVICE,extra_keys)
                    pred=predict_model(model,ds,scal_cache,bio_caches,test_bags,DEVICE,extra_keys)
                    point,ci_lo,ci_hi=bootstrap_auroc(pred["target"].values,pred["probability"].values)
                    print(f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={len(pred)} best_epoch={log['best_epoch']}")
                    run_dir.mkdir(parents=True,exist_ok=True); pred.to_parquet(run_dir/"predictions.parquet",index=False)
        for task in ["LEOP","PERG"]:
            aucs=[]
            for fold in range(5):
                for seed in SEEDS:
                    p=OUT_DIR/task.lower()/f"run-fold{fold}-seed{seed}/predictions.parquet"
                    if p.exists():
                        df=pd.read_parquet(p); pt,_,_=bootstrap_auroc(df["target"].values,df["probability"].values); aucs.append(pt)
            if aucs:
                by_fold=[np.mean([aucs[i] for i in range(f,len(aucs),5)]) for f in range(5)]
                print(f"\n  {task} {ablation} per-fold mean: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")
        print(f"\nResults saved to {OUT_DIR}")

if __name__=="__main__": main()
