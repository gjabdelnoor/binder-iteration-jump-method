"""Seed-42 TNF binder study. All GPU calls must run inside a Slurm allocation.

Generated rows are unassayed presumed negatives. They are never included in the
primary experimental test metrics. This is a target-specific sequence classifier,
not an affinity, cross-reactivity, or pH-selectivity predictor.
"""
# %% Imports and reproducibility
from __future__ import annotations
import argparse, copy, gc, hashlib, json, math, os, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

AA = "ACDEFGHIKLMNPQRSTVWY"
DEFAULTS = dict(seed=42, model="biohub/ESMC-300M", folds=3, epochs=8,
                steps_per_epoch=16, batch_size=8, rank=8, alpha=16,
                lora_lr=1e-4, head_lr=3e-4, weight_decay=0.01,
                dropout=0.1, generated_count=12000, generation_batch=32,
                generation_rounds=2, mask_fraction=0.35, temperature=1.0,
                pool_positive_fraction=0.001, synthetic_loss_weight=0.25,
                donor_targets=["EGFR", "RBX1", "TrkA", "15-PGDH"],
                arms=["lora_assay_only", "lora_presumed_negative_pool"])

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")

def require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work must be submitted with sbatch, not run on the laptop.")
    if not torch.cuda.is_available(): raise RuntimeError("Slurm job has no CUDA device.")
    if "3060" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This registered study is restricted to the RTX 3060.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.85)

# %% Data curation and the locked 70/15/15 split

def prepare_data(root: Path, cfg: dict):
    raw_path = root / "input/design_summary.csv"
    raw = pd.read_csv(raw_path)
    raw["sequence"] = raw.sequence.str.upper().str.strip()
    tnf = raw.loc[raw.target.eq("TNFa")].copy()
    final_positive = truth(tnf.binder_final)
    # Do not turn non-expression or missing assays into binding negatives.
    qualified_negative = (~final_positive & tnf.adaptyv_binding.eq("non_binder")
                          & tnf.adaptyv_expression.isin(["medium", "high", "low"]))
    eligible = final_positive | qualified_negative
    excluded = tnf.loc[~eligible, ["uuid", "adaptyv_binding", "adaptyv_expression"]]
    real = tnf.loc[eligible].copy()
    real["label"] = truth(real.binder_final).astype(int)
    real["label_source"] = "adjudicated binder_final; expressed-negative QC"
    real["label_status"] = "experimental"
    assert real.sequence.map(lambda s: 1 <= len(s) <= 512 and set(s) <= set(AA)).all()
    conflicts = real.groupby("sequence").label.nunique()
    conflict_sequences = set(conflicts[conflicts > 1].index)
    real = real.loc[~real.sequence.isin(conflict_sequences)].sort_values("uuid")
    before_dedup = len(real)
    real = real.drop_duplicates("sequence").reset_index(drop=True)
    ids = np.arange(len(real)); y = real.label.to_numpy()
    train, remainder = train_test_split(ids, test_size=0.30, random_state=cfg["seed"], stratify=y)
    val, test = train_test_split(remainder, test_size=0.50, random_state=cfg["seed"], stratify=y[remainder])
    split = {"train": sorted(train.tolist()), "validation": sorted(val.tolist()), "test": sorted(test.tolist())}
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert set(train)|set(val)|set(test) == set(ids)
    real["split"] = "train"
    real.loc[val, "split"] = "validation"; real.loc[test, "split"] = "test"
    donors = raw.loc[raw.target.isin(cfg["donor_targets"]) & truth(raw.binder_final)].copy()
    donors = donors.loc[~donors.sequence.isin(set(real.sequence))].drop_duplicates("sequence")
    donors = donors.sort_values(["target", "uuid"]).reset_index(drop=True)
    assert len(donors) > 0
    (root / "data").mkdir(exist_ok=True)
    real.to_csv(root / "data/experimental.csv", index=False)
    donors.to_csv(root / "data/donors.csv", index=False)
    excluded.to_csv(root / "data/excluded.csv", index=False)
    audit = dict(source_url="https://huggingface.co/datasets/Anthropic/claude-protein-binder-design",
                 source_file="data/tables/design_summary.parquet", local_csv_sha256=sha256(raw_path),
                 source_target_rows=len(tnf), eligible=len(real), positives=int(y.sum()),
                 negatives=int((y==0).sum()), excluded_assay_qc=len(excluded),
                 conflicting_sequences=len(conflict_sequences), exact_duplicates_removed=before_dedup-len(real),
                 donor_counts=donors.target.value_counts().to_dict(),
                 split_counts={k: dict(n=len(v), positives=int(y[v].sum()), negatives=int((y[v]==0).sum())) for k,v in split.items()},
                 target="human TNF-alpha", no_mouse_or_pH_supervision=True,
                 random_split_warning="Random sequence split, NOT a scaffold-held-out benchmark.")
    write_json(root / "results/data_audit.json", audit)
    write_json(root / "data/split_indices.json", split)
    print("DATA_AUDIT", json.dumps(audit), flush=True)
    return real, donors, split

# %% Tokenization, padding and masked mean pooling

def tokenize(tokenizer, sequences, device="cuda"):
    batch = tokenizer(list(sequences), padding=True, return_tensors="pt")
    return {k:v.to(device) for k,v in batch.items() if k in ("input_ids", "attention_mask")}

def amino_ids(tokenizer, device="cuda"):
    return torch.tensor([tokenizer.convert_tokens_to_ids(a) for a in AA], device=device)

def mean_pool(hidden, input_ids, valid_ids):
    # [batch, length, channels] -> [batch, channels]; BOS/EOS/PAD do not contribute.
    mask = torch.isin(input_ids, valid_ids)
    if not bool(mask.any(dim=1).all()): raise ValueError("A sequence has no amino-acid tokens.")
    return (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

def load_mlm(root: Path, cfg: dict):
    revision = (root / "results/model_revision.txt").read_text().strip()
    model = EsmcForMaskedLM.from_pretrained(cfg["model"], revision=revision,
                    device="cuda", attn_implementation="sdpa")
    model = model.to(dtype=torch.bfloat16).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, EsmcTokenizer()

# %% ESMC generation: off-target-derived, unassayed sequence decoys

def generate_decoys(root: Path, cfg: dict, model, tokenizer, real, donors):
    path = root / "data/generated_presumed_negatives.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if len(existing) == cfg["generated_count"]: return existing
        raise RuntimeError("Partial decoy file exists; do not silently mix two generation runs.")
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    valid = amino_ids(tokenizer)
    valid_cpu = set(valid.cpu().tolist())
    original_seen = set(real.sequence) | set(donors.sequence)
    seen = set(original_seen); rows=[]; batches=0
    target_count = cfg["generated_count"]
    start=time.monotonic()
    while len(rows) < target_count:
        if batches > math.ceil(target_count/cfg["generation_batch"])*8:
            raise RuntimeError("Generation uniqueness/quality checks rejected too many sequences.")
        parents = donors.iloc[rng.integers(0, len(donors), size=cfg["generation_batch"])]
        x = tokenize(tokenizer, parents.sequence)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(cfg["generation_rounds"]):
                content_mask = torch.isin(x["input_ids"], valid)
                update = (torch.rand(x["input_ids"].shape, device="cuda") < cfg["mask_fraction"]) & content_mask
                masked = x["input_ids"].clone(); masked[update] = tokenizer.mask_token_id
                out = model(input_ids=masked, attention_mask=x.get("attention_mask"))
                logits = out.logits.index_select(-1, valid).float() / cfg["temperature"]
                draws = torch.multinomial(logits.softmax(-1).reshape(-1,len(AA)), 1)
                sampled = valid[draws.reshape(masked.shape)]
                x["input_ids"][update] = sampled[update]
        tokens = x["input_ids"].detach().cpu().tolist()
        for parent, ts in zip(parents.itertuples(), tokens):
            seq = "".join(tokenizer.convert_ids_to_tokens(i) for i in ts if i in valid_cpu)
            freq=np.array(list(Counter(seq).values()), dtype=float)/len(seq)
            entropy=float(-(freq*np.log2(freq)).sum())
            edits=sum(a!=b for a,b in zip(seq,parent.sequence))
            if seq in seen or len(seq)!=len(parent.sequence) or edits<10 or entropy<2.3: continue
            seen.add(seq)
            rows.append(dict(uuid="decoy-"+hashlib.sha256(seq.encode()).hexdigest()[:20], sequence=seq,
                  label=0, label_status="presumed_negative_unassayed", donor_uuid=parent.uuid,
                  donor_target=parent.target, donor_edits=edits, generation_batch=batches,
                  structure_validated=False, target_conditioned_generation=False))
            if len(rows)==target_count: break
        batches += 1
        if batches%20==0: print("GENERATION",len(rows),"of",target_count,"seconds",round(time.monotonic()-start,1),flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(path,index=False)
    write_json(root/"results/generation.json",dict(n=len(frame),model=cfg["model"],seed=cfg["seed"],
        donor_targets=frame.donor_target.value_counts().to_dict(),rounds=cfg["generation_rounds"],
        mask_fraction=cfg["mask_fraction"],temperature=cfg["temperature"],seconds=time.monotonic()-start,
        sha256=sha256(path),claimed_experimental_negatives=False,claimed_off_target_binding=False,
        note="Masked sequence variants of experimentally positive off-target donors, not structure-conditioned designs."))
    return frame

# %% Frozen representations and a simple baseline

@torch.inference_mode()
def embed(model, tokenizer, sequences, batch_size=32):
    model.eval(); result=[]; valid=amino_ids(tokenizer)
    for start in range(0,len(sequences),batch_size):
        x=tokenize(tokenizer,sequences[start:start+batch_size])
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model.esmc(**x)
        result.append(mean_pool(out.last_hidden_state,x["input_ids"],valid).cpu().numpy())
    return np.concatenate(result).astype(np.float32)

def fit_baseline(features, y, train):
    scaler=StandardScaler().fit(features[train])
    clf=LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,random_state=42)
    clf.fit(scaler.transform(features[train]),y[train])
    return scaler,clf

# %% Evaluation: average precision and top-decile enrichment

def metrics(y, scores):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    if set(y)!={0,1}: raise ValueError("Test set must contain positives AND negatives.")
    k=max(1,math.ceil(len(y)*0.10)); order=np.argsort(-scores,kind="stable")[:k]
    prevalence=float(y.mean()); precision=float(y[order].mean())
    ap=float(average_precision_score(y,scores))
    return dict(n=len(y),positives=int(y.sum()),prevalence=prevalence,
                average_precision=ap,roc_auc=float(roc_auc_score(y,scores)),
                enrichment_at_10_percent=precision/prevalence,top_k=k,
                top_k_hits=int(y[order].sum()),precision_at_k=precision,
                average_precision_lift=ap/prevalence)

def identity(a,b):
    # Normalized Levenshtein identity proxy, explicitly not a structural metric.
    if len(a)>len(b): a,b=b,a
    prev=list(range(len(a)+1))
    for j,cb in enumerate(b,1):
        current=[j]
        for i,ca in enumerate(a,1):
            current.append(min(current[-1]+1,prev[i]+1,prev[i-1]+(ca!=cb)))
        prev=current
    return 1-prev[-1]/max(len(a),len(b))

def overlap_audit(real, train, test):
    highest=[max(identity(real.sequence.iloc[i],real.sequence.iloc[j]) for j in train) for i in test]
    return dict(max_train_identity=float(max(highest)),fraction_test_at_least_90pct=float(np.mean(np.array(highest)>=.9)),
                fraction_test_at_least_95pct=float(np.mean(np.array(highest)>=.95)),
                per_test_max_identity=[float(x) for x in highest])

# %% LoRA from scratch: freeze W, train A and B

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__(); self.base=base; self.scale=alpha/rank
        for p in base.parameters(): p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        self.dropout=nn.Dropout(dropout); self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5)); nn.init.zeros_(self.B)
    def forward(self,x):
        low=F.linear(self.dropout(x),self.A.to(x.dtype))
        update=F.linear(low,self.B.to(x.dtype))
        return self.base(x)+self.scale*update

def attach_lora(encoder,cfg):
    paths=[name for name,mod in encoder.named_modules()
           if isinstance(mod,nn.Linear) and ("attn" in name or "attention" in name)]
    if len(paths)<20: raise RuntimeError(f"Unexpected ESMC attention layout: {paths}")
    for path in paths:
        parent_name,child=path.rsplit(".",1); parent=encoder.get_submodule(parent_name)
        setattr(parent,child,LoRALinear(getattr(parent,child),cfg["rank"],cfg["alpha"],cfg["dropout"]))
    return paths

class BinderClassifier(nn.Module):
    def __init__(self,encoder,tokenizer):
        super().__init__(); self.encoder=encoder
        self.register_buffer("valid_ids",amino_ids(tokenizer))
        width=encoder.embed.embedding_dim
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Dropout(0.1),nn.Linear(width,1)).to("cuda")
    def forward(self,inputs):
        out=self.encoder(**inputs)
        features=mean_pool(out.last_hidden_state,inputs["input_ids"],self.valid_ids)
        return self.head(features).squeeze(-1).float()

@torch.inference_mode()
def predict(model,tokenizer,sequences,batch_size=16):
    model.eval(); outputs=[]
    for start in range(0,len(sequences),batch_size):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(tokenize(tokenizer,sequences[start:start+batch_size]))
        outputs.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(outputs)

# %% Minibatches, the loss, autograd and optimizer updates

def train_one(root,cfg,encoder,tokenizer,real,decoys,features,decoy_features,train,val,test,arm,tag):
    seed_everything(cfg["seed"])
    for mod in encoder.modules():
        if isinstance(mod,LoRALinear): mod.reset_parameters()
    model=BinderClassifier(encoder,tokenizer)
    y=real.label.to_numpy(); rng=np.random.default_rng(cfg["seed"])
    pos=np.array([i for i in train if y[i]==1]); neg=np.array([i for i in train if y[i]==0])
    baseline_scaler,baseline=fit_baseline(features,y,train)
    presumed=arm=="lora_presumed_negative_pool"
    needed=max(0,round(len(pos)/cfg["pool_positive_fraction"])-len(train)) if presumed else 0
    if needed>len(decoys): raise RuntimeError("Decoy pool is too small for requested prevalence.")
    if presumed:
        # Mine model-relative hard examples with TRAIN labels only, never val/test labels.
        margins=baseline.decision_function(baseline_scaler.transform(decoy_features))
        ranked=np.argsort(-margins,kind="stable"); pool=ranked[:needed]
        hard=pool[:max(1,len(pool)//10)]
    else: pool=np.array([],dtype=int); hard=pool
    lora_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    head_params=list(model.head.parameters())
    optimizer=torch.optim.AdamW([{"params":lora_params,"lr":cfg["lora_lr"]},
                                {"params":head_params,"lr":cfg["head_lr"]}],weight_decay=cfg["weight_decay"])
    assert all(not p.requires_grad for n,p in encoder.named_parameters() if not n.endswith((".A",".B")))
    frozen_name,frozen=next((n,p) for n,p in encoder.named_parameters() if not p.requires_grad)
    frozen_copy=frozen.detach().clone()
    best_key=(-math.inf,-math.inf); best=None; history=[]; gradient_checks=None
    out_dir=root/"checkpoints"/tag/arm; out_dir.mkdir(parents=True,exist_ok=True)
    write_json(out_dir/"training_manifest.json",dict(train_uuids=real.uuid.iloc[train].tolist(),
       validation_uuids=real.uuid.iloc[val].tolist(),test_uuids=real.uuid.iloc[test].tolist(),
       presumed_negative_count=needed,pool_positive_prevalence=float(len(pos)/(len(train)+needed)),
       effective_minibatch_positive_fraction=.25 if presumed else .5,
       synthetic_loss_weight=cfg["synthetic_loss_weight"] if presumed else 0,
       hard_mining="Top decile of train-only frozen-head scores, 50% hard / 50% uniform pool",
       note="Reservoir prevalence is NOT the class balance of weighted optimization steps.",
       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    for epoch in range(cfg["epochs"]):
        model.train(); losses=[]
        for step in range(cfg["steps_per_epoch"]):
            if presumed:
                rid=np.r_[rng.choice(pos,2),rng.choice(neg,2)]
                did=np.r_[rng.choice(hard,2),rng.choice(pool,2)]
                seq=real.sequence.iloc[rid].tolist()+decoys.sequence.iloc[did].tolist()
                labels=np.r_[y[rid],np.zeros(4)]; weights=np.r_[np.ones(4),np.repeat(cfg["synthetic_loss_weight"],4)]
            else:
                rid=np.r_[rng.choice(pos,4),rng.choice(neg,4)]
                seq=real.sequence.iloc[rid].tolist(); labels=y[rid]; weights=np.ones(8)
            inputs=tokenize(tokenizer,seq)
            labels=torch.tensor(labels,device="cuda",dtype=torch.float32)
            weights=torch.tensor(weights,device="cuda",dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=model(inputs)
            per_row=F.binary_cross_entropy_with_logits(logits,labels,reduction="none")
            loss=(per_row*weights).sum()/weights.sum()
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss.")
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(lora_params+head_params,1.0)
            if not torch.isfinite(grad_norm): raise FloatingPointError("Non-finite gradients.")
            if gradient_checks is None:
                gradient_checks=dict(lora_nonzero_gradient=any(p.grad is not None and bool((p.grad!=0).any()) for p in lora_params),
                                     head_nonzero_gradient=any(p.grad is not None and bool((p.grad!=0).any()) for p in head_params))
                assert all(gradient_checks.values())
            optimizer.step(); losses.append(float(loss.detach()))
        v=predict(model,tokenizer,real.sequence.iloc[val].tolist())
        ap=float(average_precision_score(y[val],v)); bce=float(log_loss(y[val],np.clip(v,1e-7,1-1e-7),labels=[0,1]))
        row=dict(epoch=epoch+1,training_loss=float(np.mean(losses)),validation_ap=ap,validation_bce=bce)
        history.append(row)
        key=(ap,-bce)
        if key>best_key:
            best_key=key
            best={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            best_epoch=epoch+1
        print("EPOCH",tag,arm,json.dumps(row),flush=True)
    assert best is not None
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in best: p.copy_(best[name].to(p.device))
    assert torch.equal(frozen,frozen_copy),"A frozen backbone parameter changed."
    gradient_checks["frozen_parameter_unchanged"]=True
    gradient_checks["nonzero_lora_B_after_training"]=any(bool((m.B!=0).any()) for m in encoder.modules() if isinstance(m,LoRALinear))
    assert gradient_checks["nonzero_lora_B_after_training"]
    test_scores=predict(model,tokenizer,real.sequence.iloc[test].tolist())
    result=metrics(y[test],test_scores)
    result.update(arm=arm,split=tag,best_epoch=best_epoch,train_n=len(train),train_positives=int(y[train].sum()),
                  validation_n=len(val),validation_positives=int(y[val].sum()),
                  presumed_negative_pool=needed,pool_positive_prevalence=float(len(pos)/(len(train)+needed)),
                  gpu_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  gradient_checks=gradient_checks)
    torch.save(dict(trainable_state_dict=best,config=cfg,model_revision=(root/"results/model_revision.txt").read_text().strip(),
                    target="human TNF-alpha",calibrated=False),out_dir/"adapter_and_head.pt")
    write_json(out_dir/"history.json",history)
    write_json(out_dir/"metrics.json",result)
    pred=pd.DataFrame(dict(uuid=real.uuid.iloc[test].to_numpy(),label=y[test],score=test_scores,arm=arm,split=tag))
    pred.to_csv(out_dir/"test_predictions.csv",index=False)
    print("TEST_RESULT",json.dumps(result),flush=True)
    del optimizer,model,frozen_copy; gc.collect(); torch.cuda.empty_cache()
    return result,pred

# %% Locked holdout plus three-fold CV on the development portion only

def run(root: Path,cfg: dict):
    require_slurm_gpu(); start=time.monotonic(); seed_everything(cfg["seed"])
    (root/"results").mkdir(exist_ok=True)
    write_json(root/"results/protocol.json",dict(**cfg,
       validation_design="70/15/15 seed-42 stratified random holdout; 3-fold outer CV on train+validation only, 20% inner validation",
       checkpoint_selection="maximum inner-validation average precision, then minimum BCE",
       primary_test="Experimental rows only. Synthetic rows NEVER enter experimental test metrics.",
       enrichment_definition="precision among top ceil(0.10*N) divided by experimental test prevalence",
       no_scaffold_generalization_claim=True))
    real,donors,split=prepare_data(root,cfg)
    model,tokenizer=load_mlm(root,cfg)
    decoys=generate_decoys(root,cfg,model,tokenizer,real,donors)
    print("EMBEDDING_BASELINE",flush=True)
    features=embed(model,tokenizer,real.sequence.tolist())
    decoy_features=embed(model,tokenizer,decoys.sequence.tolist())
    np.savez_compressed(root/"data/frozen_features.npz",experimental=features,decoys=decoy_features)
    paths=attach_lora(model.esmc,cfg)
    write_json(root/"results/lora_targets.json",paths)
    y=real.label.to_numpy(); dev=np.array(sorted(split["train"]+split["validation"]))
    heldout=set(split["test"]); partitions=[]
    for fold,(outer_train,outer_test) in enumerate(StratifiedKFold(n_splits=cfg["folds"],shuffle=True,random_state=42).split(dev,y[dev]),1):
        outer_train,outer_test=dev[outer_train],dev[outer_test]
        inner_train,inner_val=train_test_split(outer_train,test_size=.20,random_state=42+fold,stratify=y[outer_train])
        assert not (heldout & (set(inner_train)|set(inner_val)|set(outer_test)))
        partitions.append((f"cv{fold}",inner_train,inner_val,outer_test))
    partitions.append(("locked_holdout",np.array(split["train"]),np.array(split["validation"]),np.array(split["test"])))
    all_results=[]; predictions=[]; overlaps={}
    for tag,train,val,test in partitions:
        overlaps[tag]=overlap_audit(real,train,test)
        scaler,clf=fit_baseline(features,y,train)
        scores=clf.predict_proba(scaler.transform(features[test]))[:,1]
        baseline_result=metrics(y[test],scores); baseline_result.update(arm="frozen_esmc_logistic",split=tag)
        all_results.append(baseline_result)
        predictions.append(pd.DataFrame(dict(uuid=real.uuid.iloc[test].to_numpy(),label=y[test],score=scores,arm="frozen_esmc_logistic",split=tag)))
        for arm in cfg["arms"]:
            torch.cuda.reset_peak_memory_stats()
            result,pred=train_one(root,cfg,model.esmc,tokenizer,real,decoys,features,decoy_features,train,val,test,arm,tag)
            all_results.append(result); predictions.append(pred)
            write_json(root/"results/partial_results.json",all_results)
    p=pd.concat(predictions,ignore_index=True); p.to_csv(root/"results/predictions.csv",index=False)
    rows=[]
    for arm in ["frozen_esmc_logistic"]+cfg["arms"]:
        cv=[r for r in all_results if r["arm"]==arm and r["split"].startswith("cv")]
        ap=np.array([r["average_precision"] for r in cv]); ef=np.array([r["enrichment_at_10_percent"] for r in cv])
        pooled=p.loc[p.arm.eq(arm)&p.split.str.startswith("cv")]
        assert pooled.uuid.nunique()==len(dev) and len(pooled)==len(dev)
        row=dict(arm=arm,folds=len(cv),cv_mean_test_ap=float(ap.mean()),cv_sd_test_ap=float(ap.std(ddof=1)),
                 cv_mean_enrichment10=float(ef.mean()),cv_sd_enrichment10=float(ef.std(ddof=1)),
                 pooled_oof_ap=float(average_precision_score(pooled.label,pooled.score)),
                 locked_holdout=next(r for r in all_results if r["arm"]==arm and r["split"]=="locked_holdout"))
        rows.append(row)
    write_json(root/"results/sequence_overlap_audit.json",overlaps)
    write_json(root/"results/fold_metrics.json",all_results)
    summary=dict(status="completed",slurm_job_id=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(0),
                 torch_version=torch.__version__,seconds=time.monotonic()-start,rows=rows,
                 limitations=["Few experimental positives; fold-to-fold uncertainty is large.",
                  "Random splits can contain related scaffolds; see overlap audit.",
                  "Generated sequences are unassayed presumed negatives, not confirmed hard negatives.",
                  "0.1% is reservoir class prevalence, not TPR or effective minibatch prevalence.",
                  "No prediction of affinity, mouse binding or pH selectivity.",
                  "Pooled OOF scores from independently trained models need not be equally calibrated."])
    write_json(root/"results/summary.json",summary)
    print("STUDY_COMPLETED",json.dumps(summary),flush=True)

# %% Command-line entry point (not executed by the tutorial notebook)
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(); run(args.root.resolve(),DEFAULTS.copy())
