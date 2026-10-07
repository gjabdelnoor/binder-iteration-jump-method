"""Load a saved ESMC LoRA/head and score FASTA sequences inside Slurm."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from Bio import SeqIO
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from study import BinderClassifier, attach_lora, predict, require_slurm_gpu, AA

def load_predictor(checkpoint: Path):
    require_slurm_gpu()
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    cfg=saved["config"]
    base=EsmcForMaskedLM.from_pretrained(cfg["model"],revision=saved["model_revision"],
                device="cuda",attn_implementation="sdpa").to(dtype=torch.bfloat16).eval()
    for p in base.parameters():p.requires_grad_(False)
    tokenizer=EsmcTokenizer();attach_lora(base.esmc,cfg)
    model=BinderClassifier(base.esmc,tokenizer)
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    actual=set(saved["trainable_state_dict"])
    if expected!=actual:raise ValueError(f"Checkpoint architecture mismatch: missing={expected-actual}, extra={actual-expected}")
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in actual:p.copy_(saved["trainable_state_dict"][name].to(p.device))
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    return model,tokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--fasta",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("Refusing to overwrite an existing prediction file.")
    records=list(SeqIO.parse(args.fasta,"fasta"));seqs=[str(r.seq).upper() for r in records]
    if not records or any(not s or len(s)>512 or not set(s)<=set(AA) for s in seqs):
        raise ValueError("FASTA must contain canonical amino-acid sequences of length 1..512.")
    model,tokenizer=load_predictor(args.checkpoint)
    scores=predict(model,tokenizer,seqs)
    output={"target":"human TNF-alpha","calibrated_binding_probability":False,
            "predictions":[{"id":r.id,"binder_score":float(s)} for r,s in zip(records,scores)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"scored":len(records),"output":str(args.output)}))
if __name__=="__main__":main()
