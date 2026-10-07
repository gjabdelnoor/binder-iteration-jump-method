"""Materialize the pinned public dataset for a fresh reproduction; no GPU work."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
REPO='Anthropic/claude-protein-binder-design'
REVISION='9e1b81696da46835e9e9cde9a3da976e0abc92ab'
EXPECTED_SHA256='3e796869b8dc0c90d7ab0e12daf25308c0cf272d04c6a5e27281f903ecd18744'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); destination=args.root/'input/design_summary.csv'
    if destination.exists():raise FileExistsError('Existing input is immutable; choose a fresh study root.')
    path=Path(hf_hub_download(REPO,'data/tables/design_summary.parquet',repo_type='dataset',revision=REVISION))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256:raise ValueError('Pinned parquet checksum mismatch.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pd.read_parquet(path).to_csv(destination,index=False)
    print(json.dumps({'dataset':REPO,'revision':REVISION,'parquet_sha256':digest,'output':str(destination)},indent=2))
if __name__=='__main__':main()
