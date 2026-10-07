"""Independent checkpoint reload + upstream dataset snapshot verification."""
import argparse,gc,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from huggingface_hub import HfApi,hf_hub_download
from score import load_predictor
from study import predict,fit_baseline,write_json,sha256,DEFAULTS,require_slurm_gpu

def main(root):
    require_slurm_gpu()
    real=pd.read_csv(root/'data/experimental.csv');split=json.loads((root/'data/split_indices.json').read_text())
    test=split['test'];checks={}
    for arm in DEFAULTS['arms']:
        directory=root/'checkpoints/locked_holdout'/arm
        model,tok=load_predictor(directory/'adapter_and_head.pt')
        got=predict(model,tok,real.sequence.iloc[test].tolist())
        expected=pd.read_csv(directory/'test_predictions.csv').set_index('uuid').loc[real.uuid.iloc[test],'score'].to_numpy()
        diff=float(np.max(np.abs(got-expected)))
        checks[arm]={'max_absolute_score_difference':diff,'matches_saved_predictions':bool(diff<1e-6),
                     'checkpoint_sha256':sha256(directory/'adapter_and_head.pt'),'n_test_predictions':len(test)}
        assert diff<1e-6,(arm,diff)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    features=np.load(root/'data/frozen_features.npz')['experimental']
    scaler,clf=fit_baseline(features,real.label.to_numpy(),split['train'])
    folder=root/'checkpoints/locked_holdout/frozen_esmc_logistic';folder.mkdir(exist_ok=True)
    np.savez(folder/'head.npz',mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_,
             model_revision=np.array((root/'results/model_revision.txt').read_text().strip()))
    expected=pd.read_csv(root/'results/predictions.csv')
    expected=expected.loc[expected.arm.eq('frozen_esmc_logistic')&expected['split'].eq('locked_holdout')]
    score=clf.predict_proba(scaler.transform(features[test]))[:,1]
    checks['frozen_esmc_logistic']={'max_absolute_score_difference':float(np.max(abs(score-expected.score.to_numpy()))),
                                   'saved_head_sha256':sha256(folder/'head.npz')}
    write_json(root/'results/checkpoint_reload.json',dict(job_id=os.environ['SLURM_JOB_ID'],checks=checks))
    print('CHECKPOINT_RELOAD',json.dumps(checks),flush=True)
    repo='Anthropic/claude-protein-binder-design';rev=HfApi().dataset_info(repo).sha
    path=Path(hf_hub_download(repo,'data/tables/design_summary.parquet',repo_type='dataset',revision=rev))
    downloaded=pd.read_parquet(path);local=pd.read_csv(root/'input/design_summary.csv')
    cols=['uuid','sequence','target','binder_final','adaptyv_expression','adaptyv_binding']
    used=set(DEFAULTS['donor_targets'])|{'TNFa'}
    def canonical(df):
        df=df.loc[df.target.isin(used),cols].sort_values('uuid').fillna('')
        return df.astype(str).apply(lambda c:c.str.lower()).to_dict('records')
    verified=canonical(downloaded)==canonical(local)
    result=dict(repo=repo,revision=rev,file='data/tables/design_summary.parquet',parquet_sha256=sha256(path),
                local_input_sha256=sha256(root/'input/design_summary.csv'),used_fields_match=verified,
                compared_columns=cols,compared_targets=sorted(used))
    write_json(root/'results/upstream_snapshot.json',result)
    assert verified,'Upstream used fields differ from the frozen local snapshot; retain original data and investigate.'
    print('UPSTREAM_SNAPSHOT_VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
