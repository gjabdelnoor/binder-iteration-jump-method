"""Build and execute a self-contained tutorial from the EXACT study source/results."""
from __future__ import annotations
import json, os, sys, re, hashlib, argparse
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def build(root: Path):
    summary=json.loads((root/'results/summary.json').read_text())
    audit=json.loads((root/'results/data_audit.json').read_text())
    protocol=json.loads((root/'results/protocol.json').read_text())
    generation=json.loads((root/'results/generation.json').read_text())
    overlap=json.loads((root/'results/sequence_overlap_audit.json').read_text())
    import pandas as pd
    predictions=pd.read_csv(root/'results/predictions.csv').to_dict('records')
    histories={arm:json.loads((root/'checkpoints/locked_holdout'/arm/'history.json').read_text()) for arm in protocol['arms']}
    notebook=nbf.v4.new_notebook()
    notebook.metadata.update(kernelspec={'display_name':'Python 3','language':'python','name':'python3'},
                             language_info={'name':'python','version':sys.version.split()[0]})
    cells=[]
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# TNF binder prediction with ESMC: a research notebook and PyTorch tutorial

**Question:** Does LoRA adaptation improve experimental binder enrichment, and does an extreme presumed-negative pool help?

This notebook contains the actual data-curation, generation, fine-tuning, cross-validation, and evaluation code used in the run, followed by the recorded results. Tutorial examples execute on the CPU; full model generation/training belongs in an explicit Slurm GPU allocation.

The target is **human TNF-alpha**. A sequence classifier trained on this task is not a general binder detector, an affinity predictor, or a mouse/pH classifier. The only experimental labels are from the published dataset. Generated decoys are **unassayed presumed negatives**, not measured hard negatives.

**Read in order:** PyTorch tensors and autograd → data/splits → masked-language-model decoys → LoRA → training → held-out evidence.''')
    md('''## 1. Tensors, shapes, devices, and gradients
A tensor is an array with a dtype and a device. `requires_grad=True` asks PyTorch to track operations so it can differentiate a loss. `.backward()` accumulates gradients; an optimizer changes parameters; `zero_grad()` prevents gradients from unintentionally accumulating between independent steps.

For proteins, token IDs have shape `[batch, length]`; ESMC representations have shape `[batch, length, channels]`; a binary head returns **one logit per sequence**.''')
    code('''%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
torch.manual_seed(42)
x = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
y = torch.tensor([1.0, 0.0])
head = nn.Linear(2, 1)  # weight and bias are trainable Parameters.
optimizer = torch.optim.AdamW(head.parameters(), lr=0.05)
losses = []
for step in range(30):
    optimizer.zero_grad(set_to_none=True)
    logits = head(x).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    if step == 0:
        print('input:', x.shape, x.dtype, x.device)
        print('weight gradient:', head.weight.grad)
    optimizer.step()
    losses.append(loss.item())
print('Toy training loss:', losses[0], '->', losses[-1])
print('Toy probabilities:', head(x).sigmoid().detach().flatten().tolist())
assert losses[-1] < losses[0]''')
    md('''### Logits, probabilities, and the meaning of 0.1%
`sigmoid(logit)` maps a score into (0, 1), but a sigmoid output is not automatically calibrated. `binary_cross_entropy_with_logits` combines sigmoid and binary cross entropy stably.

**Positive prevalence** is P/(P+N). **True-positive rate / recall** is TP/(TP+FN). They are different quantities. A 0.1%-positive pool has about 999 negatives per positive. An always-negative classifier then has 99.9% accuracy and zero recall. The study therefore measures average precision and top-decile enrichment, not accuracy.

The synthetic arm constructs a 0.1%-positive training reservoir but draws balanced, confidence-weighted minibatches. It does **not** claim that extreme imbalance itself forces generalization.''')
    code('''positive_fraction = 0.001
print('Always-negative accuracy:', 1-positive_fraction)
print('Always-negative recall:', 0.0)
from torch.utils.data import TensorDataset, DataLoader
toy_dataset = TensorDataset(x, y)
loader = DataLoader(toy_dataset, batch_size=2, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
for xb, yb in loader:
    print('DataLoader batch:', xb.shape, yb.shape)''')
    explanations={
      'Imports and reproducibility': 'The following cells are the exact implementation, not pseudocode. RNG seeds, source hashes, configuration, and split identities are retained. Random seeds make the stochastic protocol repeatable, but CUDA kernels and environment versions can still affect numerical results.',
      'Data curation and the locked 70/15/15 split': 'The source is filtered to TNF-alpha. Adjudicated positives are retained; negatives must have an explicit expressed nonbinder call. Missing assays and non-expression are not relabeled as binding failures. Exact duplicate sequences are removed before splitting; contradictory duplicates are excluded. The 15% test set is untouched by cross-validation.',
      'Tokenization, padding and masked mean pooling': 'Each amino acid becomes an integer. Padding allows variable-length proteins in one batch. Mean pooling must exclude BOS, EOS and padding tokens; otherwise sequence length and padding contaminate the representation.',
      'ESMC generation: off-target-derived, unassayed sequence decoys': 'ESMC supplies masked amino-acid probabilities. Starting sequences are experimentally positive binders of other targets (EGFR, RBX1, TrkA, and 15-PGDH). Two rounds of random masking/resampling create new sequences. These are not target-conditioned or structurally validated binders; changing a known binder does not preserve its binding phenotype. Only the training arm treats them as downweighted presumed negatives.',
      'Frozen representations and a simple baseline': 'The encoder is used under inference mode to cache embeddings. Standardization and logistic regression are fitted on training rows only. This baseline tests whether LoRA improves on information that was already available in frozen representations.',
      'Evaluation: average precision and top-decile enrichment': 'Average precision is the step-weighted precision-recall summary returned by sklearn, not trapezoidal integration. EF10 is precision among the top ceil(10%×N) predictions divided by the experimental test prevalence. AP/prevalence is a separate PR-lift quantity, not EF10. The sequence-overlap audit warns when random splits share very similar sequences.',
      'LoRA from scratch: freeze W, train A and B': 'A frozen dense layer computes xW^T. LoRA adds (alpha/r)×(xA^T)B^T. A starts random and B starts at zero, so the initial model exactly matches its frozen base. The actual ESMC runtime determines which attention nn.Linear modules are wrapped; the complete list is saved. This run adapts attention output projections and trains a LayerNorm-plus-linear sequence head.',
      'Minibatches, the loss, autograd and optimizer updates': 'Both arms use the same optimizer-step budget. Assay-only batches draw four positives and four confirmed negatives. Synthetic-arm batches draw two positives, two confirmed negatives and four presumed negatives, weighted 0.25 each. Hardness is assigned using a frozen head fitted only on that fold\'s training labels. Validation selects the checkpoint; test labels never select weights or thresholds. Gradient assertions check that adapters and head learn while the base remains frozen.',
      'Locked holdout plus three-fold CV on the development portion only': 'Three-fold outer CV runs inside the 85% development partition; each outer training fold reserves 20% for inner validation. Reported CV test metrics are genuine outer-fold experimental predictions. The separate original 70/15/15 model uses only its training and validation partitions, then evaluates its locked test set once per prespecified arm. No generated row enters either experimental evaluation.'}
    source=(root/'src/study.py').read_text()
    chunks=source.split('# %% ')[1:]
    for chunk in chunks:
        title,body=chunk.split('\n',1)
        if title.startswith('Command-line entry'):continue
        md('## '+title+'\n\n'+explanations.get(title,''))
        code(body.strip())
        if title.startswith('Tokenization'):
            code('''tok = EsmcTokenizer()
small = tokenize(tok, ['ACDEFGHIK', 'MKTAY'], device='cpu')
print('Token IDs:', small['input_ids'])
example_hidden = torch.ones(*small['input_ids'].shape, 3)
pooled = mean_pool(example_hidden, small['input_ids'], amino_ids(tok, 'cpu'))
print('Masked mean-pool shape:', pooled.shape)
assert pooled.shape == (2, 3)''')
        if title.startswith('LoRA from scratch'):
            code('''torch.manual_seed(42)
base = nn.Linear(4, 2)
layer = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
inputs = torch.randn(3, 4)
old_base = base.weight.detach().clone()
torch.testing.assert_close(layer(inputs), base(inputs))
opt = torch.optim.AdamW([layer.A, layer.B], lr=0.01)
opt.zero_grad(set_to_none=True)
loss = layer(inputs).square().mean()
loss.backward()
print('B gradient norm:', layer.B.grad.norm().item())
opt.step()
print('B update norm:', layer.B.norm().item())
print('Frozen base changed:', not torch.equal(old_base, base.weight))
assert torch.equal(old_base, base.weight)''')
    md('''## Slurm execution and waiting
The GPU bootstrap and study ran through Slurm on a single RTX 3060. No cloud credit was provisioned. These are the shell scripts used; the notebook displays them rather than silently submitting another expensive job when opened. The environment lock records the resolver's actual versions, which are more authoritative than an initial requested Torch version.''')
    for filename in ['bootstrap.sh','run.slurm']:
        text=(root/'src'/filename).read_text()
        code(filename.replace('.','_').upper()+' = '+repr(text)+'\nprint('+filename.replace('.','_').upper()+')')
    md('''Submit explicitly from a configured Slurm host using `sbatch --wait --parsable ... run.slurm`. `--wait` blocks the submitting shell until completion and returns the job exit status. The study itself refuses CUDA execution without `SLURM_JOB_ID`. Failed jobs must be inspected, not presented as completed experiments.''')
    md('''## Reloading the trained classifier
The following loader reconstructs the pinned base model, installs the same adapters, validates every trainable checkpoint key, and then freezes the model for inference. It requires Slurm and does not run automatically when this notebook opens. A separate GPU verification job reloads both final LoRA checkpoints and compares all locked-test scores with the original predictions.''')
    inference=(root/'src/score.py').read_text().split('if __name__==')[0]
    inference=inference.replace('from study import BinderClassifier, attach_lora, predict, require_slurm_gpu, AA','')
    code(inference)
    if (root/'results/checkpoint_reload.json').exists():
        code('RELOAD_CHECKS = '+repr(json.loads((root/'results/checkpoint_reload.json').read_text()))+'\nprint(RELOAD_CHECKS)')
    md('''## Recorded experimental evidence
The following snapshot is generated from completed job artifacts, not manually typed scores. It remains embedded so this notebook is readable independently of the remote filesystem.''')
    code('RUN_SUMMARY = '+repr(summary)+'\nDATA_AUDIT = '+repr(audit)+'\nGENERATION_AUDIT = '+repr(generation)+'\nOVERLAP_AUDIT = '+repr(overlap)+'\nPROTOCOL = '+repr(protocol))
    code('''display(pd.DataFrame(DATA_AUDIT['split_counts']).T)
rows = []
for result in RUN_SUMMARY['rows']:
    rows.append({'model':result['arm'], 'CV AP mean':result['cv_mean_test_ap'],
        'CV AP SD':result['cv_sd_test_ap'], 'CV EF10 mean':result['cv_mean_enrichment10'],
        'CV EF10 SD':result['cv_sd_enrichment10'],
        'locked test AP':result['locked_holdout']['average_precision'],
        'locked test EF10':result['locked_holdout']['enrichment_at_10_percent']})
display(pd.DataFrame(rows))
print('GPU:', RUN_SUMMARY['gpu'], 'Slurm job:', RUN_SUMMARY['slurm_job_id'])
print('Generated presumed negatives:', GENERATION_AUDIT['n'])''')
    code('PREDICTIONS = pd.DataFrame('+repr(predictions)+')\nHISTORIES = '+repr(histories))
    code('''from sklearn.metrics import precision_recall_curve
fig, ax = plt.subplots(figsize=(8, 5))
for name, frame in PREDICTIONS[PREDICTIONS['split'].str.startswith('cv')].groupby('arm'):
    precision, recall, _ = precision_recall_curve(frame.label, frame.score)
    ax.step(recall, precision, where='post', label=name)
ax.set(xlabel='Recall', ylabel='Precision', title='Experimental outer-fold predictions only')
ax.legend(fontsize=8)
plt.show()''')
    code('''fig, ax = plt.subplots(figsize=(8, 5))
for name, history in HISTORIES.items():
    h = pd.DataFrame(history)
    ax.plot(h.epoch, h.training_loss, label=name)
ax.set(xlabel='Epoch', ylabel='Weighted training BCE', title='Locked-split training histories')
ax.legend(fontsize=8)
plt.show()
# Different sampling/loss weighting means the two loss curves are not directly equivalent.''')
    code('''display(pd.DataFrame({k:{a:b for a,b in v.items() if a!='per_test_max_identity'}
                      for k,v in OVERLAP_AUDIT.items()}).T)
for result in RUN_SUMMARY['rows']:
    if 'gradient_checks' in result['locked_holdout']:
        print(result['arm'], result['locked_holdout']['gradient_checks'])''')
    md('''## Interpretation boundaries and next decision
A high random-split AP does not demonstrate generalization to a new scaffold. Inspect the overlap table, the tiny number of positives per test fold, and the assay-only baseline before deciding whether to put this model into a search loop. A generative search can exploit a classifier's errors.

The 0.1%-positive reservoir is a **training augmentation hypothesis**. Generated presumed negatives are neither measured TNF nonbinders nor proven binders of their donor targets. A favorable result would justify further evaluation, not validate the labels. The run does not predict human/mouse cross-reactivity, affinity, or pH dependence.

Adapters/checkpoints and sequence-level working data are kept locally; the PR publishes code, tutorial, aggregate results, and provenance, not a new biological hit claim.

### Sources
- [Anthropic binder dataset and assay documentation](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design)
- [ESM implementation and ESMC models](https://github.com/biohub/esm)
- [PyTorch binary cross entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Scikit-learn average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

All experimental claims in this notebook come from the source dataset or the accompanying completed Slurm artifacts.''')
    notebook.cells=cells
    out=root/'public';out.mkdir(exist_ok=True)
    path=out/'tnf_esmc_research_tutorial.ipynb'
    nbf.write(notebook,path)
    client=NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}})
    client.execute(env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MPLBACKEND':'Agg'})
    nbf.validate(notebook);nbf.write(notebook,path)
    write={'executed':True,'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
       'cells':len(notebook.cells),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/'results/notebook_validation.json').write_text(json.dumps(write,indent=2)+'\n')
    print('NOTEBOOK_VALIDATED',json.dumps(write),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    build(p.parse_args().root.resolve())
