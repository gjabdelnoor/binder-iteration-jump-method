# TNF-alpha ESMC binder-classification study

An executed single-RTX-3060 experiment and a PyTorch tutorial. This is an independent classifier study, not a release or modification of the private binder-jump search engine.

**Start with [the executed notebook](tnf_esmc_research_tutorial.ipynb).** It contains 40 cells (23 executed code cells), the actual training implementation, beginner examples, held-out results, plots, and checkpoint-loading code. GPU training is never launched automatically by opening the notebook.

## Completed experiment

The study used the Anthropic `claude-protein-binder-design` release, restricted to **human TNF-alpha**. Of 150 TNF designs, 12 adjudicated binders and 129 expressed nonbinders were retained. Nine nonexpressing constructs were excluded rather than relabeled as binding negatives. There were no contradictory or exact-duplicate sequences after curation.

Seed 42 and stratified random splitting produced:

| Partition | Sequences | Positives | Negatives |
|---|---:|---:|---:|
| Training | 98 | 8 | 90 |
| Validation | 21 | 2 | 19 |
| Locked test | 22 | 2 | 20 |

The requested percentages were interpreted as approximately 70/15/15. Three-fold outer cross-validation was performed **only within the 119-row training+validation development portion**; each outer training fold had a separate 20% inner validation subset. The locked 22-row test set was excluded from all cross-validation fits. Checkpoint selection used inner-validation average precision, with validation BCE as a tiebreaker. No generated sequences entered the experimental validation or test metrics.

### Cross-fold experimental test results

Values are mean ± sample standard deviation across three outer test folds, not confidence intervals.

| Prespecified model | Average precision (AUPRC) | Top-decile enrichment (EF10) |
|---|---:|---:|
| Frozen ESMC + standardized logistic regression | **0.6222 ± 0.1193** | **6.0556 ± 0.9179×** |
| ESMC LoRA + head, assay labels only | 0.5618 ± 0.1318 | 5.2222 ± 2.3590× |
| ESMC LoRA + head, presumed-negative augmentation | 0.6140 ± 0.1658 | 5.2222 ± 2.3590× |

Average precision is `sklearn.metrics.average_precision_score`, not trapezoidal PR integration. EF10 is the precision among the top `ceil(0.10*N)` predictions divided by the experimental test-set positive prevalence. AP/prevalence is separately recorded as PR lift and must not be confused with EF10.

### Separate locked test

| Model | AP | EF10 | Hits among the top 3 |
|---|---:|---:|---:|
| Frozen ESMC | 0.8333 | 7.3333× | 2 |
| Assay-only LoRA | 0.8333 | 7.3333× | 2 |
| Presumed-negative LoRA | 0.5769 | 3.6667× | 1 |

**This run does not demonstrate a LoRA improvement over the frozen baseline.** Presumed-negative augmentation improved mean CV AP relative to assay-only LoRA but did not exceed the frozen baseline, and performed worse on the locked test. With only two positives in that test, neither a favorable nor unfavorable difference is a robust general conclusion.

## What 0.1% means here

ESMC generated **12,000 unique masked-sequence variants** of experimentally positive off-target donors: RBX1 (28 donors), TrkA (20), EGFR (10), and 15-PGDH (1). Two rounds masked/resampled 35% of residues at temperature 1. Candidates had at least ten edits, sequence entropy >=2.3, and no exact duplicates of retained experimental rows or original donors.

These are **unassayed presumed negatives**. They are NOT structure-conditioned BoltzGen designs, confirmed binders of their donor targets, or confirmed TNF nonbinders. Donor target difference alone does not establish post-mutation binding specificity. No generated sequence was folded or tested experimentally in this study.

For the locked-split augmentation model the training reservoir contained:

- 8 measured positives;
- 90 measured negatives;
- 7,902 generated presumed negatives.

Thus the reservoir positive prevalence was exactly **8 / 8,000 = 0.1%**. This is not true-positive rate/recall. It is also **not the effective optimizer minibatch balance**: augmented batches contained two experimental positives, two experimental negatives, and four presumed negatives; each presumed-negative loss received weight 0.25. The assay-only arm used four positives and four negatives. Both arms received the same number of optimizer steps but different proportions of experimental/synthetic examples. Therefore the comparison is a bundled augmentation/sampling ablation, not an isolated causal test of class imbalance.

A training-only frozen baseline assigned hardness scores. Half of the sampled presumed negatives came from the highest-scoring decile of the selected pool, and half were sampled uniformly. They are model-relative hard examples, not measured biological negatives. A perfectly literal 99.9%-negative, unweighted training objective was **not** run.

## Model and checks

- Base: `biohub/ESMC-300M`, pinned revision `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`.
- LoRA: rank 8, alpha 16, dropout 0.1 on the 30 attention **output projections**. The fused/custom QKV layers were not adapted. This is not full-model parameter fine-tuning.
- Head: masked mean pooling, LayerNorm, dropout, one linear binary logit.
- Trainable parameters: **463,681**. Frozen backbone parameters remained frozen.
- Eight epochs × 16 optimizer steps per fit; eight fits (two arms × three CV folds plus one locked split). Batch size eight. AdamW, LoRA LR 1e-4, head LR 3e-4, gradient clipping at 1.0, BF16.
- Training allocated peak reported by PyTorch: approximately 1.85 GiB; this excludes driver overhead and is not a measurement of all-stage GPU peak consumption.
- Eight unit tests passed. Both final LoRA checkpoints were loaded into fresh models and reproduced all locked-test predictions to <2e-8 maximum absolute score difference.
- The final notebook executed 23 code cells, contains two inline PNG plots, and has no error outputs.
- The used source fields for TNF and the donor targets match the pinned official dataset snapshot; see `results/upstream_snapshot.json`.

Slurm job 5 installed and smoke-tested the isolated environment. Job 6 completed training, evaluation, and initial notebook execution with exit code 0; its wall time was **5 minutes 41 seconds**, excluding environment setup. Job 7 independently reloaded checkpoints, verified the upstream snapshot, and regenerated the final notebook. All CUDA work used the single RTX 3060 via Slurm; there was no rented-cloud allocation and no deployment into the live binder search.

## Limitations

There are only 12 experimental positives. Outer test folds contain three or four positives and inner validation sets contain one positive. Split-dependent checkpoint selection is consequently noisy. Random splits do not prove generalization to new structural families. A normalized edit-distance audit found no train/test pair with >=90% identity, but that does not establish scaffold independence. Grouped/scaffold-held-out validation remains a separate requirement.

The score is target-specific and uncalibrated. It predicts none of affinity, human/mouse cross-reactivity, pH dependence, or competition selection. The generated decoys are not validated hard negatives. No performance metric here is a wet-lab result for newly generated proteins.

## Files

- `study.py`: full preparation, generation, baseline, LoRA, fitting, and evaluation code.
- `test_study.py`: small CPU tests for pooling, gradients, frozen weights, metrics, and the Slurm guard.
- `score.py`: load a saved adapter/head and score FASTA inputs inside Slurm.
- `verify_run.py`: independent checkpoint reload and official snapshot verification.
- `download_data.py`: fetch the pinned public parquet for a fresh run; refuses to overwrite input.
- `bootstrap.sh`, `run.slurm`, `verify.slurm`: exact operational scripts.
- `build_notebook.py`: creates and executes the tutorial from actual source and results.
- `results/`: aggregate metrics, public-dataset UUID predictions, protocol, source hashes, environment lock, and checks.

Raw working sequences, generated decoys, cached embeddings, environment binaries, and trained checkpoints are not published in this PR. They remain in the user's private study directories on the compute machine and laptop. No base model weights are redistributed.

## Reproduce on a Slurm GPU host

Use a **fresh** study root; never overwrite the recorded run. An executable `uv` must be available as `$STUDY_ROOT/uv`. The job scripts expect `src/`, `input/`, `results/`, and `logs/` below that root.

```bash
export STUDY_ROOT="$HOME/tnf_esmc_reproduction"
mkdir -p "$STUDY_ROOT"/{src,input,results,logs}
cp studies/tnf_esmc/{study.py,test_study.py,score.py,verify_run.py,download_data.py,build_notebook.py,bootstrap.sh,run.slurm,verify.slurm} "$STUDY_ROOT/src/"
cp "$(command -v uv)" "$STUDY_ROOT/uv"
cd "$STUDY_ROOT"
sbatch --wait --parsable --partition=gpu --gres=gpu:rtx_3060:1 \
  --cpus-per-task=4 --mem=12G --time=00:40:00 --job-name=tnf-esmc-setup \
  --output=logs/setup-%j.log --export=ALL,STUDY_ROOT="$STUDY_ROOT" src/bootstrap.sh
.venv/bin/python src/download_data.py --root "$STUDY_ROOT"
sbatch --wait --parsable --export=ALL,STUDY_ROOT="$STUDY_ROOT" src/run.slurm
sbatch --wait --parsable --export=ALL,STUDY_ROOT="$STUDY_ROOT" src/verify.slurm
```

The historical bootstrap initially requested an older Torch version; the ESM dependency resolver upgraded it. The **actual tested environment** was Python 3.12, Torch **2.11.0+cu130**, and ESM **3.4.1.post1**. Consult `results/environment.lock.txt` rather than assuming the first requested Torch version was retained. For a byte-for-byte environment recreation, use that lock and the matching CUDA wheel source. Kernel/platform differences can still change floating-point results.

The reference run used an existing frozen CSV; `download_data.py` exports the pinned parquet. CSV formatting hashes can differ even when the relevant source fields are identical; the independent verification checks the relevant fields, not just filenames.

### Score new sequences without retraining

From a private study root containing the saved checkpoint and environment:

```bash
sbatch --wait --partition=gpu --gres=gpu:rtx_3060:1 --cpus-per-task=4 --mem=8G \
  --time=00:10:00 --output=logs/inference-%j.log \
  --wrap="HF_HOME=\$HOME/.cache/huggingface-tnf-study .venv/bin/python src/score.py \
    --checkpoint checkpoints/locked_holdout/lora_assay_only/adapter_and_head.pt \
    --fasta candidates.fasta --output candidate_scores.json"
```

This yields uncalibrated human-TNF binder scores, not measured binding probabilities. `score.py` refuses to overwrite an existing score file.

## Sources and attribution

Experimental input: [Anthropic protein-binder design dataset](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design), revision `9e1b81696da46835e9e9cde9a3da976e0abc92ab`, CC BY 4.0 according to the dataset card. The upstream table and vendor assay documentation govern label interpretation. Public UUID-level predictions are derived from that release.

Model: [ESM/ESMC](https://github.com/biohub/esm), with upstream model/code license terms unchanged. The notebook links the official PyTorch and scikit-learn references for autograd, binary loss, and average precision.
