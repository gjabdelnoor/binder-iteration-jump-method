# Method and historical configuration

## Objective and starting material

The project sought proposed EGFR binders compatible with both human and mouse target proteins, with preferential binding at pH 6.5 versus 7.4 as a design objective. Earlier exploratory work also considered pH 7.5. These were objectives, not established properties.

The search inherited five starting families from the original `binder_evolve` campaign and later added independent sequence-only seeds. A starting backbone provided structural context only for its own sequence. Mutating a sequence did not validate its coordinates or preserve its interface by assumption.

## Iteration

GPT and DeepSeek operated separate histories with the same scoring engine. Each round recorded the current parent, a falsifiable hypothesis, proposed sequences, scores, and the selection outcome. The implemented batch limit was eight candidates. Ordinary variants used 1–10 substitutions relative to the current parent, with unchanged sequence length and canonical amino acids.

The frozen human-EGFR LoRA classifier was the optimization signal. A parent ordinarily changed only on a strictly greater stored score. ESMC masked pseudo-perplexity was context for the proposer, not a binding probability or acceptance threshold. The original engine stopped a lineage at LoRA ≥0.5; the jump fork allowed continued search.

An explicit neutral-promotion path allowed exact full-precision LoRA ties only with recorded, pre-scoring pH intent and sequence-specific structural reasoning. A lower score was not an allowed escape move. This path's existence does not establish that every lineage used it or that any neutral move improved pH behavior.

## Larger jumps

ESM3-open supplied masked sequence completion with durable, current-parent generation receipts. It was used in sequence-only mode; this was not a validated target-conditioned binder generator.

The initial jump policy required paired, exact-parent human/mouse evidence with minimum ipTM <0.60. The later version-2 policy considered verified items in the same arm's lineage history: human **or** mouse ipTM <0.60 triggered jump-only proposals. Missing evidence remained unknown; exactly 0.60 did not trigger. Retained historical lows could continue to trigger after a parent changed.

In version-2 jump mode, every scored proposal required a valid generation receipt and **more than ten actual substitutions** from its current parent. Small local fillers were excluded. The same strict score-improvement rule still applied, so a required jump could lose and leave the parent unchanged. Generation failures were recorded rather than replaced with fabricated candidates or scores. Previously frozen rounds retained their recorded policy, with specifically documented unscored-round repairs.

## Folding and downstream selection

The ordinary agent folding gate was candidate-specific LoRA ≥0.5. Later operator-authorized follow-up included below-gate accepted parents and diversity-selected representatives; the final batch must not be described as uniformly passing 0.5.

Human and mouse EGFR complexes were predicted separately. The documented ESMFold2 structural recipe used the base model independently of the classifier LoRA, no MSA, 20 refinement loops, 200 diffusion steps, and one sample. The later Boltz 2.1.1 candidate set used empty MSA, three recycles, 200 steps, and seed 20261003. Individual artifact provenance controls which recipe applies.

The October 3 broad triage used LoRA ≥0.1, Shannon sequence entropy ≥2.5 bits, and Asp+Glu fraction <0.35, processing arms separately. Leader clustering at 80%, 90%, and 95% identity supported diversity selection. A 90-candidate follow-up selected up to three representatives per 80% cluster, avoiding ≥95%-identical picks within a cluster. These were follow-up settings, not universal criteria for every round or final protein.

Later review added stronger composition checks, compared human/mouse confidence, inspected interfaces and geometry, and examined novelty against available references. Local novelty screening does not guarantee passage of the submission platform's novelty rule. Manual exceptions and baseline designs were retained in the final local batch.

## pH analysis

PROPKA bound-versus-free sidechain pKa calculations supplied a static proton-linkage estimate for pH 6.5 versus 7.4. Human and mouse estimates were kept separate; missing coverage was not treated as zero. Exploratory Proton-PottsMPNN state-energy scores were a separate analysis and were not interpreted as measured pKa, affinity, or selectivity.

An intermediate ranking gave pH 40%, minimum human/mouse ipTM 30%, LoRA 20%, and human ipTM 10%, using within-pipeline ranks. Subsequent geometry repair substantially changed the PROPKA estimates, so that ranking is historical and should not be presented as a validated final objective.

## Interpretation

- LoRA is an uncalibrated optimization score; optimizing against it can exploit the classifier.
- ipTM describes model confidence and is not measured affinity.
- Sequence-only generation does not establish target binding.
- Static pH estimates depend on the pose and protonation assumptions and omit relevant environmental effects.
- Computational human/mouse agreement does not establish biological cross-reactivity.
- Final variants require experimental binding and pH testing.
