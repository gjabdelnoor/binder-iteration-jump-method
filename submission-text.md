# Design-method form

## Name

Binder Iteration & Jump

## Repository / website

https://github.com/gjabdelnoor/binder-iteration-jump-method

## Description

Agent-guided EGFR binder search combining hypothesis-driven mutation rounds, LoRA-guided parent selection, ESMC sequence context, and ESM3 sequence jumps, followed by human/mouse complex prediction and structural review.

## Configuration for this batch

Starting from de novo EGFR designs and additional sequence-only seeds, separate GPT and DeepSeek agents proposed up to eight variants per round. Ordinary proposals used 1–10 substitutions; larger ESM3 sequence-only jumps explored alternatives when verified lineage folding evidence had ipTM <0.60. Strict improvement in a frozen human-EGFR LoRA score ordinarily selected the next parent; ESMC masked pseudo-perplexity supplied context. Deduplication and clustering selected representatives for separate human/mouse ESMFold2 and Boltz predictions. Composition, interface geometry, novelty screening, and exploratory pH 6.5-versus-7.4 calculations informed review. The final 20 include baseline designs and manually retained exploratory variants, so they did not all pass the ordinary LoRA ≥0.5 folding gate or every composition filter. Static pH estimates were sensitive to geometry repair. Binding, cross-species activity, and pH selectivity remain experimentally unverified. Method and historical settings: https://github.com/gjabdelnoor/binder-iteration-jump-method
