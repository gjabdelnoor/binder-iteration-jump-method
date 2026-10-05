# Week in review

Retrospective assembled October 5, 2026, from project artifacts and relevant coding-session history. Dates describe the reviewed campaign, not newly executed experiments.

| Period | Change in approach | Consequence |
| --- | --- | --- |
| September 28–30 | Starting designs, target preparation, surrogate screens, and pH-analysis infrastructure were explored. | Early computational evidence supplied starting hypotheses; infrastructure success did not establish pH selectivity. |
| October 1–2 | Classifier screening and separate human/mouse structural evaluation supported agent-guided mutation rounds. | The implemented engine used eight-candidate batches, strict parent-score improvement, and ESMC as context. |
| October 2–3 | The jump fork added sequence-only seeds and larger ESM3 completions, then revised how historical low-ipTM evidence triggered jumps. | Small edits could be replaced by broader exploration while preserving ancestry and score receipts. |
| October 3 | The mutation export was deduplicated and clustered for broader structural follow-up. | Diversity review became necessary because high scores could concentrate in a few families or low-complexity sequences. |
| October 4–5 | Composition and interface review, geometry repair, pH sensitivity checks, and manual final-batch changes followed. | Static pH rankings were found fragile; the final local set mixes baselines, local mutants, and exploratory exceptions. |

## Checked snapshot counts

The October 3 export contains 4,748 GPT records and 4,418 DeepSeek records: **9,166 total**, including 27 baselines and 55 archived pilot rows. It contains **322 accepted mutations**: 144 GPT and 178 DeepSeek. Candidate identifiers must include the arm because variant names can repeat between arms.

During this review, the CSV row totals were independently counted, sequences checked for canonical amino acids, and global IDs checked for uniqueness. The historical manifest records successful parent-edit reconstruction for 9,084 live mutations. That reconstruction was not independently rerun here.

The final local submission CSV contains 20 proteins: three baseline-named designs, fifteen GPT seed4 variants, and two DeepSeek seed6 variants. This is a concentration in a few families, not twenty independent backbone bets. Two manually retained DeepSeek designs were previously composition-flagged, so the entire batch cannot be described as passing the stricter composition screen.

## Findings that change the interpretation

**Classifier exploitation:** the later ranking notes describe many high-scoring, low-complexity sequences. A higher surrogate score was therefore insufficient grounds to claim improved binding.

**pH sensitivity:** geometry repair left sequences unchanged but substantially altered static PROPKA ratios. For one recorded design, the estimated pH 6.5/7.4 association-constant ratio changed from approximately 13 to approximately 1.1. This was a model-sensitivity observation, not a measured affinity change. The earlier pH ordering should not be treated as robust.

**Uneven evidence:** not every scored sequence was folded, not every sequence had both species, and missing pH results were unknown. Evidence for a parent or another variant was not evidence for a new sequence.

**No new experimental validation:** the reviewed final candidates do not have experimental binding or pH-selectivity results in this release.

## Provenance and review limits

[provenance.json](provenance.json) gives project-relative source paths and SHA-256 hashes. The underlying files remain private. The session catalog was refreshed at 2026-10-05T06:07:07.781441Z and reported Codex, Claude, and Kimi coverage with no malformed records. The week catalog was paginated through all 294 entries; relevant conversations and project artifacts were selected for review, not all 294 conversations read in full.

The October 2 design discussion established that ESMC should supply context rather than a hard gate and that search should branch from retained parents. The installed engine and later policy documents determine the actual eight-candidate limit and jump behavior, superseding earlier conversational proposals.

This was a documentation and provenance review. No GPU jobs, folding runs, or wet-lab experiments were repeated. The installed Synthetic Data Audit skill lacked its referenced detector script; row/identifier/sequence checks and direct source reconciliation were performed instead. This is not a full forensic certification of every historical computation.
