# Research migration

Arbitrium is the public home for Decision-specific research intended to use the Maidionis specialization foundation.

## What belongs in Arbitrium

Decision-specific research material belongs here:

- Experiment 001 and Experiment 002
- Decision curricula and fixtures
- the 24 / 96 / 72 lesson sets
- the original 1,000-record curriculum
- experiment configs
- dataset and manifest hashes
- seed 42 / 43 / 44 results
- label support / recall and confusion evidence
- answerability failures
- Decision-specific failure analysis
- Decision specialization artifacts and metadata

## What belongs in Maidionis

Specialization-neutral mechanisms belong in Maidionis:

- reusable model architecture abstractions
- generic training loop
- checkpoint and reproducibility mechanisms
- dataset and manifest contracts
- evaluation and calibration infrastructure
- artifact serialization
- Runtime-facing specialization interface

## Migration policy

Public migration uses new commits while preserving the research evidence itself.

Historical results must retain their original meaning. Do not rewrite unsuccessful results, alter split definitions, or reinterpret train-only diagnostics as held-out performance.

## Migration order

1. Bootstrap repository identity and boundaries.
2. Migrate research reports and immutable evidence.
3. Verify hashes, seeds, split semantics, and reported metrics.
4. Finalize the relevant Maidionis Core contracts.
5. Migrate Decision-specific implementation against those contracts.
6. Re-run controlled experiments under the public architecture.
7. Record new results separately from historical results.
8. Add Runtime integration after model and artifact contracts are stable.

## Initial research baseline

Existing research established several useful facts:

- a tightly bounded train-only curriculum can be fitted;
- class imbalance can make aggregate accuracy misleading;
- successful memorization does not establish held-out generalization;
- answerability / abstention remains unresolved;
- negative results are part of the research record.

These findings are migration requirements, not release-quality claims.
