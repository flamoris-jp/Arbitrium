# Arbitrium

**A Maidionis Decision specialization for bounded advisory judgments over supplied evidence.**

Arbitrium is the first Decision specialization built on Maidionis.

It owns Decision-specific TaskSpecs, curricula, experiment reports, measured results, failure analysis, and specialization metadata. Specialization-neutral model/training/evaluation infrastructure belongs in Maidionis. Workflow execution and runtime orchestration belong in FLAMORIS AI Runtime.

## Current status

**Bootstrap / research migration planning.**

The public repository does not yet contain the migrated implementation or experiment archive.

Existing controlled research shows that a tightly bounded train-only curriculum can be fitted. It does **not** establish reliable held-out generalization or production quality.

## Research home

Arbitrium is the public home for Decision-specific research evidence, including:

- Experiment 001 and Experiment 002
- 24 / 96 / 72 lesson curricula
- the original 1,000-record curriculum
- dataset and manifest hashes
- seed 42 / 43 / 44 results
- label support / recall evidence
- held-out failures
- answerability failures
- reproducibility information

Successful training diagnostics must not be presented as stronger evidence than they are.

## Relationship

```text
Maidionis
  ↓ specialization foundation
Arbitrium
  ↓ bounded Decision inference
FLAMORIS AI Runtime
  ↓ workflow / execution
Applications
```

## Migration policy

Do not import the previous research Git history wholesale.

Public migration uses new commits while preserving research meaning, dataset identity, hashes, seeds, metrics, and known failures.

Implementation migration should wait for the relevant Maidionis Core contracts so specialization-neutral code can move into Maidionis instead of being duplicated here.

Repository setup and migration are tracked in [Issue #1](https://github.com/flamoris-jp/Arbitrium/issues/1) and [Research migration](docs/research-migration.md).

## Getting started

There is no supported public build or run command yet.

Do not copy commands from earlier research material until the corresponding implementation has been migrated and verified in this repository.

## FLAMORIS

FLAMORIS is open-source software for creative work and AI-native production.

Use it however you like.

Commercial use is welcome and does not require permission.

FLAMORIS software is provided as-is. We do not provide individual support or guaranteed assistance.

If FLAMORIS helps you or you find it interesting, your support helps fund development and keeps the project growing. 🌱

<sub>Mostly GPU bills.</sub>

## License

Code in this repository is licensed under the [Apache License 2.0](LICENSE), unless otherwise noted.

AI models, model weights, datasets, media, and other non-code assets may use separate licenses.
