# AGENTS.md

This repository is part of the FLAMORIS ecosystem.

Arbitrium is the Decision specialization of Maidionis.

## Repository boundaries

- Keep Decision-specific TaskSpecs, curricula, experiment reports, results, and failure analysis in Arbitrium.
- Keep reusable specialization-neutral model, training, evaluation, checkpoint, and artifact mechanisms in Maidionis.
- Keep Workflow execution and orchestration in FLAMORIS AI Runtime.
- Preserve dataset identity, hashes, seeds, split definitions, and metric meaning.
- Keep unsuccessful research results alongside successful diagnostics.
- Do not describe train-only memorization as held-out generalization.
- Use new public commits for migrated material rather than importing earlier repository history.
- Verify current repository commands before using them.

## Before substantial changes

Read README.md and Issue #1, inspect the current Maidionis boundary, and classify the change as Decision-specific or specialization-neutral.

## Testing

Prefer deterministic tests and explicit contracts. For research migration, verify datasets, hashes, seeds, splits, metrics, and known limitations.

## Licensing

Unless stated otherwise, code in this repository is licensed under Apache License 2.0. Third-party materials remain subject to their own terms.
