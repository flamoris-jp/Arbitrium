# Controlled recovery sample

1,000 English synthetic lessons: 250 retry, 250 fallback, 250 stop and 250
unanswerable. This is a research fixture, not a release dataset or sealed test.

See [Experiment 001](../../docs/experiment-001.md) for generation, native training,
split counts, provenance, limitations and measured results. Each JSONL row is a
real `arbitrium.sample.v1` input/target record. The provenance index is audit-only
and never fed to the model. Public test rows are reserved and not used by the runner.
