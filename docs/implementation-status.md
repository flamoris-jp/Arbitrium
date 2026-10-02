# Initial public Decision implementation

A1 preserves 130 original research/code/fixture files under `research/legacy` with
an immutable SHA-256/byte inventory. All four dataset manifests and their eight
members retain original bytes. Original weights/tokenizer binaries/checkpoints
are unavailable; administrative metadata, source history and one document with
non-public repository coordinates are omitted. The archive annotation documents
these limits. New public commits do not import the original Git history.

A2 is a source-backed Linux x86_64 CPU FP32 reference specialization using the
pinned Maidionis Core in `dependency.lock.json` and the companion
[Maidionis PR #6](https://github.com/flamoris-jp/Maidionis/pull/6). Core owns
numerical components, deterministic training/checkpoints, immutable datasets,
artifact verification/loading and complete evaluation. Arbitrium owns the
Decision schemas, TaskSpec, finite education/audit oracle, codec, four-output
objective composition, diagnostic metrics, advisory gates, legacy partition
adapter and separate offline provider boundary.

The native composition is an explicit compiled registry, not a Decision default
in Core. Code/config/schema/build pins are independently checked. The bounded
ASCII word codec uses only original train text to construct its vocabulary,
preserves `fields.v1` field/control/segment ordering and rejects unsupported
policy/question/choice sets and >256 assembled tokens. It replaces SentencePiece
in this reference profile and is not tokenizer/archive compatible with legacy
models. Structured oracle facts are used for fixture audits, never model input
or inference. The model consumes English text, not oracle fact vectors.

Derived datasets get new IDs/digests and inventoried source ancestry. The explicit
legacy partition retains exact original family allocations/counts; it accepts
only these inventoried research fixtures and seed 42. It cannot be selected by
request metadata or installed into Core's default partition. Native validation
requires complete original family membership and split support, including empty
splits. This preserves a historical experimental condition; it does not turn
legacy grouping or public holdouts into a sealed benchmark.

A3's fixed 12-run comparison uses the 24/96/72/1,000 curricula and all seeds
42/43/44. `research/public-architecture/plan.json` records the conditions before
measurement. Each run trains actual Core components, exports selected-best,
validates without constructing a model, reloads the real archive through a
2 GiB isolated-process admission, infers/evaluates the complete selected split,
rejects missing prediction and serving attempts, finalizes `research_only` and
reloads with identical outputs. Reports remain separate from historical reports.
See the comparison report for actual results and validation evidence.

## Honest limits

- The old SentencePiece codec, initialization, separate head modules, decay/order/
  schedule details and recorded LibTorch version differ. These are controlled
  migration comparisons, not exact old-score/weight reproductions or an isolated
  test of which change caused a metric difference.
- 24/96 use train diagnostics. 72/1,000 select/report on development data; those
  reports are selection-biased development observations, not an untouched test.
  Calibration/test payloads are verified by the freezer but not fed to model
  training, vocabulary construction, epoch selection or evaluation.
- No calibration artifact is fitted. The actual exported reference composition
  always abstains on advisory output (`release_gate_not_met`); diagnostic labels,
  raw probabilities and answerability are diagnostics. Unit-tested calibrated
  gate math is a policy helper, not a calibrated-model or serving claim.
- Provider execution requires an explicit host-supplied admitted callable. No live
  teacher call, network inference, autonomous education, Runtime adapter,
  Workflow execution, GPU qualification, sealed release support or deployment
  is included. Runtime R1 remains separate.
- Public APIs and artifact formats are experimental. Python tooling expects an
  editable source-backed checkout with research assets; no standalone installed
  wheel/serving package is qualified.
