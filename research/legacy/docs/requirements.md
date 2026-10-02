# Requirements and scope — design v1

Status: proposed implementation specification; not an implemented capability. Baseline inspected: `0e88cad5ab676584e0f2e827c2b553d398c535e4` (2026-09-29). Review entry: [design-review.md](design-review.md).

## Goal and success definition

Test whether a small, randomly initialized, English-only C++ model can learn useful judgments over bounded semantic evidence. Produce a reproducible trained artifact, an external education loop, honest held-out measurements, and a transport-neutral advisory API. A negative research result is acceptable; unsupported claims of understanding are not.

The first task is `runtime.recovery.v1`, with fixed labels `retry`, `fallback`, `stop`. A single artifact implements one registered task. Binary and ordinal tasks reuse the architecture but have their own task specification, dataset, weights, calibration, and release gate. Arbitrary candidate ranking, arbitrary questions, arbitrary policy interpretation, and zero-shot new tasks are not v1 capabilities.

| ID | Requirement | Specification / acceptance |
|---|---|---|
| R01 | C++20 owns model, training, calibration and inference | [architecture](architecture.md); no Python model implementation |
| R02 | Random initialization; repository-owned architecture | [model-design](model-design.md); no pretrained weights or vocabulary |
| R03 | Python owns school; GPT-OSS is external | [education-system](education-system.md); offline replay succeeds without teacher |
| R04 | Typed, bounded, task-specific requests and results | [interfaces](interfaces.md); invalid requests rejected before forward |
| R05 | Immutable data with verifiable labels and lineage | [dataset-design](dataset-design.md); no teacher-only records in release data |
| R06 | Evaluation cannot silently become training | [evaluation](evaluation.md); group isolation and sealed-test access ledger |
| R07 | Calibrated probabilities and explicit abstention | [calibration](calibration.md); coverage/risk and answerability reported |
| R08 | Offline, restartable education and training | [operations](operations.md); fault recovery without duplicated accepted records |
| R09 | Versioned, verified model bundles | [artifact-format](artifact-format.md); corrupt or mismatched bundles fail load |
| R10 | Caller owns execution and policy | [integration](integration.md); no action executed by core |
| R11 | Reproducible results with bounded claims | [training](training.md); exact environment recorded; cross-device equivalence measured |
| R12 | Resource limits and explicit failures | [interfaces](interfaces.md), [security](security.md); no silent truncation or CPU fallback |
| R13 | Implementable sequence with evidence gates | [implementation-plan](implementation-plan.md); completion is traceable to acceptance tests |

## Non-goals

No chatbot, agent loop, workflow engine, RAG, unrestricted English comprehension, chain-of-thought generation, continual production learning, OS/device management, automatic teacher deployment, custom tensor framework, MCP inside core, or merging Oblivionis persistence into Arbitrium. GPU acceleration and multi-task shared weights are extensions, not prerequisites for the first CPU result.

## Defaults and boundaries

Linux x86-64 CPU FP32 is the reference execution profile. C++20/CMake 3.24+, LibTorch, SentencePiece, JSON/cryptographic parsing helpers are allowed. Python 3.11+ hosts education. No assertion is made about a particular GPU/driver combination until its build and parity tests pass. Exact tested dependency versions are recorded at implementation bootstrap, never fabricated by this design.

All inference input is caller-supplied; the model has no hidden history. Task policy fixes the decision interpretation. In v1 each TaskSpec also fixes one canonical policy text; DecisionRequest.policy must match that canonical policy exactly after the specified line-ending normalization. Paraphrasing or dynamically changing policy rules is not a supported inference capability. Operational permissions, retry budgets, and tool availability should be resolved in ordinary code whenever possible. The initial recovery task is a controlled research example over linguistic evidence, not a replacement for a reliable numeric retry policy.

## Delivery boundary

This change contains documentation, design schemas and illustrative fixtures only. Existing scaffold remains intact. Commands marked **planned** do not yet exist. Design approval permits an implementation plan to be reviewed; it does not claim measured quality, a trained model, or deployment readiness. [Roadmap](roadmap.md) separates implementation completion from experimental success.
