# Roadmap — reviewed-design first

Current: bootstrap source exists; complete v1 design is proposed for review. No trained language model or quality result is claimed.

| Phase | Scope | Exit / work packages |
|---|---|---|
| 0 — Design review | requirements, contracts, model, school, evaluation, artifacts and operational semantics | review decisions in [design-review](design-review.md); this PR |
| 1 — Foundation | CPU toolchain, packaging, contracts and scenario fixtures | I01–I03 evidence |
| 2 — Education and data | strict teacher/reviewer, provenance/audit, family splits, tokenizer/loader | I04–I07; one replayable frozen dataset |
| 3 — Measurable baseline | pooled model, answerability, training/resume and metrics | I08–I09,I11; three-seed baseline report |
| 4 — Encoder and calibrated artifact | explicit encoder, calibration/gates, trusted bundle and inference | I10,I12–I13; verified offline candidate |
| 5 — Bounded learning cycle and choice result | aggregate dev feedback, registered choice experiment, sealed evaluation | I14–I15; honest positive/negative research conclusion |
| 6 — Typed primitive extensions | separate binary and ordinal task artifacts | I16; no generic arbitrary-task claim |
| 7 — Integration proof | local caller/worker with fake executor and policy checks | I17; external adapter is a follow-on repo change |
| 8 — Evidence-driven research | broader language corpus/MLM, larger model, GPU, dynamic candidates, multi-task or Oblivionis bridge | separate proposals only after a measured need |

[Implementation plan](implementation-plan.md) specifies dependencies and tests. Implementation may finish even if the first model fails quality gates; the correct deliverable then is a reproducible research-only artifact and failure report. Automatic deployment or indefinite teacher loops are not roadmap exit criteria.
