# External teacher and test loop

The school is Python; model learning and inference are C++. GPT-OSS runs behind an operator-managed llama.cpp/OpenAI-compatible endpoint, outside Attention and outside the student dependency graph.

Authoritative details:

- [Education system](education-system.md): concept map, scenario oracle, blind review, semantic audit, bounded HTTP and durable state machine.
- [Dataset design](dataset-design.md): immutable records/provenance, family splits, train/dev/calibration/test isolation.
- [Training](training.md): C++ loss/optimizer/checkpoints and frozen data boundary.
- [Evaluation](evaluation.md): dev feedback and sealed final testing.
- [Operations](operations.md): commands, budgets, resume and failure recovery.

The loop is generate → blind review → verify/audit → freeze → C++ train → calibrate → dev analysis → new targeted curriculum. It runs at most the registered cycle budget. Final test is a separate one-time release evaluation, not part of that feedback loop. Same-model teacher/reviewer agreement is explicitly recorded as correlated evidence; GPT-OSS never appoints itself the ground-truth authority.

Existing `gpt_oss.py` and `orchestrator.py` are bootstrap code, not this full pipeline. Their migration gaps are listed in [implementation-plan](implementation-plan.md). Normal student inference needs neither Python education nor a teacher endpoint.
