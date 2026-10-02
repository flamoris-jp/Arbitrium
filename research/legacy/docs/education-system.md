# Python education system

Normative design v1. Teacher and reviewer are external to every C++ model layer. The school can stop, replay or be removed without changing inference behavior.

## Concept-driven curriculum

Version `curriculum.v1` fixes: negation, contrast, comparison, quantifiers, conditionals, temporal_order, causal_evidence, uncertainty, contradiction, evidence_sufficiency. Each concept has difficulty 1..5: single explicit fact; paraphrase/distractor; two linked clauses; nested/time-changing evidence; conflicting or missing evidence. Domain/task is always explicit. Broad Web pretraining is not a prerequisite.

The initial recovery scenario generator chooses only finite decision-relevant structured facts: operation repeat-safe (true/false/unknown), provider state (healthy/transient/permanent/unknown), retry budget (zero/positive/unknown), fallback available (true/false/unknown), fallback permitted (true/false/unknown), fatal policy violation (true/false/unknown), plus distractor events and final-time reference. The policy fixes precedence:

1. Confirmed fatal violation → stop.
2. Otherwise repeat-safe AND transient-or-healthy provider AND budget > 0 → retry.
3. Otherwise explicitly available permissible fallback → fallback.
4. Otherwise all options explicitly unavailable → stop.
5. If decisive facts are unknown or contradictory and multiple outcomes remain possible → unanswerable.

Implement the oracle by enumerating finite completions of unknown facts and collecting possible outcomes; answerable iff all admissible completions agree. Contradictions are unanswerable unless a clear later event supersedes the earlier one. Do not label all missing facts unanswerable if they cannot change the answer. The model receives English policy/evidence only, not structured oracle facts. If a product already has those reliable structured facts, it should execute this policy in ordinary code.

Python creates scenario IDs and expected outcomes **before** GPT-OSS writes paraphrases. The v1 TaskSpec owns the canonical policy text; GPT-OSS does not author or paraphrase policy rules. Teacher gets facts, the canonical supported policy as read-only context and requested concept/difficulty, but no sealed examples. Its proposed label may be retained for comparison, never substituted for the oracle. Deterministic controlled-language renderings and independently authored fixtures anchor the benchmark; LLM prose adds variety only after semantic review.

## Candidate and review stages

Candidate generation response: object `{lessons:[...]}`, each lesson has `state`, `proposed_target`, `proposed_answerable`, `rationale`. The harness injects the TaskSpec's canonical `policy` and `question`; generated output cannot replace either. Requested task, scenario ID, source identity and concept tags are attached by the harness, not trusted from model-generated IDs. `rationale` is an optional short explanation for audit; do not require or ingest hidden reasoning.

Blind review receives input, task/rubric and allowed facts, but **not** proposed label or rationale. It returns strict `{decision, target, answerable, extracted_facts, issues, notes}`. `decision` is `accept|reject|correct`; target is label ID or null. Extracted facts must match source scenario; omissions/extraneous decisive facts invalidate the paraphrase. Deterministic code checks ranges, identities, oracle outcome and fact equality; model extraction itself is not a deterministic proof of semantics.

Adjudication then compares oracle, proposed label and blind review. Outcome:

| Evidence | Disposition |
|---|---|
| Controlled rendering + oracle + schema valid | verified, eligible after audit |
| Teacher paraphrase + blind fact check + oracle agree | semantic-model-reviewed, eligible only after batch audit below |
| Teacher/reviewer disagree; facts or label uncertain | quarantine; no automatic majority promotion |
| Obvious repair | new candidate ID with `supersedes`; complete review again |
| Teacher only / no independent label basis | research-only quarantine |

Same GPT-OSS instance with different prompts is procedurally blind, **not an independent source of knowledge**. Record the identical weight hash and `same_model=true`. Deterministic verification can validate structured outcomes, but cannot by itself establish that free-form English faithfully encodes those facts. A release dataset requires a human semantic audit of at least max(100, ceil(1% of accepted paraphrases)) per task/teacher/template batch, or every record if fewer than 100. Any mislabeled/ambiguous audit record quarantines the entire affected template batch for repair and re-audit. Record auditor and outcome. Until that evidence exists, run controlled-rendering experiments; do not claim fully automatic semantic validation.

## Provider interface and transport

Separate `TeacherProvider.generate(spec)` and `ReviewerProvider.review(input)` instances/interfaces. Initial transport is HTTPX to an operator-configured llama.cpp server running GPT-OSS. `ARBITRIUM_TEACHER_BASE_URL` is an **origin without `/v1`**; append `/v1/chat/completions` once. Reject embedded credentials, URL query/fragment and redirects. A separately configured API key is read from its environment variable and redacted from logs. Nonloopback remote use must be explicitly configured by the operator.

At session start record/probe model alias, immutable teacher weight SHA-256 (operator-supplied file hash if endpoint cannot expose it), quantization, llama.cpp commit/version, chat-template hash and generation settings. Model alias `gpt-oss` alone is insufficient provenance. Capability probe verifies an actual successful strict-JSON completion and final assistant content extraction. Parse only final `message.content`; do not parse reasoning/tool-call fields. Reject truncated (`finish_reason=length`), empty or malformed content.

Use nonstreaming requests. Prefer schema-constrained response when supported by the tested endpoint; JSON-object mode plus local schema validation is an explicit recorded compatibility mode. No regular-expression extraction from Markdown fences and no boolean coercion such as `bool("false")`.

Defaults: generation temperature 0.8, reviewer 0.2; top_p=1; max output tokens 4096; batch size 8 (cap 32); concurrency 1; connect timeout 10 s, read timeout 120 s, total per-call deadline 150 s. Limits must be recorded, configurable and bounded. Stop a cycle at 500 HTTP attempts or 2 hours, whichever occurs first. No endless education loop by default.

Retry only network timeout, 429 and 5xx, at most two retries after the initial call, with exponential backoff 1/2 s plus seeded jitter [0,0.25] and capped Retry-After 30 s. 400/401/403/schema errors do not retry the identical request. One explicit corrective generation attempt is allowed for malformed output, stored as a new attempt; it consumes the same budget. Cancellation records incomplete attempts and stops new work. The client closes connections on normal/exceptional exit.

## Durable orchestration

Cycle states: planned → generated → reviewed → verified → frozen → trained → calibrated → dev_evaluated → reported. Only verified records freeze; failed/incomplete/quarantined items remain separate. Transition log is append-only, written before an output becomes visible. A record advances only after output hashes are verified. Stage output uses temporary files and same-filesystem atomic rename, with flush/fsync for durable checkpoint boundaries. One writer per cycle; lock file prevents concurrent writers.

Attempt key hashes curriculum version, task/scenario, prompt hash, teacher identity, role, parameters and attempt number. On resume, reuse existing verified responses; do not silently re-call teacher and replace them. An HTTP timeout can leave remote completion uncertain; store this fact. Retries may generate different content and must have different attempt IDs. Deduplication happens before acceptance. Replaying saved raw responses reproduces decisions without a network dependency.

## Weakness feedback

Only dev records may drive targeted lessons. Aggregate confusion, high-confidence errors (p>=0.80), answerability mistakes and concept/difficulty counts. A concept needs >=50 dev examples to drive automatic quota changes; otherwise report low support. Next-cycle quota: 50% fixed coverage curriculum, 30% largest supported error slices, 20% verified contrast/minimal pairs; each concept remains >=5% where ten concepts apply. Cap at 10,000 newly accepted records per cycle and 3 cycles per experiment. Stop when two cycles fail to improve dev macro-F1 by at least 0.005 without calibration/risk regression; do not mine sealed test. No automatic promotion occurs.

See [dataset](dataset-design.md), [evaluation](evaluation.md) and [operations](operations.md) for split assignment, final gates and restart semantics.
