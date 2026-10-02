# Contracts and errors

Status: normative design v1. [JSON Schemas](contracts/request.schema.json) are machine-readable shape constraints. The semantic constraints below are equally mandatory. All JSON rejects duplicate keys, nonfinite numbers and unknown properties; no coercion of strings to booleans/numbers. Text is UTF-8. Opaque IDs use `[A-Za-z0-9._:-]`, 1..128 characters. All SHA-256 values are lowercase 64 hexadecimal characters.

## TaskSpec (immutable artifact member)

| Field | Type / rule |
|---|---|
| schema_version | literal `arbitrium.task.v1` |
| task_id | versioned opaque ID; changed semantics require a new ID |
| kind | `choice`, `binary`, or `ordinal` |
| question | nonempty canonical English question, <=2 KiB |
| description, policy_scope | nonempty English task/rubric and supported policy limits |
| canonical_policy | nonempty canonical English policy text, <=8 KiB; fixed for the task |
| labels | 2..16 objects `{id, description, anchor}` in canonical order; unique IDs; anchor null except ordinal |
| positive_label | `true` for binary; null otherwise |
| max_sequence_length | 256 in v1 |
| assembly_version | `fields.v1` |

Binary label IDs are exactly `false,true`. Ordinal anchors are strictly increasing finite numbers; first/last define the fixed range. The initial recovery question is `What should the workflow do next?`. Fixed labels: retry = repeat the same operation; fallback = use an explicitly available alternative; stop = stop this workflow attempt. A missing answer is not stop. The TaskSpec canonical policy defines when repeat/alternative/stop is allowed; missing or conflicting decisive evidence is labeled unanswerable.

Later binary fixture task `evidence.support.v1` asks `Does the evidence support the claim?`, with claim in policy and evidence in state; false means evidence refutes, missing evidence means unanswerable. Ordinal fixture task `criteria.coverage.v1` asks `How many of the four criteria are satisfied?`: policy lists exactly four yes/no criteria; state supplies evidence for each; labels `level0`..`level4` map to anchors 0..1 in quarters. Unknown/conflicting criterion evidence is unanswerable. These are explicit extension tasks, not generic quality judges.

## DecisionRequest

Required fields: `schema_version="arbitrium.request.v1"`, `request_id`, `task_id`, `kind`, `language="en"`, `state`, `policy`, `question`, `choices`. See [schema](contracts/request.schema.json) and [example](examples/recovery-request.json).

`choices` is an array of label IDs, containing **every** artifact label exactly once; order may differ from canonical. Validate kind/task match, exact canonical question, exact canonical policy and identical label set before inference. Inference returns probabilities in request order, keyed by IDs. Arbitrary choices, subsets, extra tools, new label meanings, variable scoring ranges, and non-English language declarations are rejected. This resolves the scaffold's misleading suggestion of unrestricted candidate selection.

No artifact filesystem path, endpoint URL, prompt template, target, or execution command is accepted as a request field. Loaded artifact selection belongs to the host application's allowlist.

## DecisionResult

Required fields: schema/version, echoed request/task IDs, artifact ID, kind, status, verdict, probabilities, score, confidence, answerability, reason_code, calibration_id, error. See [schema](contracts/result.schema.json).

| Status | Semantics |
|---|---|
| `ok` | verdict = argmax label ID (tie by canonical task order); probabilities sum to 1 within 1e-5; confidence=max probability; answerability in [0,1]; ordinal score=sum(p*anchor); other scores null; reason/error null |
| `abstain` | verdict/score/confidence null; calibrated probabilities and answerability retained as diagnostics; reason `low_answerability`, `low_confidence` or `release_gate_not_met`; error null; never translate into stop |
| `error` | verdict/score/confidence/answerability null, probabilities empty; structured error `{code,retryable,message}`; reason null; artifact/calibration IDs may be null if not loaded |

Confidence is conditional class probability within this task, not an overall probability that execution will succeed. For ordinal tasks it is the modal bin probability, not confidence in the mean score. No generated explanation or hidden reasoning is returned. Host must branch on status before reading probabilities. Diagnostics on abstention are not recommendations.

## C++ and process boundary

Planned C++ API: `DecisionEngine::load(bundle_dir, expected_digest, DeviceOptions) -> engine-or-LoadError`; `decide(const DecisionRequest&) -> DecisionResult`. `forward` is internal training API. Serving loads read-only weights in eval mode with no autograd. The initial engine is single-call-at-a-time; host serializes concurrent requests. Per-worker engines may later share immutable storage after measured concurrency tests. No C++ ABI stability is promised across toolchains.

Python invokes C++ CLI using an argv list, never `shell=True`. No pybind/FFI is needed in v1. Planned CLI:

| Command | Inputs | Outputs |
|---|---|---|
| `arbitrium validate-data --dataset DIR` | frozen manifest + split JSONL | validation report |
| `arbitrium train --config FILE --dataset DIR --tokenizer DIR --run DIR` | validated config/data | checkpoints + train/dev metrics |
| `arbitrium calibrate --checkpoint DIR --dataset DIR --out DIR` | fixed weights + calibration splits only | calibrated candidate bundle |
| `arbitrium infer --artifact DIR --expected-digest HEX --input FILE --output FILE` | unlabeled request JSONL | exactly one result per validly framed input line |
| `arbitrium infer --artifact DIR --expected-digest HEX --stdio` | JSONL stream stdin | JSONL stream stdout; stderr logs only |
| `arbitrium evaluate --artifact DIR --expected-digest HEX --dataset DIR --split NAME --out DIR` | reviewed targets kept outside forward | predictions + primitive metrics |
| `arbitrium inspect --artifact DIR --expected-digest HEX` | manifest | validation summary, no execution |

Malformed JSON lines produce `invalid_request` with null request/task/kind if unparseable, then processing continues. Oversized lines are drained without unbounded allocation. Blank lines are malformed, not ignored. Single-request time limits are enforced by host process; if a worker is killed, pending requests become caller-generated timeout errors, never fictional model answers. Infer exits 0 when stream fully processed even if some lines are errors; exits 2 for invalid invocation, 3 artifact/config failure, 4 compute/internal failure, 5 I/O/resource failure. Evaluate additionally exits 6 on incomplete predictions or metric failure; incomplete runs cannot pass a gate. It validates all input records before evaluation begins.

## Error taxonomy

| Code | Retriable default | Meaning |
|---|---|---|
| invalid_request | false | malformed/type/empty/duplicate keys |
| unsupported_schema / unsupported_task / unsupported_language | false | incompatible contract or task |
| invalid_choices / question_mismatch | false | task shape changed |
| input_too_long | false | byte or assembled token limit |
| artifact_invalid / artifact_incompatible / calibration_missing | false | corrupt, mismatched or incomplete model |
| resource_exhausted | false | caller must change resource configuration |
| nonfinite_output | false | failed numerical integrity |
| cancelled / deadline_exceeded | true | caller may retry once within its own policy |
| io_error / internal_error | false | operational failure; inspect diagnostics |

Operational retryability is distinct from model verdict `retry`. No automatic teacher/remote-model fallback occurs. Unsupported devices error; CPU fallback must be explicitly configured by the operator.
