# Implementation plan and acceptance matrix

Status: design only. This list describes future work; no item is complete merely because its specification exists. Implement after design review. Preserve the C++/Python boundary and do not enable deployment automatically.

## Inspected scaffold gaps

At main `0e88cad5ab676584e0f2e827c2b553d398c535e4`:

- CMake builds `arbitrium_core`, `arbitrium_smoke`, `arbitrium_tests`; only mean-pooled embeddings plus a linear choice head exist. There is no tokenizer, trainer, calibration, artifact loader or typed text API.
- `forward` has no mask/shape/range validation; `decide` calls scalar `.item()` and assumes a single input. Softmax confidence is uncalibrated. Smoke IDs are synthetic and do not demonstrate language understanding.
- `configs/tiny.yaml` is intent, not loaded config. Its old architecture name/max length must migrate to resolved v1 config with validation.
- Python `pyproject.toml` lives in `education/python` but package/test paths repeat `education/python/...`; fix to `src` and `tests` relative to that project root and verify installed-package import from outside the checkout.
- Teacher adapter hardcodes temperature 0.2, lacks strict schema/provenance/capability checks, has ambiguous base-URL joining and no explicit close/retry state. `bool(payload["accepted"])` would accept a nonempty `"false"` string.
- Review currently sees proposed label/rationale and uses the same provider. Orchestrator mutates corrected labels in place, losing original label/review lineage.
- No frozen split registry, audit persistence, benchmark or training implementation exists. Existing tests are bootstrap tests, not evidence for this specification.

These observations are from source inspection; this design task did not build LibTorch or call a live teacher.

## Work packages

| ID | Deliverable / affected areas | Depends on | Acceptance evidence |
|---|---|---|---|
| I01 | Build/package baseline; CMake dependency discovery; `education/python/pyproject.toml`; exact tested dependency manifest | design approval | clean CPU build; installed Python import from unrelated cwd; offline tests; recorded compiler/LibTorch/SentencePiece hashes |
| I02 | `contracts`, strict config/JSON validators and C++ typed request/result/task/errors | I01 | shared design fixtures + negative mutations agree in C++/Python; duplicate keys, string bools, unknown versions/fields rejected |
| I03 | Python curriculum/scenario oracle, source IDs, lineage and deterministic fixtures | I02 | explicit truth tables including unknown facts/precedence/time overrides; expected outcomes independently checked |
| I04 | Teacher/reviewer transport, blind prompts and audit journal | I02,I03 | mocked HTTP success/429/5xx/timeout/truncation/cancel/replay tests; same-model identity recorded; no loss of original label; optional separately recorded live probe |
| I05 | Review adjudication, human audit references, family union, dedup, split registry and atomic freeze | I03,I04 | no unverified record freezes; cross-split paraphrase rejected; corrupt hash/ref rejected; interrupted/resumed freeze; manifest counts |
| I06 | Native tokenizer adapter, trainer invocation, token cache and input assembly | I01,I02,I05 | train-only corpus audit; C++/Python token parity; special-string and UTF-8/byte/256-token boundary fixtures |
| I07 | C++ JSONL reader, task-bound dataset loader and deterministic batch order | I02,I05,I06 | target/provenance excluded from tensors; unsupported labels/malformed rows fail; stable shuffle/resume inventory |
| I08 | Pooled model, answerability head, masked loss and full training loop | I07 | finite-gradient test; tiny fixture overfit; unanswerable decision loss is zero; padding/batch invariance; real train/dev report |
| I09 | C++ native checkpoints, config/RNG/optimizer/scheduler resume | I08 | interrupted epoch resumes from previous boundary; uninterrupted vs resume parity; mismatched data/config denied |
| I10 | Explicit tiny encoder Q/K/V, positions/segments, FFN and CLS heads | I08 | forward/backward parity fixtures, key/query mask correctness, finite gradients; registered pooled/encoder comparison |
| I11 | C++ primitive metrics and Python benchmark/report aggregation | I02,I08 | hand-computable accuracy/F1/Brier/ECE/RPS/coverage cases; missing/duplicate predictions invalidate run; family-bootstrap report |
| I12 | Temperature fitting, gate selection and versioned calibration | I09,I11 | synthetic logits known optimum sanity; T=1 no-worse check; sparse support and no-valid-gate behavior; incompatible bindings denied |
| I13 | Bundle loader/export, manifests/model card and planned inference CLI | I06,I09,I12 | corrupt/oversized/path-traversal/mismatched artifacts rejected; CPU save/load parity; offline serving without teacher; status/label-order mapping |
| I14 | Bounded school cycle, dev-only feedback, budgets and recovery | I04,I05,I11,I13 | crash/replay avoids duplicates; teacher gets no heldout records; max-cycles/attempt/time stop; no auto-promotion |
| I15 | Registered choice experiment, audited dataset and sealed evaluation | I10,I12,I13,I14 | all seeds/selected candidate justified by dev; sealed ledger; explicit gate outcome and limitations; no claim if failed |
| I16 | Binary and ordinal task fixtures/artifacts using same infrastructure | I13,I15 | separate task/calibration identity; fixed binary ordering; ordinal expectation/RPS/range; per-task evidence or explicit research-only status |
| I17 | Local caller/worker integration proof and handoff | I13,I15 | fake executor only on policy-approved ok; abstain/error/timeout/stale-state cases; external Intelligence contract mapped only after its repo is inspected |

I01–I14 and I17 constitute the reusable implementation. I15 provides first-task research evidence. I16 completes the planned primitive extensions; unsupported tasks remain unadvertised until their evidence is available. No fixed dataset size or number of iterations guarantees that the scientific quality gate will pass.

## Work order and independently implementable areas

Land contracts before modules consume them. After I02/I03, provider/audit development can proceed independently of native tokenizer/training work using shared fixtures. Model improvements follow the measured pooled baseline. Calibration depends on stable metric definitions; artifact loader depends on explicit component hashes. No parallel task may redefine a shared schema locally. This is a dependency plan, not an instruction to spawn agents during the design task.

Suggested implementation PR groups: foundation/contracts (I01–I03); education/data (I04–I07); pooled training/checkpoint/metrics (I08–I09,I11); encoder/calibration/bundles (I10,I12–I13); cycle/experiments/extension/integration (I14–I17). If one implementation branch is preferred, retain these independently reviewable commits and gates. Do not merge speculative GPU or generic-agent features into this sequence.

## Required semantic and failure cases

| Case | Expected behavior | Requirements |
|---|---|---|
| Same input with choices permuted | same ID-keyed probabilities/verdict, output in request order | R04 |
| New/subset/duplicate labels or task | explicit request error before forward | R04,R12 |
| Retry vs unsafe repeat with otherwise same wording | target changes under policy; must be evaluated on held-out pairs | R02,R05 |
| Missing decisive evidence | answerability supervision, eventual abstain; never forced stop | R05,R07 |
| Answerable despite irrelevant missing fact | oracle and model benchmark do not over-abstain | R05,R07 |
| Teacher correction | new sample/lineage; original preserved | R03,R05 |
| Same model teacher/reviewer agreement | recorded correlation, cannot replace semantic audit | R05 |
| Literal control token / extra field / string boolean | safe content encoding or strict rejection | R04,R12 |
| Padding and B=1 vs batched inference | numerical parity | R02,R11 |
| Dataset or tokenizer changed after calibration | invalid artifact binding | R07,R09 |
| Failed release gate / no threshold pair | research-only, no automatic deployment | R07,R10 |
| Sealed errors opened | suite retired; no teacher feedback under independent-test claim | R06 |
| Teacher unavailable after student export | inference still works | R03,R10 |
| Interrupted cycle/train | replay or epoch-boundary resume with audit trail | R08,R11 |
| Runtime state changes after inference | caller refuses stale action | R10 |

## Definition of implementation done

Every package has its stated evidence, real command documentation and explicitly pinned tested dependency profile. All schema/shape and semantic fixtures pass; source/current docs agree; an end-to-end controlled dataset run works offline except deliberate teacher generation; resumed runs preserve lineage; no secrets/large artifacts are committed. Reports state passed/failed/untested gates. Remaining scientific risks are conclusions, not hidden TODOs. Deployment, a new external repo API and GPU acceleration are separately scoped changes.
