# Operation, budgets and recovery

All commands in this document are **planned**. [build.md](build.md) documents the existing smoke targets.

## Local run workspace

Use `runs/<run-id>/` for resolved configs, stage journal, raw responses, audit records, reports and checkpoints; generated contents remain ignored/untracked. IDs are immutable; local output paths are supplied by operator. The school orchestrator never assumes a particular hostname/GPU/server-manager layout.

Planned school commands: `arbitrium-education generate --config FILE --run DIR`; `review --run DIR`; `verify --run DIR`; `freeze --run DIR --out DIR`; `benchmark --artifact DIR --dataset DIR --split dev --out DIR`; `analyze --report DIR --out DIR`; `cycle --config FILE --run DIR --max-cycles 3`; `resume --run DIR`. All require explicit run/config paths. Benchmark defaults to dev; sealed test requires an experiment registration and explicit `--sealed-test` invocation. Cycle never invokes sealed test or promotion.

## First end-to-end procedure

1. Bootstrap and pin tested dependencies; run C++ and Python offline tests.
2. Register task/experiment/config and generate a small controlled fixture set to prove all stages.
3. Configure existing teacher origin/model identity; capability-probe one generation and one blind review request. No server provisioning is implied.
4. Produce scenario families and concept-balanced lessons within the recorded budget. Review/verify/audit and freeze a dataset only when eligibility/minimum-support gates pass.
5. Train SentencePiece on train fields, persist exact vocabulary/options, then train pooled model for the registered seeds. Measure dev; then train tiny encoder with the same data.
6. Select candidate on dev, fit temperatures on calibration_fit and gates on calibration_select, export candidate bundle.
7. Evaluate registered sealed test once. Finalize immutable report/model card/bundle; leave deployment selection for an explicit operator action.

A small smoke dataset is tagged `fixture_only` and can exercise mechanics without passing research support gates. It cannot produce a promotable artifact.

## Failure recovery

| Failure point | Resume action |
|---|---|
| Teacher timeout after send | record uncertain attempt, bounded retry with new ID; dedup before acceptance |
| Invalid reviewer JSON | quarantine attempt; optional one corrective request, never bool/string coercion |
| Process dies while writing response | ignore incomplete temp file; retain journal; replay completed hashes only |
| Freeze interrupted | discard incomplete destination, revalidate source records and regenerate byte-identical manifest content using the journaled created_at |
| Training dies mid-epoch | resume last complete checkpoint; report discarded progress |
| Calibration fails/insufficient support | invalid temperature fit: diagnostic report only; valid fit but no gate: accept-none research bundle; no threshold guessing |
| Evaluation missing outputs | mark invalid_run; do not compute success from remaining records |
| Main branch changes during design/implementation | rebase/review relevant changes; do not overwrite unrelated work |
| Bundle load fails | keep previous caller engine active; report error |

Re-run idempotency uses content hashes and stage completion markers, not merely file existence. Immutable IDs refuse conflicting contents. Interrupted report output is regenerated atomically. A file lock protects each writable run; stale lock recovery verifies that the owner process is gone before operator-approved release.

## Limits and observability

Default teacher limits: concurrency 1, 8 lessons/request (max32), 500 HTTP attempts or 2h/cycle, 10,000 accepted records/cycle, max3 cycles. At these settings a cycle may stop before the 10,000-record ceiling; the first 10,000-record dataset can be accumulated from several audited runs. Exhaustion creates a resumable `budget_exhausted` report, not an endless retry. No automatic paid endpoint substitution.

Host inference default deadline 5s/request for local tooling, configurable; the research latency gate is separately 100ms p95 on the recorded CPU. Limit queue to 32 pending requests per worker, reject overload; batch evaluation is not a user-serving queue. Log stage/run/request IDs, statuses, durations, counts, hashes, retries and resource peaks. Logs must not include raw request state or secrets by default. Include unsuccessful attempts in throughput/cost reports.

Stop a cycle at explicit cancellation and finish only atomic write boundaries. Education never auto-promotes a model, mutates production weights or changes Runtime routing policy.

## Available controlled experiment command

`python -m arbitrium_education.experiment generate` and `run` are implemented.
They are a bounded offline research workflow, not the planned general education
CLI or I14 teacher cycle. See [Experiment 001](experiment-001.md).
