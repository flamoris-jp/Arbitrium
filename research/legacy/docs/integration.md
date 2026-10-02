# FLAMORIS integration boundary

Arbitrium returns advisory judgments over supplied English evidence. It does not own workflow state, retries, routing, credentials, tool execution, GPU selection or product edits.

## Core adapter contract

A prospective `flamoris-intelligence-mcp` adapter prepares a registered-task DecisionRequest and calls an already-loaded C++ worker over JSONL or links the C++ API. This is a proposed boundary only; the current external repository/API has not been inspected or changed for this design. No claim of plug-in compatibility is made. Actual external tool naming/versioning belongs in its own implementation PR after that repository is inspected.

Suggested capability semantics: `evaluate_decision` accepts the request shape in [interfaces](interfaces.md); artifact selection is a server-side configured task mapping, never an arbitrary path from the client. Production adapter loads only finalized promotable bundles; research tooling may evaluate candidates explicitly. Response includes artifact/calibration IDs and status. MCP transport/envelopes/authentication live entirely outside this repo's core library.

## Caller sequence

1. Enforce authorization, hard safety constraints, resource limits and deterministic policy first.
2. If reliable structured facts suffice, decide in code without Arbitrium.
3. Otherwise normalize/redact a bounded evidence snapshot, select a supported task, supply its canonical question and valid policy scope.
4. Apply caller deadline; validate returned IDs/status. `error` means operational failure; `abstain` means no model verdict. Route either according to an explicit caller policy (ask a human, use a larger model, or stop), without misrepresenting it as Arbitrium's `stop` judgment.
5. For `ok`, recheck current state, allowed choices, task policy and evidence freshness before executing anything. A stale advisory response cannot authorize an action.
6. Log request/artifact/result identities and observed outcomes with privacy controls. Logging is opt-in for later education, never automatic learning.

No retry loop is built into the model. No teacher is called because a result has low confidence. The host may choose another evaluator, but that is a different component and is labeled as such.

## Task changes

New label IDs, question/rubric meaning, unsupported policy dialect or ordinal range require a new TaskSpec and model artifact. Reordering all existing choice IDs is allowed and must preserve keyed results. Reducing the available choice set is not a compatible request: the caller can express unavailable options as evidence under the registered policy, then enforce them again after inference. Calibration is not transferable to new semantics.

## Oblivionis

A future bridge may supply a bounded, versioned, sanitized state summary as evidence. Snapshot source/version/freshness is tracked by caller logs, not interpreted as model authority. Arbitrium never reads or mutates Oblivionis memory. A new state representation requires its own task/dataset evaluation before operational use.

## Integration proof acceptance

Use a local fixture caller with a fake action executor: one supported ok case, one abstain, one corrupt artifact, one invalid task, one timeout and one state-changed-after-request case. Only a policy-approved ok result may reach the fake executor. Demonstrate operation with Python education and GPT-OSS server absent. Real Runtime integration is a separately approved follow-on; this repository may finish its interface implementation without changing external repositories.
