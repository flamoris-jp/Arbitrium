# A1–A3 review and regression record

Reviewed the current main and both open PR heads, source-backed implementation,
companion Core partition amendment, complete archive integrity, supported build,
dataset/codec/objective/provider boundaries and experiment accounting.

## Findings fixed

| Finding | Fix and regression |
|---|---|
| Python admitted renamed legacy family aliases while Native rejected them. | Bind original family and provenance IDs; reject rehashed alias derivatives. |
| Native source-row validation did not bind the original provenance ID. | Compile source provenance IDs and reject a rehashed train record whose provenance alias was replaced. The regression first reproduced acceptance before the fix. |
| Python `Decision.validate` admitted a relabeled `registered_evaluation` purpose. | Require `research_fixture`, matching the Native partition adapter. Public historical fixtures never gain sealed/release support. |
| Legacy dataset name could select an arbitrary path outside the inventoried archive. | Admit only the four dataset names in the content-pinned dependency lock before any name-derived access. |
| Large finite logits could overflow scaled softmax into NaN. | Reject logits outside the registered ±1,000,000 raw schema bound in Python and Native; reject nonnumeric/Boolean/out-of-range temperatures. |
| Provider schema-only checks skipped canonical policy/token limits and allowed an uncalibrated `ok` envelope. | Require the explicit Decision composition, validate its codec before host invocation, enforce abstention, probability coverage/order/normalization and a trusted artifact digest. |
| A host could mutate the request that was subsequently used to bind its result. | Keep an independent request snapshot; reject rebinding. |
| FP32 native probability rounding turned distinct near-tie logits into a canonical tie, disagreeing with Python. | FP64 postprocessing plus exact-tie/near-tie regressions; CPU FP32 training is unchanged. |
| Native CLI usage named the wrong executable and omitted the config argument. | Correct the actual Arbitrium command contract. |

## Re-review boundary

All 19 Python/native tests and the compiled binding CTest must pass without skips.
The native composition regression exercises all-unanswerable loss masking with
invalid categorical targets, zero categorical gradients, positive answerability
gradients, ties, near ties and nonfinite/out-of-bound decode rejection. The actual
lifecycle smoke trains, resumes in the contract tests, exports, admits and reloads
selected weights, validates actual native envelopes through the separate provider,
evaluates every selected sample, rejects missing predictions and serving, and
finalizes/reloads research-only outputs. Hosted Decision CI repeats the supported
tests and actual smoke. Companion Maidionis #6 remains unchanged.

Historical files and preregistered full-comparison reports are not rewritten.
Their original build and evidence identities remain attached to the measured
snapshot, not to the revised head. The new tests/smoke qualify the reviewed
offline mechanics only. Weak unanswerable recognition, selection-biased dev
results, absent calibration, no Runtime R1 and no production release remain
explicit limitations, not issues claimed fixed by this review.
