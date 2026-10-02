# Calibration and abstention

Calibration is a fitted postprocessing layer, not a property guaranteed by softmax. It is implemented in C++ and bound to exact weights/task/tokenizer/assembly hashes.

## Two independent fits

After choosing weights with dev only, run them in eval/no-grad mode on calibration_fit. Fit scalar `T_dec > 0` minimizing categorical NLL on answerable records; `p=softmax(z/T_dec)`. Fit scalar `T_ans > 0` minimizing binary NLL on all records; `q=sigmoid(a/T_ans)`. The latter predicts answerability under the dataset's reviewed definition. Do not multiply q*p and call it calibrated joint correctness.

Fit log-temperature in [-4,4] using deterministic bounded scalar minimization (golden-section search, 200 iterations maximum, stop when interval width <1e-6). Use FP64 objective accumulation. Include T=1 as an explicit candidate and keep it if optimization does not lower NLL. Store fit counts, objective before/after, boundary-hit and convergence status. Insufficient coverage, nonfinite logits, or nonconvergence blocks candidate release. No vector/isotonic/Platt model is an unresolved choice in v1; scalar temperature is the fixed baseline.

Calibration_fit minimums and label balance are in [dataset](dataset-design.md). Calibrating a score means calibrating the ordinal-bin distribution; its mean is not a probability. Fit temperatures independently for each task artifact.

## Gate selection

Using calibration_select only, evaluate `(tau_q,tau_p)` on grid `{0.50,0.55,...,0.95,0.99}` squared, where accept iff `q>=tau_q` and `max(p)>=tau_p`. Every unanswerable accepted sample counts as an error; on answerable samples, wrong modal class counts as error. For ordinal tasks, wrong modal anchor also counts as an error; mean-score MAE is reported separately.

For each pair compute accepted n and errors e. Require n>=200, coverage>=0.50 and one-sided 95% Wilson upper error bound <=0.05. Use z=1.6448536269514722 and upper bound `(r+z²/(2n)+z*sqrt(r*(1-r)/n+z²/(4n²)))/(1+z²/n)`, r=e/n. Select highest coverage, then lower upper bound, then higher tau_q, then higher tau_p. Grid selection is adaptive, so this bound is a tuning criterion, **not** a postselection statistical guarantee. Independent sealed testing is still required.

If no pair passes, result is `research_only` and no promotable calibration is emitted. Keep diagnostics and a disabled gate (`accept_none=true`) for research evaluation. Never lower gates using test performance. Gate policy version and grid are recorded.

## Serving semantics

1. Validate request and bundle compatibility.
2. Compute logits; fail on nonfinite values.
3. Apply recorded temperatures.
4. If accept_none=true: abstain / release_gate_not_met (research bundle only).
5. Else if q < tau_q: abstain / low_answerability.
6. Else if max(p) < tau_p: abstain / low_confidence.
7. Else return ok and typed verdict.

No missing or unknown calibration artifact is allowed in normal serving. Research commands may evaluate raw logits explicitly, recording `uncalibrated` in reports; they cannot masquerade as calibrated DecisionResult. A bundle with failed gate selection accepts none. A candidate awaiting sealed testing can remain research_only while retaining its successfully selected gates, so its behavior can be evaluated. A production adapter accepts only finalized promotable bundles. Invalid temperature fits produce a diagnostic report only, not a calibrated bundle. Changing any weight, tokenizer, task label/rubric, assembly, precision profile, or quantization invalidates calibration and demands reevaluation; a merely changed request label order does not.

## Evaluation limits

Measure NLL, Brier, ECE and reliability bins before/after calibration on dev descriptively and sealed test once. Temperature scaling preserves class order; it cannot repair a wrong class, detect every unsupported English statement, or guarantee reliability after distribution shift. Answerability examples improve a bounded behavior but do not prove semantic understanding. Report support counts and covered-domain limits with every result.
