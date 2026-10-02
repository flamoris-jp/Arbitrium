# Public-architecture migration comparison

Measured on 2026-10-02. These are new observations under the pinned Maidionis Core reference implementation, separate from the [immutable original reports](../legacy/docs/experiments). The [preregistered plan](plan.json) fixes four curricula, seeds 42/43/44, hyperparameters and selection scopes. [Full results](results.json) retain every loss history, metric support/recall/confusion, environment/config/dataset/component digest and actual load receipt.

| Curriculum | Scope | Seed | Train decision accuracy | Dev decision accuracy | Dev unanswerable recall | Best epoch (zero-based) |
|---|---|---:|---:|---:|---:|---:|
| recovery-current-tiny-v1 | train_diagnostic | 42 | 100.00% | not evaluated (empty dev) | not evaluated | 199 |
| recovery-current-tiny-v1 | train_diagnostic | 43 | 100.00% | not evaluated (empty dev) | not evaluated | 195 |
| recovery-current-tiny-v1 | train_diagnostic | 44 | 100.00% | not evaluated (empty dev) | not evaluated | 195 |
| recovery-current-diagnostic-v1 | train_diagnostic | 42 | 100.00% | not evaluated (empty dev) | not evaluated | 118 |
| recovery-current-diagnostic-v1 | train_diagnostic | 43 | 100.00% | not evaluated (empty dev) | not evaluated | 119 |
| recovery-current-diagnostic-v1 | train_diagnostic | 44 | 100.00% | not evaluated (empty dev) | not evaluated | 119 |
| recovery-current-grouped-v1 | dev | 42 | 92.31% | 70.00% | 0.00% | 18 |
| recovery-current-grouped-v1 | dev | 43 | 96.15% | 60.00% | 25.00% | 21 |
| recovery-current-grouped-v1 | dev | 44 | 92.31% | 80.00% | 0.00% | 17 |
| controlled-recovery-1000-v1 | dev | 42 | 54.77% | 45.00% | 0.00% | 0 |
| controlled-recovery-1000-v1 | dev | 43 | 35.50% | 35.00% | 0.00% | 1 |
| controlled-recovery-1000-v1 | dev | 44 | 45.23% | 45.00% | 0.00% | 1 |

## Meaning and denominators

Decision accuracy uses only answerable inputs. Answerability metrics use all inputs; unanswerable recall uses the actual unanswerable support. Zero support is undefined, not zero or perfect. The 72-row fixture has 10 answerable and 4 unanswerable dev rows, with label support retry/fallback/stop = 3/3/4. The 1,000-row fixture has 60 answerable and 28 unanswerable dev rows, with label support 14/21/25. All original split allocations/counts are preserved in new-format derivatives.

Train-only 24/96 results demonstrate bounded fitting, not unseen-problem performance. The 72/1,000 dev set is used for epoch selection, so its reported result is a development diagnostic rather than an untouched held-out test. Public fixture holdouts are not sealed benchmarks. Calibration/test payloads were not used for training, codec vocabulary, selection or measurement.

The new 96-row diagnostic fits minority labels in this profile, unlike the historical 25% retry-recall observation. The 72-row development accuracy is higher than the old 40/30/20%, but only 10 answerable dev rows are available. Answerability remains unresolved: 0/1/0 of 4 unanswerable dev inputs are recognized for seeds 42/43/44. The 1,000-row dev recognizes 0 of 28 for every seed. Successful fitting does not conceal these failures.

## Comparison boundary

- Train-only ASCII word tokenizer replaces SentencePiece; fixed fields/control tokens retained.
- Core seeded default initialization, one joint four-output layer, explicit module decay roles, SHA-256 epoch ordering and linear-decay schedule replace legacy implementations.
- LibTorch 2.5.1+cpu replaces recorded 2.14.0+cpu. No bitwise or score equality claimed.

The codec, initialization/head implementation, optimizer decay roles/order/schedule and toolchain differ together. No exact old-weight/score parity or causal attribution is claimed. Brier/ECE remain uncalibrated diagnostics. Every finalized artifact is `research_only`; every evaluation summary has `passing=false`. The exported reference composition abstains on advisory output because calibrated serving support is absent.

## Actual lifecycle evidence

All twelve runs changed parameters with nonzero gradients, exported selected-best, passed metadata-only validation with zero model constructions, loaded a real archive in a disposable 2 GiB host, matched selected-best logits within 1e-6, accounted for the whole selected split, rejected a missing prediction and a serving attempt, finalized a new immutable research artifact, and reloaded with identical outputs.

Checkpoint fresh-process resume is separately tested bit-exactly by `tests/test_native.py`. No inference substitutes the in-memory pre-export model for archive loading. No teacher, network model, deployment or Runtime R1 was used.
