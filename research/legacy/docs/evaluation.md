# Evaluation and release gates

## Dataset roles and authority

Train teaches weights and tokenizer. Dev chooses model/checkpoint and supplies aggregate weak-concept feedback. Calibration_fit fits temperatures. Calibration_select selects gates. Test is sealed and used for one registered candidate after all choices are fixed. Python orchestrates; C++ performs inference and primitive metric accumulation; Python may independently recompute metrics for validation and aggregate/bootstrap reports, but never implement another model.

Sealed access ledger records experiment ID, candidate/config/gate hashes, evaluator revision, timestamp, reason and report hash. Ordinary training/teacher jobs receive no test path. A release evaluation can compare the preregistered pooled and encoder candidates, but the winner is selected on dev beforehand. Test is not a competition from which to choose the best of many attempts. A failed release test ends that experiment. Aggregate results may inform the next experiment, but reuse must be called adaptive; independent release claims require a newly sealed suite. If exact errors are opened for diagnosis, retire that suite immediately.

## Prediction record

C++ evaluation writes one JSONL record per sample: schema_version `arbitrium.prediction.v1`, run/sample/task IDs, artifact/dataset IDs, split, target, decision logits (canonical label order), answerability logit, DecisionResult, elapsed inference microseconds. Target is joined outside forward and never included in model input. Hash and count the prediction file. Missing/duplicate IDs, error results or nonfinite data invalidate a release run rather than shrinking its denominator. Error rate is still reported.

## Metric definitions

Let A be answerable records, C accepted records, K classes. Always state n, denominator and exclusions.

| Metric | Definition |
|---|---|
| Raw accuracy / macro-F1 | modal class on A, independent of abstention; absent-class F1=0 with warning |
| NLL | mean -log(max(p_y,1e-12)) on A |
| Multiclass Brier | mean sum_k (p_k-onehot_yk)^2 on A; unnormalized, range [0,2] |
| Binary Brier | same 2-class Brier convention for comparability; optional positive-class Brier separately named |
| ECE | top-label confidence, 15 equal-width bins on [0,1], mean weighted abs(accuracy-confidence) over A; last bin includes 1 |
| Answerability | AUROC if both labels exist, BCE, scalar Brier mean(q-a)^2, confusion at tau_q over all records |
| Coverage | |C| / all valid records |
| Selective risk | (wrong accepted answerable + all accepted unanswerable) / |C|; undefined if |C|=0 |
| False acceptance | accepted unanswerable / all unanswerable |
| High-confidence error | modal wrong at p>=0.80 on A; report count/rate and accepted subset |
| Ordinal MAE | mean abs(sum(p*anchor)-target_anchor) on A; also modal accuracy |
| Ranked probability score | mean over A of sum over j<K-1 of (CDF_p(j)-CDF_y(j))² / (K-1) |

Slice by concept, difficulty, verification source, label, length and answerability. Do not invent confidence for sparse slices: fewer than 50 records is descriptive only. Reliability bins include sample counts. C++ and Python aggregators must agree on hand-calculable fixtures, including empty accepted sets, ties and zero-support classes.

## Preregistered research gates

These thresholds are project acceptance defaults, not claimed measurements or universal safety standards. Changing them creates a new experiment registration before a new sealed test.

| Gate | Required result |
|---|---|
| Contract integrity | 0 malformed/corrupt accepted inputs or missing results; all data/artifact integrity checks pass |
| Independent support | test >=1000, >=100 per decision label, >=200 unanswerable, target >=20 held-out template families and require >=10; independently authored subset >=200 |
| Semantic usefulness | selected model test macro-F1 on answerable >=0.80 and >=0.05 above frozen deployable rule/majority best |
| Encoder necessity | architecture is selected by dev-only paired family-bootstrap CI before calibration/test; report the preregistered comparison on test without switching the candidate |
| Calibration | test ECE<=0.05; calibrated Brier and NLL no worse than raw by more than 0.01 absolute |
| Abstention | coverage>=0.50, accepted n>=200; one-sided 95% Wilson upper selective-risk bound<=0.05; same formula as calibration |
| Ambiguity | false acceptance on unanswerable <=0.05 |
| Source robustness | independently authored subset raw accuracy>=0.75; disclose teacher-generated comparison |
| Latency/memory | explicit reference CPU profile; warm B=1 p95<=100 ms at L=256, peak inference RSS<=1 GiB |

Binary extension uses same class gates; ordinal extension additionally MAE<=0.15 on the normalized 0..1 range and RPS improvement over uniform distribution. An artifact only advertises the task whose evidence passed. Lack of encoder advantage does not invalidate the project: a pooled artifact selected on dev may be the research outcome. Never switch to it because the selected encoder failed sealed test. Neither implementation completion nor an internal experiment gate constitutes automatic production permission.

Family bootstrap: sample families with replacement, include all their records, 2000 replicates, seed 314159, percentile 2.5/97.5% CI on paired accuracy differences. Dev bootstrap intervals used after seed/checkpoint selection are explicitly model-selection heuristics and are not reported as independent confirmatory confidence intervals. Resample both models together. Single-model accuracy and all-seed summaries also get intervals; Wilson selective bound is a reported binomial criterion and does not account for all within-family correlations. Include family-bootstrap selective-risk intervals as a sensitivity analysis; flag conflicting conclusions and withhold promotion pending more independent families.

## Performance and numerical verification

Record CPU model/core count, OS, compiler flags, exact LibTorch, thread count, device, dtype, actual length histogram and batch size. Separate load time, tokenization, forward, calibration and total request latency. Warm up 20 calls; measure 1000 calls with lengths 32,128,256; report median/p95 and throughput for B=1 and B=32. Measure process peak RSS; training memory is separate. GPU results are separate profiles, never compared without disclosure.

Inference parity gate: same-build CPU save/load and batch-vs-single p/logit absolute tolerance 1e-5, relative 1e-4. Cross-device profile initially absolute 1e-4, relative 1e-3 plus identical status and verdict on release fixtures; if near-threshold changes occur, that profile needs its own calibration and release evidence. Never weaken tolerances silently. Tensor forward tests do not count as language capability tests.

## Final report

Include registration, dataset split/hash and audits, all seeds/checkpoints, baseline comparison, raw/calibrated/slice metrics, coverage-risk curve, latency/RSS, abstention examples, failures, known limitations, artifact digest, reproducible commands and conclusion (`promotable`, `research_only`, or `invalid_run`). No benchmark examples are fed to the teacher merely because a report exists.
