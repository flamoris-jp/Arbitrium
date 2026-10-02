# Experiment 001 — controlled recovery education

## Purpose and boundaries

Run one offline education experiment end to end: validated controlled English →
family-separated frozen data → train-only SentencePiece → C++ pooled training →
dev report for seeds 42, 43 and 44. Python orchestrates; only C++ implements and
trains the model. GPT-OSS is not called. Nothing is deployed or promoted.

SPEC-001 is resolved by aligning the example TaskSpec and request/sample policy
with the normative healthy-or-transient retry rule. Fatal violations still take
precedence. This changes the task content hash: old datasets/tokenizers/checkpoints
must not be reused as if the policy were unchanged.

## Included sample

`examples/education/controlled-recovery-1000-v1/` contains **1,000 actual JSONL
samples**, their task, family registry, provenance and hash manifest.

| Target | Records |
|---|---:|
| retry | 250 |
| fallback | 250 |
| stop | 250 |
| unanswerable (null label) | 250 |

50 of the unanswerable examples contain explicit simultaneous contradictory
reports. All examples distinguish earlier observations from current observations;
current observations explicitly supersede earlier ones. The oracle applies to
current structured facts. Historical variants of the same scenario and each
phrasing skeleton stay together. The generator retains those source facts and
its own/verifier SHA-256 in provenance. No LLM review or human audit is claimed.

| Split | Records | Use in this runner |
|---|---:|---|
| train | 712 | Tokenizer and model weights |
| dev | 88 | Early stopping, epoch selection and report |
| calibration_fit | 49 | Reserved; not read |
| calibration_select | 51 | Reserved; not read |
| test | 100 | Reserved; not read; public, **not sealed** |

This is 1,000 total samples, not 1,000 training rows plus evaluation data. Actual
split sizes follow the existing family hash; families are not relocated for
convenient quotas. This small fixture does not satisfy production support gates.
There are 40 related wording skeletons, and repeated base scenarios are not
independent evidence. Training can succeed computationally without demonstrating
robust language understanding. A pooled model also lacks temporal token order.

## Build and run

Install the native prerequisites in [build.md](build.md), including LibTorch,
SentencePiece, Abseil, OpenSSL and nlohmann/json, then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ARBITRIUM_DEPENDENCY_PREFIXES" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$ARBITRIUM_JSON_INCLUDE"
cmake --build build -j 2
ctest --test-dir build --output-on-failure
python -m venv .venv
.venv/bin/python -m pip install 'education/python[dev]'
.venv/bin/python -m pytest education/python/tests -q
.venv/bin/python -m arbitrium_education.experiment run \
  --dataset examples/education/controlled-recovery-1000-v1 \
  --output runs/experiment-001 \
  --trainer build/arbitrium_experiment
```

Output directories are immutable: select a new path for another run. Failures
return nonzero and do not invent a successful summary. This first runner does
not automatically resume an interrupted education cycle. Its epoch checkpoints
use the existing partial I09 implementation; full release inventory and serving
remain out of scope.

The native runner uses the seed-owning model factory, default pooled model and
TrainingConfig (20 epochs, batch 64, LR 0.0003, dropout 0.1, CPU FP32, one thread).
It reloads the epoch with the best dev objective before reporting. Reports include
loss history, initial/selected accuracy, macro-F1, Brier/ECE, answerability metrics,
a train-majority baseline, parameter count and toolchain. Probabilities are **not
calibrated**; a 0.5 answerability diagnostic is not a production acceptance gate.

## Regenerate the sample

```sh
.venv/bin/python -m arbitrium_education.experiment generate \
  --task docs/examples/recovery-task.json \
  --output datasets/frozen
```

The generated records and provenance are deterministic for the same generator
bytes. A fresh freeze allocates a new timestamp, so its manifest/dataset digest
changes. The committed fixture's manifest remains authoritative. Tests regenerate
all 1,000 records and compare their exact content/provenance; they also check
balance, source scenario separation, policy precedence and corrupted train hashes.

No weights, tokenizer binaries, build outputs or private data are committed.
Research reports do not use the unresolved release-format schemas from SPEC-004/005.

## Measured result (2026-09-30)

[Full three-seed report](experiments/experiment-001-results.json).

| Seed | Selected epoch (zero-based) | Dev decision accuracy | Macro-F1 |
|---|---:|---:|---:|
| 42 | 5 | 35.00% | 0.1728 |
| 43 | 5 | 23.33% | 0.1261 |
| 44 | 5 | 35.00% | 0.1728 |

Accuracy/F1 use the 60 answerable dev examples; answerability uses all 88.
The train-majority baseline scores 35.00%, and uniform random expectation is
33.33%. **This run does not demonstrate useful decision learning.** The pipeline
works, but model quality is below the experimental goal. Do not promote a model
or claim the education succeeded from the loss decrease or working CLI alone.
The small sample, limited scenario diversity, optimization and pooled model's
inability to distinguish temporal order remain possible factors; this experiment
does not isolate a cause. Calibration and encoder comparison remain pending.

Validation: complete native build, CTest **6/6**, installed-package Python
**42/42**, then the full three-seed run on the committed fixture. Measured profile:
Linux x86-64, GCC 13.3.0, CMake 4.4.3, LibTorch 2.14.0+cpu, SentencePiece 0.2.2,
Abseil 20260526 and Python 3.12.14. This differs from the repair report's older
profile. CMake conditionally links `absl::status_builder` when available for the
newer SentencePiece build, without requiring that target on older Abseil.
