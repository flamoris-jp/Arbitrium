# Supported offline build and verification

The supported profile is Linux x86_64, Python 3.12, C++20, OpenSSL development
headers, CPU FP32 and LibTorch 2.5.1+cpu. Dependencies are installed before the
verification tools; build/verification themselves download nothing.

Use Arbitrium's PR checkout and the exact Core checkout pinned by
`.github/workflows/decision-ci.yml`. The content digest is enforced independently
of the Git branch name. Maidionis PR #6 must be reviewed/merged before this PR.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ../Maidionis/tools/dependencies.txt
python -m pip install numpy==2.1.3
python -m pip install --no-deps --no-build-isolation ../Maidionis
python -m pip install --no-deps --no-build-isolation -e .
python tools/verify.py --maidionis-root ../Maidionis
```

`tools/verify.py` checks the pinned dependency and immutable research archive,
generates a compiled approved composition, builds the actual native driver, and
runs native binding and Python/native contract, integrity, codec, gate, metric,
resume and substitution tests. It does not execute archive code as the new model.

A quick actual lifecycle check:

```bash
python tools/run_experiments.py --output runs/smoke --smoke
```

The fixed full comparison (four curricula × three seeds):

```bash
python tools/run_experiments.py --output runs/comparison
```

Choose a new output directory for every run. Outputs/checkpoints are immutable;
interruptions leave evidence for inspection. A failed native operation returns
nonzero. There is no automatic retry, source replacement, run-root overwrite,
promotion or deployment. The runner uses finite subprocess/address-space/CPU
bounds. It requires no API key, live teacher, service or GPU.

Historical instructions under `research/legacy` are archived observations, not
current setup instructions. No binary/model weights are committed. Retain your
run outputs if you need to reproduce their exact artifact/evidence digests.
