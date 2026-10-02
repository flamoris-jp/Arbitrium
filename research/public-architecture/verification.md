# Verification evidence — 2026-10-02

- Research source: all 136 source Git blobs inspected/copied locally and checked
  against their exact original Git blob identities. The public archive contains
  130 approved originals plus explicit annotation/inventory. Full original
  datasets/audits/reports were examined; no original hashed member was edited.
- The 130-file SHA-256/byte inventory is pinned by `dependency.lock.json` and
  checked before generating the compiled native composition. Public text scan
  found no non-public repository coordinates, private host names/addresses,
  personal email or deployment home paths in the deliverable.
- Maidionis companion PR #6: local Linux CPU verification passed 4 native tests
  and 37 Python tests, including all 12 A21 tests without skips. Hosted
  [Core CI](https://github.com/flamoris-jp/Maidionis/actions/runs/36998088127)
  succeeded at `09b119f4315c5742d075e63669daabe5d0fc55ac`, including numerical,
  schema regeneration and reduced dependency profiles.
- Arbitrium: native compiled-source binding test and all 10 Python/native tests
  passed without skips through `tools/verify.py`. These cover archive/family
  integrity, oracle unknown/conflict/precedence cases, native/Python codec parity,
  request order and bounds, gate ties/precedence, undefined metric support,
  fresh-process bit-exact resume and rehashed target/profile substitution.
- A3: the full 12-run local comparison completed, not merely `--smoke`. All
  selected-split evaluation reports are complete and all summaries retain
  `passing=false`. Actual candidate/final archive loading and numerical parity,
  missing-prediction rejection and serving rejection succeeded for every run.
- GitHub Decision CI additionally runs the supported build/tests and a finite
  actual two-epoch lifecycle smoke. Its result is reported on the PR; it is not
  the evidence for the already-recorded full local 12-run comparison.

The exact measured numerical environment, build/codec/config identities and
metric denominators are in `results.json`. Original binary checkpoints/tokenizers
are not supplied. This qualifies the initial offline reference mechanics only:
not sealed generalization, calibrated serving, GPU compatibility or Runtime
holder/resource transfer.

## Review hardening

The full 12-run results above are immutable observations from the implementation
snapshot published at `b26d5d66bfdeff2f5a1991effcdef0a5a730e2b4`.
Subsequent review fixes change the content-derived build identity, so the existing
artifact/evidence digests must not be relabeled as results for the revised code.
The revised head is separately verified with 19 Python/native tests and an actual
two-epoch lifecycle smoke; see `docs/review-a1-a3.md`. No new full 12-run numerical
comparison or model-quality improvement is claimed by those boundary fixes.
