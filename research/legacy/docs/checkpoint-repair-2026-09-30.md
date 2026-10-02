# PR #3 checkpoint repair — 2026-09-30

## Scope and provenance

Checkpoint `4888e687` deleted reviewed design files and first-checkpoint Python
files. At `db73b0b3f15286932c399c6ea5f5961a791497ec`, imports and tests still
required those files. A repair was appended to the existing branch history;
no commits were reset, rebased, amended or force-pushed.

Restoration commit: `d82d3a761fa28245f5c82865aa370925698bfd58`.
Its complete tree is `c925d3f49136cb1c9f4ddb893eabecbf60ef263f`.

- 37 missing files were restored from reviewed main
  `d76b747c46fff0a70fde15523fc88ccc254b999f`.
- 7 further missing files were restored from first implementation checkpoint
  `6c2245567fbf2a7fc9499416b8c3de3822a29814`: `contracts.py`, `oracle.py`, four
  packaged schemas and `test_contracts_oracle.py`.
- All 44 restored blobs match their source Git object IDs exactly.
- All 38 files already present at the previous PR head remain byte-identical
  in the restoration commit. The repaired snapshot contains 82 files.
- The main-to-PR comparison has no deleted files.

The restoration commit message's 36/8 source split was a counting typo;
the audited split above is 37/7. The total of 44 and the restored tree are correct.

Clean CMake configuration also exposed a link to nonexistent
`absl::status_builder` in the tested Abseil release. Removing that unused target
fixes configuration without changing model or education behavior.

## Verification environment

Linux CPU reference, Ubuntu 24.04, GCC 13.3.0, C++20, CMake 4.4.3, Ninja 1.13.2,
LibTorch 2.8.0+cpu (C++ headers and libraries from the official CPU wheel),
OpenSSL 3.0.13, Python 3.12. This does not introduce Python torch as model
implementation authority.

Source dependencies:

- SentencePiece v0.2.1: `31646a467d2051eb904e0b45de3a73e91fe1c1e3`,
  static build with its internal Abseil provider.
- Abseil 20250127.1: `d9e4955c65cd4367dd6bf46f4ccb8cd3d100540b`.
- nlohmann/json v3.11.3: `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`.

Python tests use a fresh venv, SentencePiece 0.2.1, pytest 9.1.1 and a normal
wheel/package installation, without editable install or PYTHONPATH overrides.
Installed `contracts`, `oracle` and all four packaged schemas were checked.

Results from the clean checkout with the CMake repair:

- Complete CMake configure and build passed.
- CTest: 6/6 passed (model, contracts, data, metrics, training/resume,
  postprocessing).
- Python: 39/39 passed with the installed package. SentencePiece's SWIG wrapper
  emitted two deprecation warnings; no test failed.
- All four documentation/package schema pairs are byte-identical.
- The fresh checkout had no untracked source files or tracked modifications.

Validation commands, after provisioning the dependencies above:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ARBITRIUM_DEPENDENCY_PREFIXES" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$ARBITRIUM_JSON_INCLUDE"
cmake --build build -j 2
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ctest --test-dir build --output-on-failure
python -m venv .venv
.venv/bin/python -m pip install 'education/python[dev]' 'sentencepiece==0.2.1'
.venv/bin/python -m pytest education/python/tests -q
```

The dependency prefixes must resolve the recorded LibTorch/Abseil packages and
SentencePiece installation. Dependencies were provisioned outside the source
checkout; no downloaded weights, libraries or build outputs were committed.

## Safe checkpoint publication

The Git history proves deletion in checkpoint 2; it does not prove which tool
or command created the incomplete tree. Preserve complete snapshots explicitly:

1. Read the current branch head and its tree before publishing a checkpoint.
2. With the GitHub Git Trees API, always supply that tree as `base_tree` when
   applying partial file changes. Omitting it constructs a new tree containing
   only the supplied entries. Do not publish such a partial tree as the whole
   repository.
3. Review the complete old/new tree and PR diff. Unintended deletions are a
   blocking error, including files introduced by earlier checkpoints.
4. Create the commit with the observed head as its parent. Recheck branch head
   before updating the ref, use a non-force update, and stop if it advanced.
5. Test the committed snapshot in a separate clean checkout with a fresh build
   directory and installed Python package. Local untracked files are not evidence.

No automatic GitHub Actions workflow was added. Local clean-checkout validation
is sufficient for this repair and does not change the existing Actions policy.

## Limits

This repair preserves PR #1's design. It does not complete the production
dataset gates, checkpoint inventory/publication, or release evaluation.
The review's model-initialization seed issue remains a separate implementation
follow-up; the current training API receives an already constructed model.
