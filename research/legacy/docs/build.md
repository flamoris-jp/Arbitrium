# C++ Build Bootstrap

## Requirements

- C++20-capable compiler
- CMake 3.24 or newer
- LibTorch development package

Arbitrium does not vendor LibTorch.

Point CMake at the local LibTorch package through `CMAKE_PREFIX_PATH`.

## Configure

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

For a Release build:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

On multi-config generators such as Visual Studio, choose the configuration at build time.

## Build

```sh
cmake --build build --config Release
```

## Test

```sh
ctest --test-dir build -C Release --output-on-failure
```

## Smoke executable

The `arbitrium_smoke` target constructs the Phase 0 baseline from random initialization and performs one synthetic decision.

It does **not** prove language understanding or model quality.

Its purpose is only to prove:

- native C++ build;
- LibTorch linkage;
- model construction;
- forward inference;
- normalized decision probabilities.

## Dependency pinning

Do not silently pin an arbitrary LibTorch release in documentation.

When the first training implementation is introduced, record the actually tested toolchain and LibTorch version in the experiment metadata and CI/build documentation.

## Design v1 migration status

The commands above describe the existing CMake smoke targets only. `arbitrium train`, calibration, evaluation, artifact serving and education CLI commands in the design are **planned** and are not present yet. No successful build on a particular toolchain is asserted by this documentation-only change.

Implementation package I01 in [the implementation plan](implementation-plan.md) adds native tokenizer/JSON/hash dependencies, fixes Python package-relative paths and records exact tested versions. Do not infer a working install from the current Python scaffold's `pyproject.toml`. See [dependency decisions](dependencies.md) for the selected boundaries.
