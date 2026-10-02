# Dependency decisions

C++ implements the mathematical model; external libraries implement numerical and format infrastructure. No pretrained weights or pretrained vocabulary are imported.

| Dependency | Adopted responsibility | Project-owned responsibility |
|---|---|---|
| LibTorch C++ | tensors, autograd, nn primitives, AdamW, kernels, native archives | attention/encoder composition, losses/loop, checkpoint semantics, calibration and inference |
| SentencePiece C++ / trainer | BPE algorithm and native model parsing | train-only corpus, options, special tokens, input assembly and artifact binding |
| nlohmann/json | strict JSON parsing/serialization | semantic validation, duplicate-key rejection, schema/version behavior |
| OpenSSL libcrypto EVP | SHA-256 file/content integrity | manifest identities, trusted expected digests and provenance policy |
| Python HTTPX | teacher HTTP with bounded transport | prompts, capability profile, retry/error semantics |
| Pydantic v2 / PyYAML | strict education records / human-authored config | schema authority, validation rules, no type coercion or unsafe YAML tags |
| Python jsonschema (dev) | validate design/runtime fixtures against schemas | semantic cross-checks and contract parity |
| pytest / Ruff | Python test/lint tooling | meaningful offline/fault and contract tests |
| llama.cpp + GPT-OSS | operator-managed external teacher inference | lesson policy, review, verification and audit |

Use system/operator-provided dependencies via CMake find_package with explicit versions/build digests in a tested lock manifest. Do not fetch moving branches at configure time. Exact versions are pinned in implementation bootstrap after a successful clean build; the inspected repository has not provided such evidence. This is an environmental validation gate, not an undecided model-design choice. CMake >=3.24/C++20 and Python >=3.11 are the minimum project constraints. Linux CPU is the initial reference profile; other platforms/devices earn support through separate recorded build/parity evidence.

The JSON parser must reject duplicate keys through parser callback/SAX validation rather than silently keeping the last value. Schema fixtures are normative but a runtime JSON-schema engine in C++ is not required: explicit validators enforce the same shape plus semantic constraints, checked against shared fixtures. Hashing uses a standard dependency rather than handwritten cryptography. YAML is configuration only; interprocess contracts use JSON/JSONL.

No Hugging Face Transformers, Python torch, pybind, database, workflow framework, MCP SDK, GPU manager, safetensors loader, Parquet or custom numerical kernel is required for v1. Parquet, portable weight export, dynamic candidate scoring, multi-task training and GPU optimization are explicit future extensions.

## Version/license evidence

Implementation must capture exact distribution/source hashes, compiler ABI/runtime, licenses and dependency inventory. A release loader accepts only tested LibTorch compatibility profiles; filename extension alone never determines compatibility. Do not infer AMD/CUDA/OpenCL support from a library's general feature list. A configured unavailable device is an error, not silent fallback.

See [sources](sources.md) for upstream capabilities checked during design. These sources do not establish that this repository currently builds or trains.
