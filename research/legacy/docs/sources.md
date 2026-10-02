# Source checks and evidence limits

Checked 2026-09-29 while preparing this design. These upstream links substantiate available component capabilities; the architecture, defaults, contracts and project gates are Arbitrium design decisions. Moving upstream documentation is not a tested dependency lock.

| Primary source | Fact used / consequence |
|---|---|
| [PyTorch C++ frontend](https://docs.pytorch.org/cppdocs/frontend) | Native C++ tensor/nn/optimizer infrastructure supports the selected language boundary. |
| [C++ serialization](https://docs.pytorch.org/cppdocs/api/serialize/save_load.html) | C++ model/optimizer serialization exists; v1 uses a documented native archive profile. |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Cross-release/platform bitwise reproducibility cannot be assumed; record environment and measure parity. |
| [SentencePiece options](https://github.com/google/sentencepiece/blob/master/doc/options.md) | Trainer supports explicit token IDs, normalization and BPE/byte-fallback controls. |
| [SentencePiece special symbols](https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md) | Control symbols can be injected separately from content encoding; verify literal-string behavior. |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | Chat completion and structured-output capabilities are exposed; exact runtime compatibility still needs a capability probe. |
| [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Temperature scaling is an established post-hoc baseline, not a guarantee for this new task or distribution. |

No trained Arbitrium accuracy, throughput, memory result or target-machine GPU compatibility was measured in the design task. No external FLAMORIS adapter implementation was audited. Repository facts were read at main `0e88cad5ab676584e0f2e827c2b553d398c535e4`; future implementations must refresh that inventory.
