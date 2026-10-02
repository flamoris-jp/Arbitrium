# Artifact and checkpoint format

## Selected v1 format

Use explicit versioned bundles with **LibTorch native C++ archives** for v1 weights/checkpoints and SentencePiece `.model` for vocabulary. This intentionally resolves the bootstrap's tentative safetensors/tokenizer.json language. No Python pickle loader, Python converter or TorchScript program is an inference dependency. Native archive compatibility is restricted to the recorded/tested LibTorch build profile; portable cross-framework weights are not promised. Safetensors is a future versioned exporter project, not an ambiguous alternate v1 loader.

`bundle/` contains exactly these required files:

| File | Contents |
|---|---|
| manifest.json | format/identity/hash/compatibility inventory |
| model-config.json | architecture ID and fully resolved tensor dimensions/options |
| task.json | immutable TaskSpec |
| tokenizer.model | native SentencePiece vocabulary and normalization |
| tokenizer-config.json | trainer options, control ID map, actual V, assembly version and corpus digest |
| weights.pt | C++ OutputArchive module parameters/buffers; no optimizer or executable plugin |
| calibration.json | temperatures, gates and exact bound component hashes |
| training-metadata.json | dataset/run/toolchain/config identities and reproducibility information |
| model-card.md | intended use, metrics, limitations, license/provenance and release status |
| evaluation-summary.json | registered report digests, gate outcomes and support counts |

Manifest schema `arbitrium.artifact.v1` fields: format_version=1, artifact_id, created_at UTC, task_id, architecture_id, dtype=`float32`, status=`research_only|promotable`, files mapping (path→sha256/bytes) containing every required bundle file except `manifest.json` itself and no extras, compatibility (exact LibTorch version/build digest, compiler ABI family, OS/arch, SentencePiece version, assembly and contract versions), parameter_count, tensor_inventory (name/dtype/shape), max_total_bytes, expected_model_bytes, training_run_id. Do not include the manifest's own hash inside itself. Bundle digest is SHA-256 of final manifest bytes; expected digest is supplied through the caller's trusted registry/CLI. Artifact ID is an immutable logical label; digest is the integrity identity.

Model config must enumerate architecture, V, hidden size, layers, heads, FFN size, max length, dropout, normalization epsilon, initialization, pooling, K and answerability head. Pooled-only fields use explicit null where not applicable. A loaded artifact determines supported dimensions; request data cannot allocate a new architecture.

Calibration schema `arbitrium.calibration.v1`: calibration_id, weights_sha256, task_sha256, tokenizer_sha256, model_config_sha256, assembly_version, calibration_dataset_digest, fit/select split hashes, T_dec, T_ans, tau_p, tau_q, accept_none, fit_support, selection_support, optimizer settings/convergence/boundary flags, raw/fitted NLL, gate_policy_version, gate_metrics. Temperatures finite in [exp(-4),exp(4)]; thresholds in the registered grid, or null if accept_none=true. Unknown/mismatched binding fails load. A `promotable` bundle cannot accept_none.

## Loader order and limits

1. Validate directory, expected manifest digest, schema and allowlisted compatibility profile.
2. Reject absolute/parent-traversal paths, symlinks, duplicated file entries, unexpected files and missing required files. Only regular files beneath bundle root are accepted.
3. Before parsing archives, cap bundle at 512 MiB, weights at 256 MiB, JSON files at 4 MiB each, tokenizer at 32 MiB, parameters at 30M and tensor dimension/count products at architecture expectations.
4. Hash every listed file and verify byte size. Require task/config/tokenizer/calibration cross-bindings.
5. Instantiate an allowlisted compiled architecture, load native archive from a **trusted provenance** bundle onto CPU, inspect parameter keys/shapes/dtypes and finite values. Reject extras/missing tensors.
6. Check actual vocabulary/control IDs, set eval/no-grad, run a finite-output self-check, then optionally move to the explicitly selected validated device.

Checksums establish integrity, not authenticity or native-deserializer safety. Do not load arbitrary downloaded `.pt` files. Untrusted format conversion needs a separately isolated importer in a later project. A malicious actor able to replace both bundle and expected digest is outside checksum protection.

## Training checkpoint

Checkpoint directory holds `manifest.json`, `model.pt`, `optimizer.pt`, `rng.pt`, `state.json`, resolved config/task/tokenizer/data references and metrics history digest. State schema `arbitrium.checkpoint.v1` records completed_epoch, next_epoch, global_step, scheduler phase/step/base LR, best_dev_objective, best_epoch, patience counter, RNG inventories and exact code/environment hashes. Serialize optimizer parameter ordering and all RNG streams; data resume is at epoch boundary. Checkpoint is not a serving artifact and cannot bypass calibration.

Write to a fresh temporary directory on the same filesystem, flush/fsync files and directory, verify complete manifest, then rename atomically. A crash leaves a removable incomplete directory; it never changes the last good checkpoint. Resume validates all hashes/config identities and refuses to guess missing optimizer/RNG state. Export weights as CPU FP32 in a new bundle; verify logits/decisions against the checkpoint before calibration/evaluation.

## Publication and rollback

Training and calibration create candidate bundles; evaluation writes separate immutable reports. After report completion, finalize bundle summary/card/manifest and revalidate; weights/task/tokenizer/calibration must remain byte-identical to those evaluated. Report binds their hashes and candidate digest so the final manifest does not create a circular self-hash. Finalization does not rerun model selection.

Promotion is an explicit operator action updating a trusted registry's artifact_id→digest/path entry after gates pass. Never overwrite a bundle in place. Caller loads and validates the new engine before atomically swapping its reference; in-flight requests finish on their original artifact. Retain previous bundle for rollback. No production registry, signing infrastructure or remote model downloader is implemented inside Arbitrium v1.
