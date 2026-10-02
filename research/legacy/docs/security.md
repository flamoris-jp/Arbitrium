# Security, provenance and resource boundary

## Threats relevant to this project

| Surface | Failure | Required control |
|---|---|---|
| Teacher prose | prompt injection, fabricated fields/labels | strict schema, blind review, source-fact verification and batch audit |
| Training/evaluation | label leakage or near-duplicate split leakage | input allowlist, family union, immutable split ledger |
| Native artifacts | malicious/corrupt archives and oversized tensors | trusted provenance plus manifest/hash/resource/shape checks before and after load |
| HTTP teacher | credential disclosure, endpoint substitution | configured origin, no redirects, redacted credentials, recorded model identity |
| CLI boundary | shell execution/path injection | argv subprocess, allowlisted task and local files, no shell interpolation |
| Model output | overconfidence or stale evidence | calibration/abstention plus independent caller policy/freshness checks |
| Experiment logs | private state leak | approved synthetic data by default, redaction before capture, no default raw production logging |

All text from teacher, reviewer, input evidence or external files is data, never executable instruction for the harness. Generated statements cannot change a target directory, add network endpoints, execute a command, choose a dependency or waive verification. A JSON schema guarantees shape only, not truth.

Core model/training libraries have no network dependency. Native code may read explicitly supplied local datasets and artifacts; that is not an authority to mutate product/OS state. The teacher server is operator-managed. This task does not install/start services, download GPT-OSS weights or change firewall/GPU settings.

## Data handling

Use controlled synthetic cases unless a separately documented source is approved. No private prompts, credentials, personal names or machine identifiers enter Git, fixture data, teacher requests or model artifacts by default. Normalize approved runtime-derived events before capture. Record permission/license basis for every source and for any teacher-generated-data usage restrictions; do not assume that a model's label settles licensing. This design does not add a repository LICENSE without the owner's choice.

Store secrets in environment/secret store, not resolved config/report. Config snapshots retain the variable name and redacted marker. Teacher URL in shared metadata contains origin class/provider, not secret host/query/user information; keep deployment-specific endpoint in local protected logs if necessary. Hashes of private text are not an anonymization guarantee.

Retain raw responses, rejected records, audit decisions and lineage for reproducibility in access-controlled experiment storage. There is no automatic retention purge in v1; export/retention is an operator decision. Frozen data cannot be silently edited for privacy removal: withdraw the version and publish a sanitized successor with a withdrawal record. Never commit generated datasets/checkpoints/model weights to this repo.

## Fail closed where evidence is missing

Missing calibration, unsupported schema, unknown model/task hash, nonfinite tensors and insufficient audit evidence block release. Do not convert these failures to confidence zero or model stop. An integrity hash alone cannot make an untrusted archive safe. English-only scope is declared, not a perfect automatic language detector; caller responsibility and measured out-of-domain limitations remain explicit.
