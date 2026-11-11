# AGENTS.md

This repository is part of the FLAMORIS ecosystem.

AI agents and human contributors should inspect the current repository before making substantial changes. Do not assume that setup, build, deployment, service names, paths, configuration, or architecture match another FLAMORIS repository.

## Arbitrium-specific boundaries

- Arbitrium is the Decision specialization for Maidionis.
- Keep Decision-specific TaskSpecs, curricula, experiment reports, measured results, failure analysis, and specialization metadata in Arbitrium.
- Keep reusable specialization-neutral model, training, evaluation, checkpoint, artifact, and interface mechanisms in Maidionis.
- Keep Workflow execution, scheduling, shared execution state, and model execution in FLAMORIS AI Runtime.
- Preserve dataset identity, hashes, seeds, split definitions, metric meaning, and known failures when migrating research.
- Keep unsuccessful research results alongside successful diagnostics.
- Do not describe train-only memorization as held-out generalization.
- Use new public commits for migrated material rather than importing earlier repository history.
- Do not copy build/test commands from earlier research until the corresponding implementation exists and is verified here.

## Core principles

1. **Current implementation is authoritative**
   - Read the repository documentation, configuration, tests, and relevant source before changing behavior.
   - Do not invent repository-specific commands, paths, services, or configuration.

2. **Keep responsibility clear**
   - Keep this repository focused on its documented purpose.
   - Preserve application and service ownership boundaries.
   - Do not create a second source of truth for state owned elsewhere.

3. **Reuse deliberately**
   - Check existing FLAMORIS shared packages and repositories before duplicating common infrastructure.
   - Reuse code only when the dependency direction and ownership boundary remain clear.
   - Avoid speculative abstractions for requirements that do not yet exist.

4. **Security and privacy are architectural requirements**
   - Never commit or log secrets, credentials, tokens, private keys, or sensitive user data.
   - Prefer least-privilege access and bounded resource use.
   - Treat external input and remote responses as untrusted.

5. **Stable behavior over cleverness**
   - Prefer explicit, testable contracts and straightforward implementations.
   - Preserve existing public behavior unless a change intentionally modifies it.
   - Document externally visible behavior and compatibility impact.

6. **Documentation must track reality**
   - Mark the current implementation/status explicitly when a repository has both shipped behavior and future phases.
   - Do not describe implemented behavior as merely planned, and do not describe planned behavior as already shipped.
   - Keep public architecture portable. Machine names, private topology, credentials, and deployment-only paths belong in private deployment documentation rather than public repository defaults.

7. **AI-native, human-authoritative**
   - AI-assisted development is welcome.
   - Humans remain responsible for reviewing behavior, security, licensing, and compatibility.

## Before implementing a substantial change

- read this file and README.md;
- identify **what this repository is, what it owns, what it does not own, its current status, and where it fits in FLAMORIS**;
- read the [organization map](https://github.com/flamoris-jp/.github) and the relevant family map when cross-repository context matters;
- read relevant docs, Issues, and Pull Requests;
- inspect current implementation and tests;
- identify the source of truth and dependency direction;
- check whether reusable FLAMORIS infrastructure already exists;
- verify repository-specific setup and deployment details instead of guessing.

## Testing

Add or update tests where practical.

Prefer deterministic tests and explicit contracts. When behavior differs by platform, runtime, provider, or environment, document the supported boundary and test the relevant cases.

## Licensing

Unless stated otherwise, code in this repository is licensed under Apache License 2.0.

Do not add third-party code, models, model weights, datasets, fonts, media, or generated assets unless their licenses are compatible and clearly documented.

## Support

FLAMORIS does not provide guaranteed individual support.

Use the repository documentation, Issues, tests, logs, and source code as primary references when diagnosing problems.

## Audio/video storage policy

Do not commit audio or video files (including MP3, MP4, WAV, FLAC, M4A, OGG, MOV, WebM and equivalents), regardless of size, ownership, license, or whether they are generated demos or test samples. This also prohibits Git LFS pointers, renamed/encoded media and media hidden inside archives. Do not bypass the rule with `git add -f` or ignore exceptions.

Store media outside Git in approved artifact/media storage. Commit references, SHA-256 digests, provenance, analysis results, scripts and small text/symbolic fixtures instead. Generate temporary synthetic media during tests when needed. Licensing review does not waive this storage policy. MIDI/project data and images retain their existing repository-specific rules.

The `.gitignore` covers common audio/video extensions case-insensitively. Where installed, the `Media policy / No tracked audio or video` PR check also detects force-added files and LFS pointers by filename; it is not a content/archives scanner or a server-side push restriction. Do not rewrite history or remove external assets as part of this rule. See the [shared policy](https://github.com/flamoris-jp/flamoris-commons/blob/main/docs/repository-policy.md#121-audio-and-video-storage).
