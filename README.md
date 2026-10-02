# Arbitrium

**A Maidionis Decision specialization for bounded advisory judgments over supplied evidence.**

Arbitrium is the first experimental Decision specialization built on Maidionis.

It owns Decision-specific TaskSpecs, curricula, experiment reports, measured results, failure analysis, and specialization metadata. Specialization-neutral model/training/evaluation infrastructure belongs in Maidionis. Workflow execution and runtime orchestration belong in FLAMORIS AI Runtime.

## Current status

**Initial A1–A3 offline CPU implementation and controlled research comparison.**

The repository preserves the historical research archive and implements an explicit Decision composition on pinned Maidionis Core. See [implementation status](docs/implementation-status.md), [architecture](docs/architecture.md) and [comparison results](research/public-architecture/comparison.md). Runtime integration and production model qualification remain separate.

Existing controlled research shows that a tightly bounded train-only curriculum can be fitted. It does **not** establish reliable held-out generalization or production quality.

## Research home

Arbitrium is the public home for Decision-specific research evidence, including:

- Experiment 001 and Experiment 002
- 24 / 96 / 72 lesson curricula
- the original 1,000-record curriculum
- dataset and manifest hashes
- seed 42 / 43 / 44 results
- label support / recall evidence
- held-out failures
- answerability failures
- reproducibility information

Successful training diagnostics must not be presented as stronger evidence than they are.

## Relationship

```text
Maidionis
  ↓ specialization foundation
Arbitrium
  ↓ bounded Decision inference
FLAMORIS AI Runtime
  ↓ workflow / execution
Applications
```

## Migration policy

Do not import the previous research Git history wholesale.

Public migration uses new commits while preserving research meaning, dataset identity, hashes, seeds, metrics, and known failures.

The live implementation links specialization-neutral Maidionis Core. Historical source under `research/legacy` remains an integrity-checked archive, with separate legacy readers and new-format derived dataset identities.

Repository setup and migration are tracked in [Issue #1](https://github.com/flamoris-jp/Arbitrium/issues/1) and [Research migration](docs/research-migration.md).

## Getting started

Use [the supported Linux CPU build and verification instructions](docs/build.md). Offline verification exercises real training/checkpoints, admitted archive loading, complete evaluation and research finalization. The initial codec and artifact format are experimental; no calibrated serving or Runtime R1 is claimed.

## FLAMORIS

FLAMORIS is open-source software for creative work and AI-native production.

Use it however you like.

Commercial use is welcome and does not require permission. If you'd like, we'd be happy to hear what you used FLAMORIS for. This is completely optional.

FLAMORIS software is provided as-is. We do not provide individual support or guaranteed assistance.

If you run into trouble, let your AI assistant read the repository, documentation, Issues, tests, logs, and source code and help you solve it.

If FLAMORIS helps you or you find it interesting, your support helps fund development and keeps the project growing. 🌱

<sub>Mostly GPU bills.</sub>

---

## FLAMORISについて

FLAMORISは、クリエイティブ制作とAIネイティブな制作環境のためのオープンソースソフトウェアです。

勝手に使ってください。改造しても、組み込んでも、面白いものや変なものを作ってもOKです。

商用作品や製品で使う場合も許可は不要です。もしよければ「こんなのに使ったよ」と教えてもらえるとうれしいです。もちろん強制ではありません。

FLAMORISのソフトウェアは現状のまま提供されます。個別サポートや動作保証はありません。

困ったときは、README、ドキュメント、Issue、テスト、ログ、ソースコードをあなたのAIに読ませて、自己サポートしてもらってください。

もしお役に立てたり、面白いと思っていただけたなら、開発費用をご支援いただけるとうれしいです。FLAMORISは元気になって育ちます。🌱

<sub>主にGPU代とか。</sub>

## License

Code in this repository is licensed under the [Apache License 2.0](LICENSE), unless otherwise noted.

AI models, model weights, datasets, media, and other non-code assets may use separate licenses. State their applicable licenses alongside those assets.
