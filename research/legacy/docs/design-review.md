# Arbitrium 設計レビュー — v1

設計基準: `main@0e88cad5ab676584e0f2e827c2b553d398c535e4`。2026-09-29。

**設計一式を作成しました。実装・学習・実機投入はまだです。** この文書から判断事項を一括レビューできます。設計は合理的な既定値を選んで完結させ、性能・環境・学習成果に関する未検証事項はリスクとして区別しています。

## 今回の設計の中心

**C++が学習して判断する。PythonとGPT-OSSが外から教育する。実行を決めるのは呼び出し側。** Attentionには教材生成・外部通信・運用権限を入れません。

| 判断事項 | 採用した設計 | 理由 / 代償 |
|---|---|---|
| タスクと選択肢 | 1 artifact = 1 task。学習済みの固定ラベル。並べ替えだけ可 | 固定分類ヘッドで未知の候補まで判断できるように見せない。汎用ランキングは別設計 |
| 最初の課題 | retry / fallback / stop。TaskSpec固定のcanonical policyと英文の根拠を読む | v1ではpolicy自体の自由な言い換え・変更を学習対象にせず、数値だけで決まるケースは通常コードに任せる |
| 判断保留 | 行動3択とは別のanswerability head + 校正確率の閾値 | 情報不足をstopに混ぜない。未知入力を完全検知できるとは主張しない |
| モデル | masked pooled baseline → 4層・256幅・4headの自前Encoder | Attentionの効果を比較できる。8k語彙時Encoder約5.32M parameter |
| Tokenizer | SentencePiece BPE、自前学習、最大8192語彙、256 token | 既成モデルの語彙を引き継がない。語彙学習はtrainのみ |
| 教材の正解 | 有限状態の構造化シナリオ + oracle + 英文対応確認 + 監査 | GPT-OSS自身の賛成を正解にしない。GPT-OSSはpolicyを書き換えずstate表現を多様化し、自由英文には人間の意味監査が残る |
| 教師とレビュー | blind review、原案と修正を別記録、同じモデルなら相関を明示 | プロンプトを変えるだけで独立教師とは呼ばない |
| データ分割 | train / dev / calibration_fit / calibration_select / test | 重み選択・校正・閾値選択・最終試験を分離 |
| 弱点教育 | devの集計だけを教材生成へ返す。最大3 cycle | 試験問題を覚える循環と無限実行を避ける |
| 校正 | decision / answerability別のtemperature scaling | 単純でC++実装可能。分布変化に対する保証ではない |
| 保存形式 | v1は明示したLibTorch native bundle + SentencePiece model | C++だけで完結。safetensors/他frameworkへの可搬性は後続版 |
| 実行環境 | Linux x86-64 CPU FP32を基準 | GPU対応を推測しない。正確な依存版は実装時の成功ビルドで固定 |
| binary / score | binaryは2クラス、scoreは固定ordinal bins | 同一基盤で実装可能。任意の質問や任意の尺度は受け付けない |
| 連携 | transport-neutral C++ API + JSONL CLI | MCPやRuntimeの責任を持ち込まない。外部repoの実装は別PR |

## レビューしてほしい判断

1. **汎用的に何でも選ぶモデルではなく、タスク単位で育てる**方針です。新タスクにはデータ・重み・校正が必要です。
2. **教育の自動化と、正解の信用を分けます。** GPT-OSSが自由に書いた英文の意味が元の状況と一致するかは、モデル同士の合意だけでは保証できません。最初は制御した英文で仕組みを証明し、自由英文には抽出チェックとサンプル監査を加えます。
3. **最終試験を教育ループへ戻しません。** 詳細を開いて改善に使った場合は、その試験を独立評価として扱うのをやめ、新しい試験を用意します。
4. **実装完了と研究成功を分けます。** 目標を満たせなければ「研究用」として結果を残します。失敗をごまかすための閾値変更・自動採用はありません。

## 既存コードとの不整合をどう扱ったか

既存C++はmaskなしのmean pooling + linear、ランダム初期化のsmokeです。Tokenizer、学習、校正、モデル保存、英文判断の実装はありません。Pythonは接続骨格ですが、レビューに原案の答えが見え、修正ラベルを上書きし、文字列のbool変換も厳密ではありません。パッケージ設定の相対パスも修正対象です。

今回は実装を変更せず、[実装計画](implementation-plan.md)に移行手順と検証条件を具体化しました。READMEと既存設計文書は新仕様へ整理し、実装済みと設計上の予定を区別しています。

## 残るリスクと検証先

| リスク | いま分かること | 実装後に確かめること |
|---|---|---|
| 小型モデルが意味を学べるか | モデル構造と教育方法は設計済み | 未知の表現・否定・時間順序でbaselineを上回るか |
| 教師の癖を覚えるだけにならないか | family分割・独立作問・監査を規定 | 独立作問subset、表現family別の精度 |
| 保留と校正が機能するか | fit/threshold/testを分離 | coverageと誤答率の両立、情報不足の誤受理 |
| データ数・計算時間 | 初回1万件以上の目標、CPU基準 | 実測学習時間・必要件数。成功を保証する件数ではない |
| GPUと依存環境 | CPUで開始可能な依存境界 | 対象GPUのbuild、演算、数値一致。互換性は未確認 |
| GPT-OSS接続差異 | endpoint契約とprobeを定義 | 実際のserver版、model hash、chat template、JSON出力 |
| native archive | v1形式を固定、信頼したbundleのみ | 同一buildのsave/load/resume。汎用可搬性は約束しない |
| 実運用との隔たり | advisory APIとcaller責任を定義 | 外部Intelligence/Runtime repoを読んだ後のadapter実証 |

実装のために未選択のモデル方式・ファイル形式・分割方式・主要既定値はありません。依存版、GPU対応、教師の実接続、学習精度は実測で埋める項目です。人間の教材監査・将来の実運用採用は、その具体的な成果物ができた時点で扱います。今回の設計作業をそこで止める理由にはしていません。

## 文書案内

| 読みたい内容 | 文書 |
|---|---|
| 目的・範囲・完了条件 | [requirements](requirements.md) |
| 責任分界・コンポーネント | [architecture](architecture.md) |
| Encoder・Attention・head | [model-design](model-design.md) |
| 語彙・入力組み立て | [tokenizer](tokenizer.md) |
| API・エラー・CLI | [interfaces](interfaces.md) |
| 教師・レビュー・教材 | [education-system](education-system.md), [teacher-loop](teacher-loop.md) |
| データと分割・履歴 | [dataset-design](dataset-design.md) |
| 学習・再現・再開 | [training](training.md) |
| 校正・保留条件 | [calibration](calibration.md) |
| 評価指標・合否 | [evaluation](evaluation.md) |
| モデル保存・更新・rollback | [artifact-format](artifact-format.md) |
| 外部連携 | [integration](integration.md) |
| 安全性・運用・依存 | [security](security.md), [operations](operations.md), [dependencies](dependencies.md) |
| 実装の順番とテスト | [implementation-plan](implementation-plan.md), [roadmap](roadmap.md) |
| 根拠となる公式資料 | [sources](sources.md) |
| 機械可読の契約と例 | [request schema](contracts/request.schema.json), [example guide](examples/README.md) |

## 設計検証の記録

実施済み: ローカルリンク93件、JSON 10ファイル、JSON Schema 4種の仕様適合、保存した例6件、binary/ordinal/保留などの分岐5件、誤ったshapeを拒否する23件を検証しました。例のtask・質問・label順序・確率和・target参照も確認しています。`git diff --check` は正常で、C++/Python実装、CMake、実行configは未変更です。

横断レビューでは、最終試験の結果でEncoderからpooledへ選び直せてしまう曖昧さを修正し、方式選択はdevで完了するよう統一しました。校正fit自体の失敗と、fit成功後に保留閾値が決まらない場合も分け、後者だけをaccept-none研究用bundleにします。

これは設計・文書・例の検証です。C++/Pythonの将来のvalidator実装、ビルド、学習、性能、教師実接続は未検証です。
