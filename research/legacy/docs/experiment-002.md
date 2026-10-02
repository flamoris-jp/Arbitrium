# Experiment 002 — 簡単な教材から始める教育診断

2026-09-30。**24問の基本教材は3 seedで全問正解になった。ただし、未知問題への対応と判断不能の識別は改善を実証できていない。** 教材を増やす前に、学習データで覚えられる範囲と、別の問題に対応できる範囲を分けて測定した。

## 元の教材を残す

[Experiment 001](experiment-001.md) の `examples/education/controlled-recovery-1000-v1/` は、データ・provenance・manifestを含む9ファイルのSHA-256がすべて変更前と一致する。1,000件の生成コード `experiment.py` と [実測レポート](experiments/experiment-001-results.json) もそのまま保存した。過去情報の上書きや矛盾を扱う、後の段階の教材として使える。

旧教材も新教材も、決定的な有限状態oracleで正解を付けた合成の英語教材である。実際に学習へ投入したJSONLを保存しているが、実運用から採取したデータ、LLMの独立レビュー、人による意味監査ではない。

## 見つかった問題と行ったこと

| 問題・観察 | 今回行ったこと | 分かったこと・残る限界 |
|---|---|---|
| 旧1,000件は最初から earlier/current の上書き情報を含む | 現在状態だけの短い文を別バージョンで追加 | 時系列の難しさを外して基本学習を診断できる |
| 旧実験の損失減少は、三択の改善を意味しなかった。選択epochのdecision NLLは初期値より悪化 | 学習用・devの三択とanswerabilityを別々に記録。ラベル別support/recallと混同行列を追加 | 判断不能を全部「判断できる」とする失敗も見える |
| 旧trainのretryは194行だが現在状態の組合せは14通り、devのretryは14行だが1通り | 新しい96問・72問は、各教材内で同じ現在状態を繰り返さない | 行数を独立した学習例の数として扱わない |
| pooledモデルは語順・segmentの所属を利用せず、内容tokenを平均する | モデル構造を据え置いて簡単なデータから診断 | 時間・否定の係り先が難しいという仮説は残る。今回だけで原因と断定できない |
| 完全な96状態でもstopが78問あり、全体accuracyが高く見える | 全体accuracyに加えて各ラベルrecallを確認し、8問ずつの24問を追加 | 96問ではretry recallが25%。24問の暗記は全ラベル100% |
| 暗記の成功と未知問題への対応が混同されやすい | `train_diagnostic` と `dev` の選択を明示。毎回新規モデルから開始 | 24問の100%を未知問題の成績として報告しない |

C++がモデル構築・学習・計測を担当し、Pythonは教材作成・SentencePiece・実験の起動を担当する。外部teacherは呼び出していない。既存のsplit hash、重複検出、oracle、学習モデルの構造は変更していない。

## 始める順序と教材

| 段階 | データセット | 内容・件数 | 今回の使い方 |
|---|---|---|---|
| 最初 | `recovery-current-tiny-v1` | 現在の確定事実のみ24問。retry/fallback/stop 各8問 | trainのみの暗記診断 |
| 次 | `recovery-current-diagnostic-v1` | 現在の確定事実の全96状態。retry 8 / fallback 10 / stop 78 | trainのみの全状態診断 |
| 次 | `recovery-current-grouped-v1` | 過去情報なし。不明な事実も含む72状態。三択と判断不能 各18問 | familyを分離したtrain/dev診断 |
| 後 | `controlled-recovery-1000-v1` | 旧1,000件。過去情報の上書き・明示的矛盾を含む | 保存。今回再学習しない |

24問は96問の部分集合で、retryの8問に対して `fatal_violation` だけを反転したstopの8問を対にした。fallbackの8問も含む。stopの他の理由、不明な事実、語順の一般化はこの24問では学べない。24問と96問は同じ状態を含むので、独立した評価集合として扱わない。

72問は24個の表現テンプレートから作り、同じ表現familyを既存のhashで同じsplitへ配置する。シナリオを性能に合わせてsplit間で移動しない。各状態の英語表現は一つだけで、同じ教材内に水増し用の言い換え行はない。

| 72問のsplit | retry | fallback | stop | 判断不能 | 合計 |
|---|---:|---:|---:|---:|---:|
| train | 7 | 9 | 10 | 8 | 34 |
| dev | 3 | 3 | 4 | 4 | 14 |
| calibration_fit | 1 | 2 | 0 | 1 | 4 |
| calibration_select | 1 | 1 | 1 | 2 | 5 |
| test | 6 | 3 | 3 | 3 | 15 |

trainは12 family、devは5 family。devの各クラスは3〜4問しかなく、結果は不安定である。公開の研究fixtureであり、sealed testや本番のsupport-count要件を満たさない。診断用24問・96問は登録した一つのtrain familyにまとまり、devは空。空のdevを評価したことにはしない。

これらは教材の難しさを順に調べる独立した実験であり、前段階の重みを次に引き継ぐ連続学習は実装していない。診断モデルの重みをheld-out評価に流用しない。

## 実行条件と記録

| 条件 | 24問 | 96問 | 72問 |
|---|---:|---:|---:|
| config | `current-tiny.json` | `current-diagnostic.json` | `current-generalization.json` |
| 最大epoch | 200 | 120 | 120 |
| learning rate | 0.01 | 0.003 | 0.003 |
| batch size | 24 | 16 | 16 |
| patience | 200 | 120 | 20 |
| weight decay | 0 | 0.01 | 0.01 |
| epoch選択 | train objective | train objective | dev objective |

configは `configs/experiments/` に保存した。共通条件は `pooled_embedding_v1`、hidden size 256、dropout 0.1、seed 42/43/44、CPU FP32、一つのthread。tokenizerは各データセットのtrainだけで新規作成する。旧実験と教材・学習率・epoch・batchが同時に変わるため、「教材だけが改善の原因」とは結論しない。

96問と72問の条件は実行前に固定し、3 seedを実行した。96問で少数ラベルの学習不足が見えた後に、追加の24問と専用configを固定して3 seedを実行した。先の失敗した実験も、すべて下記レポートへ保存した。無期限の再試行やdevでの自動教材生成はしていない。

| データセット | manifest SHA-256 |
|---|---|
| 24問 | `1ba240825e9ec17feb880d0696757d47c5ca117f4d7d740c301d0ffff02b1619` |
| 96問 | `417c2a787f6fd08e5eee3d3cc5c56da5cedc7b713fdfeced7e17af381b95275c` |
| 72問 | `66e53f042b2e5918c4d96dd5cf5e83f92e3d67e4aec8dfea1dd2e30bbda56c85` |

各reportにはconfig、tokenizerとC++ executableのSHA-256、toolchain、全epochの損失、初期値と選択epochのtrain/dev計測を残した。tokenizer語彙が異なるためparameter countも異なる（24問・96問: 254,468、72問: 318,980）。実測環境はGCC 13.3.0 / LibTorch 2.14.0+cpu / SentencePiece 0.2.2 / Python 3.12.14 / CMake 4.4.3。

## 実測結果

以下のepochは0始まり。三択accuracy/F1は判断可能な行だけ、answerabilityの計測は全行を使う。最終epochではなく、指定したobjectiveで選ばれたepochを再ロードして測定した。

[24問の全レポート](experiments/experiment-002-tiny-results.json):

| Seed | 選択epoch | train三択accuracy | 各ラベルrecall | Brier | ECE |
|---|---:|---:|---|---:|---:|
| 42 | 192 | 100% | 100% / 100% / 100% | 0.000051 | 0.003515 |
| 43 | 196 | 100% | 100% / 100% / 100% | 0.000036 | 0.002947 |
| 44 | 199 | 100% | 100% / 100% / 100% | 0.000068 | 0.003803 |

ラベル順はretry/fallback/stop。多数派基準は33.33%。基本的な英語入力の学習経路が働き、限定した24状態を覚えられることは確認できた。全行が判断可能なので、answerability accuracy 100%は判断不能を学べた証拠にならない。

[96問の全レポート](experiments/experiment-002-diagnostic-results.json):

| Seed | 選択epoch | train三択accuracy | retry recall | fallback recall | stop recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 115 | 90.625% | 25% | 70% | 100% | 0.6785 |
| 43 | 116 | 90.625% | 25% | 70% | 100% | 0.6785 |
| 44 | 111 | 89.583% | 25% | 60% | 100% | 0.6442 |

全問stopでも81.25%になる。全体accuracyはこの基準を上回るが、全状態を覚えられたとは言えない。

[72問の全レポート](experiments/experiment-002-grouped-results.json):

| Seed | 選択epoch | train三択accuracy | dev三択accuracy | Dev Macro-F1 | Dev Brier | Dev ECE |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 20 | 38.462% | 40% | 0.3238 | 0.661564 | 0.038247 |
| 43 | 25 | 46.154% | 30% | 0.1538 | 0.641840 | 0.296827 |
| 44 | 30 | 53.846% | 20% | 0.1212 | 0.619671 | 0.393832 |

dev三択は10問が分母。train多数派をそのまま答える基準は40%、均等ランダムの期待値は33.33%。すべてのseedでdevのretry recallは0%。answerabilityは14問中10問正解（71.429%）だが、**判断不能の4問をすべて「判断できる」と予測している**。小さいECEだけで信頼できるモデルと判定しない。

温度校正はfitしていない。Brier/ECEは生の確率の診断値である。calibration_fit/calibration_select/testのpayloadは学習runnerで読まない。新しい教材を凍結・検証する処理と、学習時のpayload利用は区別する。

## 再実行

依存関係・ビルドは [build.md](build.md) と [Experiment 001の手順](experiment-001.md#build-and-run) を使う。通常インストールしたPython packageとnative executableを用意してから、**まず24問**を実行する。

```sh
.venv/bin/python -m arbitrium_education.current_curriculum run \
  --dataset examples/education/recovery-current-tiny-v1 \
  --output runs/experiment-002-tiny \
  --trainer build/arbitrium_experiment \
  --config configs/experiments/current-tiny.json
```

次の二つも独立した新規モデルで実行する。

```sh
.venv/bin/python -m arbitrium_education.current_curriculum run \
  --dataset examples/education/recovery-current-diagnostic-v1 \
  --output runs/experiment-002-diagnostic \
  --trainer build/arbitrium_experiment \
  --config configs/experiments/current-diagnostic.json
.venv/bin/python -m arbitrium_education.current_curriculum run \
  --dataset examples/education/recovery-current-grouped-v1 \
  --output runs/experiment-002-grouped \
  --trainer build/arbitrium_experiment \
  --config configs/experiments/current-generalization.json
```

Pythonはtrain-only SentencePieceを作り、各seedについて次の形でC++を起動する。

```sh
build/arbitrium_experiment DATASET TOKENIZER OUTPUT SEED RESEARCH_CONFIG
```

outputは上書きしない。再試行は新しいoutput名を使う。研究configは厳密なキーと型、最大200 epoch、LR 0.02以下、batch 64以下で制限する。configを渡さない旧呼び出しは旧default条件で動く。研究用reportであり、release artifact schemaの変更ではない。

再生成は別の保存先で行う。既存データセットを置き換えない。生成コードのbytesが同じなら教材とprovenanceが一致する。manifestの生成時刻は新しくなるため、実験にはコミット済みmanifestを使う。

```sh
.venv/bin/python -m arbitrium_education.current_curriculum generate \
  --task docs/examples/recovery-task.json --output datasets/frozen
.venv/bin/python -m arbitrium_education.tiny_curriculum \
  --task docs/examples/recovery-task.json --output datasets/frozen
```

## 検証と次の課題

native build、CTest 6/6、通常インストールしたPython packageで48/48を確認した。旧1,000件と新3教材の再生成一致、ファイルhash、oracle正解、family分離、24問のfatal最小対、改変trainの拒否、診断時に予約splitを読まない境界を検証する。nativeでも、予約payloadを削除した診断と不正configの拒否を確認する。公開前に完全なGit treeを比較し、既存ファイルの削除を防止する。

次は24問の暗記成功を足場に、少数ラベルが埋もれない96状態の練習と、不明な情報が決定に影響する対・影響しない対を分けた教育を行う。既知問題のラベル別成績を安定させてから、十分な独立familyを持つ未知問題の評価を増やす。その後に否定の係り先、時間順、矛盾を追加し、保存した旧1,000件を使う。現在情報の段階で残る誤りを確認したうえでpooled/encoderを同条件で比較する。今回の結果からreleaseや本番への昇格は行わない。
