# Model design

Normative v1 design; see [requirements](requirements.md). All weights start random. One task per artifact; no Python mirror model.

## Tensor contract

`forward(token_ids, attention_mask, segment_ids)` receives int64 IDs `[B,L]`, boolean mask `[B,L]` (`true = real token`), and int64 segments `[B,L]`. `1 <= B <= 64`, `1 <= L <= 256`. IDs are within artifact vocabulary; masks must be contiguous real tokens then right padding. At least the required control tokens and nonempty state are present. Forward returns FP32 `decision_logits[B,K]` and `answerability_logit[B]`. It returns logits, never calibrated probabilities. Public service calls use batch size 1; evaluation may batch.

Input assembly is exclusively defined by [tokenizer](tokenizer.md). Target, rationale, teacher confidence, source and split metadata never enter tensors. Answerability target is separate from the K decision classes; `stop` is an actual decision, not abstention.

## Baselines

1. **Majority / lexical rules**: report train-majority and fixed keyword heuristics using only the same visible input. Freeze their rules before dev scoring. The structured scenario oracle labels examples but is not a fair deployable comparator unless structured facts are actually available to the caller.
2. **Pooled embedding**: token embedding `[V,256]`; masked mean over content tokens (exclude controls and padding); Linear(256,256), GELU, dropout(0.1); independent Linear(256,K) and Linear(256,1) heads. No positional or segment embeddings. This intentionally order-insensitive baseline exposes whether word order matters. It replaces the current mean+linear smoke only during implementation.
3. **tiny_encoder_v1**: the following explicit encoder, introduced only after the pooled baseline is measurable.

| Parameter | Fixed tiny default |
|---|---|
| Vocabulary | actual trained size, at most 8192 |
| Hidden width / layers | 256 / 4 |
| Attention heads / head dimension | 4 / 64 |
| FFN width / activation | 1024 / GELU, exact erf form |
| Maximum sequence | 256, including controls |
| Positions / segments | learned absolute `[256,256]` / learned `[4,256]` |
| Dropout | 0.1 on embeddings, attention probabilities, attention output and FFN output |
| LayerNorm | pre-norm, epsilon `1e-5`, final norm after last block |
| Pooling | final hidden vector at CLS, position 0 |
| Decision / answerability heads | independent linear projections from pooled vector |
| Precision | FP32 initially, no automatic mixed precision |

Initialize embeddings and all Linear weights from Normal(0,0.02); biases zero; LayerNorm weights one/bias zero. Padding embedding row zero and excluded from updates. Position/segment embeddings use the same normal initialization. Every checkpoint records architecture ID and actual parameter count. At V=8192 and K=3 the encoder is approximately 5.32M parameters; the pooled baseline is approximately 2.16M. Targets are not performance measurements. A 10–30M variant is deliberately not another default: expansion requires a documented failed-capacity experiment.

## Explicit attention and masking

At each block let `U = LayerNorm(X)`. Separate learned affine projections produce Q,K,V, reshaped `[B,H,L,D]`. Compute `S = Q K^T / sqrt(D)`. Mask padded **key** columns to negative infinity before softmax; apply probability dropout; multiply by V, concatenate heads and apply output projection/dropout. Add residual to X. Then pre-normalize, apply Linear→GELU→Linear→dropout FFN and add the second residual. Zero padded query rows after each residual. CLS ensures no valid query has an all-masked key set. No causal mask is used: the encoder sees the complete bounded evidence.

Implement composition in repository C++ using LibTorch tensor/nn primitives; do not use a pretrained encoder or hide model semantics in a Python exporter. An optimized attention kernel is a later substitution requiring output/gradient parity with this reference.

## Task heads

Choice uses K=2..16 ordered task labels; initial K=3. Binary uses two logits, canonical labels `[false,true]`, and the same categorical cross-entropy/calibration path. Ordinal score uses K=2..16 strictly ascending finite anchors; the initial rubric has anchors `[0,0.25,0.5,0.75,1]`. Use categorical CE over the adjudicated anchor; ordering is evaluated with MAE and ranked probability score. Output score is the probability-weighted mean of anchors, not a learned free-text score. No arbitrary caller-supplied range or bin count is accepted.

Answerability uses one sigmoid logit trained on reviewed `answerable` labels. It is an empirical predictor for supported input, not a proof of completeness or an OOD detector. Calibration and routing gates are specified in [calibration](calibration.md).

## Implementation acceptance

Padding invariance; evaluation-mode repeated inference; batch-vs-single agreement within FP32 tolerance; finite gradients; nonzero gradients for Q/K/V and both heads on appropriate fixtures; tiny-set overfit; label-order mapping; no target/provenance serialization; save/load parity. Test negation/temporal minimal pairs on held-out families to establish the encoder's usefulness. Shuffling tokens is a diagnostic, not a benchmark success criterion.
