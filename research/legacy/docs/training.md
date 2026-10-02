# C++ training specification

All model learning and calibration fitting is C++/LibTorch. Python prepares data and starts commands. The model never calls a teacher.

## Experiment registration

Before training, freeze: task/dataset IDs and hashes, tokenizer/assembly hash, architectures to compare, three seeds `[42,43,44]`, training config, metrics/gates and resource profile. v1 compares majority/rules, pooled embedding, tiny encoder on identical splits. One artifact per task, no multi-task mixed loss. No hidden pretraining on dev/test or inherited model weights.

Training config schema `arbitrium.train.v1` has required fields: architecture_id; task_id; seed; batch_size; epochs; learning_rate; weight_decay; betas; epsilon; warmup_fraction; min_lr_ratio; max_grad_norm; answerability_loss_weight; eval_every_epochs; patience; device; dtype; deterministic; num_workers; max_sequence_length. Unknown keys fail. Initial resolved values are the following; dataset/tokenizer paths are explicit CLI arguments, not machine paths in Git.

| Setting | Default |
|---|---|
| architecture | pooled_embedding_v1, then tiny_encoder_v1 |
| seed | each of 42, 43, 44 in separate runs |
| batch / epochs | 64 / maximum 20 |
| optimizer | AdamW, lr 3e-4, betas 0.9/0.999, epsilon 1e-8 |
| decay | 0.01 on Linear weights only; zero on bias, norm and embeddings |
| scheduler | linear warmup first ceil(0.1*planned_steps), then cosine to 0.1*initial_lr |
| gradient clipping | global norm <=1.0, after backward before step |
| loss | decision CE + 1.0 * answerability BCEWithLogits |
| validation / patience | each epoch / 3 consecutive nonimproving epochs |
| device / dtype | CPU / FP32 |
| data workers | 0, stable sample order before seeded shuffle |
| gradient accumulation | none; do not silently alter effective batch size |

## Loss and optimizer order

For batch B, let A be answerable examples. `L_dec = sum(CE(logits_i,y_i), i in A)/|A|`, or exact zero when |A|=0. `L_ans = sum(BCEWithLogits(a_i,answerable_i))/B`. Total `L=L_dec+L_ans`. No invented decision label contributes gradient for an unanswerable record. No label smoothing or class weights initially because they complicate interpretation; report empirical class priors. CE also handles binary and ordinal anchor classes. The ordinal task additionally reports distance-sensitive metrics, not a second unchosen regression loss.

Per step: zero_grad → tokenize/cache validated batch → forward in train mode → check finite loss → backward → verify finite gradients → clip → optimizer.step → scheduler.step → increment global_step. Nonfinite loss/gradient aborts the run with failure status and last good checkpoint; never skip bad batches invisibly. Smaller final batch is kept. Each epoch shuffles stable sorted sample IDs with recorded C++ RNG state; evaluation uses sorted IDs and eval/no-grad mode.

Dev selection objective is decision NLL averaged over answerable dev records plus answerability BCE averaged over all dev records. Save best on improvement >=1e-4; tie chooses earlier epoch. Early stopping after three nonimproving epochs. Evaluate all three seeds per architecture; choose its representative by lowest dev objective, tie by smaller seed. Compare these two representatives on dev using the preregistered paired family bootstrap in evaluation.md. This interval is a dev-set model-selection heuristic after seed/checkpoint selection, not independent inferential evidence. Select encoder only if its accuracy advantage has a strictly positive lower 95% bound; otherwise select pooled. Report all seeds and both representatives, not only the winner. Once selected, weights are frozen before calibration. No test or calibration-based checkpoint selection.

## Language curriculum and pretraining

The initial hypothesis uses supervised concept curriculum only. GPT-OSS provides English surface variety; learning still occurs through C++ gradient updates. Curriculum iteration changes frozen data, not the Attention implementation or weights during production inference.

Optional language experiment after the first result: masked-language pretraining on train-only approved text, mask 15% of non-control/nonpadding tokens; of selected positions 80% MASK, 10% random nonspecial token, 10% unchanged, CE only selected positions. It requires a new tokenizer/version with MASK control and a tied-vocabulary projection, hence a separately registered architecture/assembly version; it is deliberately excluded from v1 implementation completion. Bounded real English requires explicit permission/license and train-only inclusion. No speculative alternative objective is required to implement v1.

## Reproducibility and resume

Record git commit, clean/dirty diff digest, CMake/compiler/options, exact LibTorch build/distribution digest, dependency lock hashes, CPU/OS/thread count, device/driver if used, seed and RNG states, dataset/tokenizer/task hashes, resolved config, sample order, epoch/step, optimizer/scheduler state, timings and peak memory. Configure deterministic algorithms and fixed CPU threading for the reference run; unsupported deterministic operations fail rather than silently relaxing settings.

Checkpoints at epoch boundaries contain model/optimizer, scheduler fields, RNG states, next epoch and data iterator boundary. Resume requires exact task/tokenizer/data/config/toolchain identity. A mid-epoch interrupted run restarts from the previous completed epoch; record discarded progress and do not append duplicate step metrics. GPU states must be captured if GPU training is explicitly enabled. Cross-version or cross-device bitwise reproducibility is not promised. Same-build resume is verified with a tiny fixture; inference parity tolerances are in [evaluation](evaluation.md).

## Resource policy and exit

CPU training is the baseline even if slow. Batch/architecture/memory estimates are printed before start. OOM aborts explicitly; operator can create a new run with a smaller batch. Do not opportunistically consume another GPU or start a teacher. Teacher generation, student training and teacher review run sequentially by default to avoid shared accelerator contention; device scheduling belongs to the external manager.

A completed train run produces best checkpoint, final checkpoint, history JSONL and train/dev report. It does not produce a promoted inference artifact. Next stages are [calibration](calibration.md), [evaluation](evaluation.md), and [artifact promotion](artifact-format.md).

## Implemented research entry point

Use `make_training_model(model_config, training_config)` before `train_pooled`.
The factory sets the CPU seed **before** model construction. The low-level loop
seeds training randomness but cannot retroactively control initial weights.
See [Experiment 001](experiment-001.md) for the executable offline workflow.
