#pragma once

#include "arbitrium/data.h"
#include "arbitrium/model.h"

namespace arbitrium {

struct TrainingConfig {
  std::string architecture_id{"pooled_embedding_v1"};
  std::string task_id;
  int64_t seed{42}, batch_size{64}, epochs{20};
  double learning_rate{3e-4}, weight_decay{.01}, beta1{.9}, beta2{.999};
  double epsilon{1e-8}, warmup_fraction{.1}, min_lr_ratio{.1}, max_grad_norm{1};
  double answerability_loss_weight{1};
  int64_t eval_every_epochs{1}, patience{3}, num_workers{0}, max_sequence_length{256};
  std::string device{"cpu"}, dtype{"float32"};
  bool deterministic{true};
  void validate() const;
};

struct EpochMetrics {
  int64_t epoch{0}, global_step{0};
  double train_loss{0}, dev_objective{0}, learning_rate{0};
  bool improved{false};
};

struct TrainResult {
  std::vector<EpochMetrics> history;
  int64_t best_epoch{-1};
  double best_dev_objective{0};
  bool early_stopped{false};
  bool interrupted{false};
};

struct TrainRunOptions {
  std::string checkpoint_root;
  std::string identity;
  bool resume{false};
  int64_t max_epochs_this_call{0};  // zero means unbounded by the caller
};

// Construct before training: the training seed owns initial weights as well as loop RNG.
ChoiceModel make_training_model(const ModelConfig&, const TrainingConfig&);

double evaluate_objective(ChoiceModel&, const std::vector<EncodedSample>&, size_t batch_size);
TrainResult train_pooled(ChoiceModel&, const std::vector<EncodedSample>& train,
                         const std::vector<EncodedSample>& dev, const TrainingConfig&,
                         const TrainRunOptions& = {});

}  // namespace arbitrium
