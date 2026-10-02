#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

namespace arbitrium {
struct ModelConfig {
  std::string architecture_id{"pooled_embedding_v1"};
  int64_t vocab_size{8192}, hidden_size{256}, num_choices{3};
  int64_t num_layers{4}, num_heads{4}, ffn_size{1024}, max_sequence_length{256};
  double norm_epsilon{1e-5};
  // IDs are supplied by the validated tokenizer, never inferred from spelling.
  std::vector<int64_t> controls{2,3,4,5,6,7,8};
  double dropout{0.1};
};
struct ModelOutput { torch::Tensor decision_logits, answerability_logit; };

class EncoderBlockImpl final : public torch::nn::Module {
 public:
  explicit EncoderBlockImpl(const ModelConfig& config);
  torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& mask);
 private:
  int64_t heads_, head_dimension_;
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
  torch::nn::Linear q_{nullptr}, k_{nullptr}, v_{nullptr}, output_{nullptr};
  torch::nn::Linear ffn1_{nullptr}, ffn2_{nullptr};
  torch::nn::Dropout attention_dropout_{nullptr}, output_dropout_{nullptr}, ffn_dropout_{nullptr};
};
TORCH_MODULE(EncoderBlock);

class ChoiceModelImpl final : public torch::nn::Module {
 public:
  explicit ChoiceModelImpl(const ModelConfig& config);
  ModelOutput forward(const torch::Tensor& ids, const torch::Tensor& mask,
                      const torch::Tensor& segments);
  const std::string& architecture_id() const { return config_.architecture_id; }
 private:
  ModelConfig config_;
  torch::nn::Embedding embedding_{nullptr};
  torch::nn::Embedding position_embedding_{nullptr}, segment_embedding_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear projection_{nullptr}, decision_{nullptr}, answerability_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};
TORCH_MODULE(ChoiceModel);
// Decision targets on unanswerable rows may be -1 and are never indexed.
torch::Tensor supervised_loss(const ModelOutput&, const torch::Tensor& targets,
                              const torch::Tensor& answerable);
void validate_tensor_input(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,int64_t);
}
