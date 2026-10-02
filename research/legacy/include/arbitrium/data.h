#pragma once

#include "arbitrium/contracts.h"
#include "arbitrium/tokenizer.h"
#include <torch/torch.h>

#include <random>

namespace arbitrium {

struct EncodedSample {
  std::string id;
  std::string family_id;
  std::vector<int64_t> token_ids;
  std::vector<int64_t> segment_ids;
  int64_t target{-1};
  bool answerable{false};
};

struct Batch {
  torch::Tensor token_ids;
  torch::Tensor attention_mask;
  torch::Tensor segment_ids;
  torch::Tensor targets;
  torch::Tensor answerable;
};

std::vector<EncodedSample> load_split(const std::string& path, const std::string& split,
                                      const std::string& expected_sha256,
                                      const TaskSpec&, const Tokenizer&);
std::vector<size_t> epoch_order(size_t count, uint64_t seed, int64_t epoch);
Batch make_batch(const std::vector<EncodedSample>&, const std::vector<size_t>& indices,
                 size_t begin, size_t batch_size);

}  // namespace arbitrium
