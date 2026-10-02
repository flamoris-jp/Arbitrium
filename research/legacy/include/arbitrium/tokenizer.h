#pragma once

#include "arbitrium/contracts.h"
#include <sentencepiece_processor.h>

namespace arbitrium {

struct TokenizedInput {
  std::vector<int64_t> token_ids;
  std::vector<int64_t> segment_ids;
};

class Tokenizer final {
 public:
  Tokenizer(const std::string& model_path, const std::string& config_path);
  TokenizedInput assemble(const DecisionRequest& request, const TaskSpec& task) const;
  int64_t vocab_size() const { return processor_.GetPieceSize(); }
  std::vector<int64_t> control_ids() const;

 private:
  sentencepiece::SentencePieceProcessor processor_;
  Json config_;
  int64_t id(const std::string& name) const;
  void append_encoded(const std::string&, int64_t, TokenizedInput&) const;
  void append_control(const std::string&, int64_t, TokenizedInput&) const;
};

}  // namespace arbitrium
