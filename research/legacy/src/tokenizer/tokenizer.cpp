#include "arbitrium/tokenizer.h"

#include <set>

namespace arbitrium {
namespace {
void require_tokenizer(bool condition, const char* message) {
  if (!condition) throw ContractError(message, "artifact_invalid");
}
}  // namespace

Tokenizer::Tokenizer(const std::string& model_path, const std::string& config_path) {
  config_ = parse_json(read_file(config_path, 4 * 1024 * 1024), 4 * 1024 * 1024);
  exact_keys(config_, {"schema_version", "sentencepiece_version", "assembly_version",
                       "corpus_sha256", "model_sha256", "actual_vocab_size",
                       "special_ids", "trainer_options"});
  require_tokenizer(config_["schema_version"] == "arbitrium.tokenizer.v1" &&
                        config_["assembly_version"] == "fields.v1",
                    "unsupported tokenizer configuration");
  const auto model = read_file(model_path, 32 * 1024 * 1024);
  require_tokenizer(sha256(model) == config_["model_sha256"].get<std::string>(),
                    "tokenizer hash mismatch");
  auto status = processor_.LoadFromSerializedProto(model);
  require_tokenizer(status.ok(), "invalid SentencePiece model");
  require_tokenizer(processor_.GetPieceSize() == config_["actual_vocab_size"],
                    "tokenizer vocabulary mismatch");
  const auto& ids = config_["special_ids"];
  exact_keys(ids, {"PAD", "UNK", "BOS", "EOS", "CLS", "TASK", "POLICY", "STATE", "QUESTION"});
  require_tokenizer(ids["PAD"] == processor_.pad_id() && ids["UNK"] == processor_.unk_id() &&
                        ids["BOS"] == processor_.bos_id() && ids["EOS"] == processor_.eos_id(),
                    "special token mismatch");
  std::set<int64_t> controls;
  for (const auto* name : {"CLS", "TASK", "POLICY", "STATE", "QUESTION"}) {
    const auto value = id(name);
    require_tokenizer(value >= 0 && value == processor_.PieceToId("<" + std::string(name) + ">"),
                      "control token mismatch");
    require_tokenizer(controls.insert(value).second, "colliding control token");
  }
}

int64_t Tokenizer::id(const std::string& name) const {
  const auto& value = config_["special_ids"].at(name);
  require_tokenizer(value.is_number_integer(), "invalid token ID");
  return value.get<int64_t>();
}

std::vector<int64_t> Tokenizer::control_ids() const {
  return {id("BOS"), id("EOS"), id("CLS"), id("TASK"), id("POLICY"),
          id("STATE"), id("QUESTION")};
}

void Tokenizer::append_control(const std::string& name, int64_t segment,
                               TokenizedInput& output) const {
  output.token_ids.push_back(id(name));
  output.segment_ids.push_back(segment);
}

void Tokenizer::append_encoded(const std::string& text, int64_t segment,
                               TokenizedInput& output) const {
  std::vector<int> pieces;
  const auto status = processor_.Encode(text, &pieces);
  if (!status.ok()) throw ContractError("SentencePiece encode failure", "internal_error");
  for (const auto piece : pieces) {
    require_tokenizer(piece != id("PAD"), "PAD appeared in real input");
    output.token_ids.push_back(piece);
    output.segment_ids.push_back(segment);
  }
}

TokenizedInput Tokenizer::assemble(const DecisionRequest& request, const TaskSpec& task) const {
  if (request.task_id != task.id || request.kind != task.kind || request.policy != task.policy ||
      request.question != task.question) {
    throw ContractError("unvalidated request/task mismatch");
  }
  TokenizedInput output;
  append_control("CLS", 0, output);
  append_control("TASK", 0, output);
  append_encoded(request.task_id, 0, output);
  append_control("POLICY", 1, output);
  append_encoded(request.policy, 1, output);
  append_control("STATE", 2, output);
  append_encoded(request.state, 2, output);
  append_control("QUESTION", 3, output);
  append_encoded(request.question, 3, output);
  output.token_ids.push_back(id("EOS"));
  output.segment_ids.push_back(3);
  if (output.token_ids.size() > 256) throw ContractError("assembled token limit", "input_too_long");
  return output;
}

}  // namespace arbitrium
