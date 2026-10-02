#pragma once
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace arbitrium {
using Json = nlohmann::json;
struct ContractError : std::runtime_error {
  std::string code;
  explicit ContractError(std::string message, std::string category = "invalid_request")
      : std::runtime_error(std::move(message)), code(std::move(category)) {}
};
Json parse_json(const std::string& bytes, size_t limit = 65536);
std::string normalized_text(const Json& value, size_t limit);
void exact_keys(const Json&, const std::vector<std::string>& keys);
void valid_id(const Json&);
struct Label { std::string id, description; double anchor = 0; };
struct TaskSpec {
  std::string id, kind, question, policy;
  std::vector<Label> labels;
  static TaskSpec from_json(const Json&);
};
struct DecisionRequest {
  std::string id, task_id, kind, state, policy, question;
  std::vector<std::string> choices;
  static DecisionRequest from_json(const Json&, const TaskSpec&);
};
struct Probability { std::string label; double probability; };
struct ResultError { std::string code; bool retryable; std::string message; };
struct DecisionResult {
  std::optional<std::string> request_id, task_id, artifact_id, kind, verdict;
  std::string status;
  std::vector<Probability> probabilities;
  std::optional<double> score, confidence, answerability;
  std::optional<std::string> reason_code, calibration_id;
  std::optional<ResultError> error;
  static DecisionResult from_json(const Json&, const TaskSpec&, const DecisionRequest&);
};
struct Sample {
  std::string id, family_id, split, provenance_id;
  DecisionRequest input;
  bool answerable;
  int64_t target;
  static Sample from_json(const Json&, const TaskSpec&);
};
std::string sha256(const std::string&);
std::string read_file(const std::string&, size_t limit);
}
