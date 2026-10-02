#include "arbitrium/postprocess.h"
#include "arbitrium/calibration.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace arbitrium {
namespace {
bool grid_threshold(const std::optional<double>& value) {
  static const std::vector<double> grid{.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99};
  return value && std::find(grid.begin(),grid.end(),*value)!=grid.end();
}
bool valid_temperature(double value) {
  return std::isfinite(value)&&value>=std::exp(-4)&&value<=std::exp(4);
}
}  // namespace

DecisionResult postprocess_logits(const TaskSpec& task,const DecisionRequest& request,
                                  const std::vector<double>& logits,double answer_logit,
                                  const CalibratedGates& gates) {
  // Revalidate typed request values as well as their exact task binding.
  const Json request_json={{"schema_version","arbitrium.request.v1"},
    {"request_id",request.id},{"task_id",request.task_id},{"kind",request.kind},
    {"language","en"},{"state",request.state},{"policy",request.policy},
    {"question",request.question},{"choices",request.choices}};
  (void)DecisionRequest::from_json(request_json,task);
  valid_id(gates.artifact_id);valid_id(gates.calibration_id);
  if(!valid_temperature(gates.decision_temperature)||
     !valid_temperature(gates.answerability_temperature)||
     (gates.accept_none ? gates.tau_p.has_value()||gates.tau_q.has_value() :
                          !grid_threshold(gates.tau_p)||!grid_threshold(gates.tau_q)))
    throw ContractError("invalid calibrated gates","artifact_invalid");
  if(logits.size()!=task.labels.size()||logits.size()<2||logits.size()>16)
    throw ContractError("decision logit shape mismatch","internal_error");
  if(!std::isfinite(answer_logit)||!std::all_of(logits.begin(),logits.end(),
                                               [](double v){return std::isfinite(v);}))
    throw ContractError("nonfinite model output","nonfinite_output");
  const auto probabilities=calibrated_softmax(logits,gates.decision_temperature);
  const auto answerability=calibrated_sigmoid(answer_logit,gates.answerability_temperature);
  // std::max_element selects the first maximum: canonical task order wins ties.
  const auto winner=static_cast<size_t>(std::distance(probabilities.begin(),
                         std::max_element(probabilities.begin(),probabilities.end())));
  const auto confidence=probabilities[winner];
  Json result={{"schema_version","arbitrium.result.v1"},{"request_id",request.id},
    {"task_id",task.id},{"artifact_id",gates.artifact_id},{"kind",task.kind},
    {"status","ok"},{"verdict",task.labels[winner].id},{"probabilities",Json::array()},
    {"score",nullptr},{"confidence",confidence},{"answerability",answerability},
    {"reason_code",nullptr},{"calibration_id",gates.calibration_id},{"error",nullptr}};
  std::map<std::string,double> keyed;
  double score=0;
  for(size_t i=0;i<task.labels.size();++i) {
    keyed.emplace(task.labels[i].id,probabilities[i]);
    if(task.kind=="ordinal")score+=probabilities[i]*task.labels[i].anchor;
  }
  for(const auto& label:request.choices)
    result["probabilities"].push_back({{"label",label},{"probability",keyed.at(label)}});
  if(task.kind=="ordinal")result["score"]=score;
  if(gates.accept_none)result["reason_code"]="release_gate_not_met";
  else if(answerability<*gates.tau_q)result["reason_code"]="low_answerability";
  else if(confidence<*gates.tau_p)result["reason_code"]="low_confidence";
  if(!result["reason_code"].is_null()) {
    result["status"]="abstain";result["verdict"]=nullptr;
    result["score"]=nullptr;result["confidence"]=nullptr;
  }
  return DecisionResult::from_json(result,task,request);
}

}  // namespace arbitrium
