#include "arbitrium/contracts.h"

#include <cstdlib>
#include <iostream>
#include <limits>

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    const std::string root = ARBITRIUM_SOURCE_DIR;
    auto task = arbitrium::TaskSpec::from_json(arbitrium::parse_json(
        arbitrium::read_file(root + "/docs/examples/recovery-task.json", 4 * 1024 * 1024),
        4 * 1024 * 1024));
    auto request_json = arbitrium::parse_json(
        arbitrium::read_file(root + "/docs/examples/recovery-request.json", 65536));
    auto request = arbitrium::DecisionRequest::from_json(request_json, task);
    check(request.policy == task.policy && request.choices.size() == 3, "valid request");
    auto sample = arbitrium::Sample::from_json(arbitrium::parse_json(
        arbitrium::read_file(root + "/docs/examples/recovery-sample.json", 128 * 1024),
        128 * 1024), task);
    check(sample.answerable && sample.target == 0, "valid sample");
    auto ok_result_json=arbitrium::parse_json(arbitrium::read_file(
        root+"/docs/examples/recovery-result.json",65536));
    auto ok_result=arbitrium::DecisionResult::from_json(ok_result_json,task,request);
    check(ok_result.status=="ok"&&ok_result.verdict=="retry"&&ok_result.probabilities.size()==3,
          "valid ok result");
    auto abstain_result=arbitrium::DecisionResult::from_json(arbitrium::parse_json(
        arbitrium::read_file(root+"/docs/examples/abstain-result.json",65536)),task,request);
    check(abstain_result.status=="abstain"&&abstain_result.reason_code=="low_answerability",
          "valid abstain result");
    auto error_result=arbitrium::DecisionResult::from_json(arbitrium::parse_json(
        arbitrium::read_file(root+"/docs/examples/error-result.json",65536)),task,request);
    check(error_result.status=="error"&&error_result.error->code=="invalid_request",
          "valid error result");
    const auto rejects_result=[&](arbitrium::Json value) {
      try {(void)arbitrium::DecisionResult::from_json(value,task,request);}
      catch(const arbitrium::ContractError&) {return true;}
      return false;
    };
    for(const auto& field: {"request_id","task_id","artifact_id","calibration_id","kind",
                           "verdict","confidence","answerability"}) {
      auto value=ok_result_json;value[field]=nullptr;
      check(rejects_result(value),"required ok result field accepted as null");
    }
    for(const auto& item:ok_result_json.items()) {
      auto value=ok_result_json;value.erase(item.key());
      check(rejects_result(value),"missing result field accepted");
    }
    auto value=ok_result_json;value["extra"]=true;check(rejects_result(value),"extra result field");
    value=ok_result_json;value["schema_version"]="arbitrium.result.v2";
    check(rejects_result(value),"unknown result schema");
    value=ok_result_json;value["probabilities"][0]["probability"]=true;
    check(rejects_result(value),"boolean probability accepted");
    value=ok_result_json;value["confidence"]=std::numeric_limits<double>::infinity();
    check(rejects_result(value),"typed infinity accepted");
    value=ok_result_json;value["score"]=std::numeric_limits<double>::quiet_NaN();
    check(rejects_result(value),"typed NaN accepted");
    auto error_json=arbitrium::parse_json(arbitrium::read_file(root+"/docs/examples/error-result.json",65536));
    value=error_json;value["error"]["retryable"]="false";
    check(rejects_result(value),"string boolean accepted");
    value=error_json;value["error"]["code"]="policy_mismatch";
    check(rejects_result(value),"invented error code accepted");
    std::string unicode_message;for(int i=0;i<1024;++i)unicode_message+="\xc3\xa9";
    value=error_json;value["error"]["message"]=unicode_message;
    check(!rejects_result(value),"1024-code-point error message rejected");
    value["error"]["message"]=unicode_message+"x";
    check(rejects_result(value),"oversized error message accepted");
    value=error_json;value["error"]["message"]="  \r\n";
    check(arbitrium::DecisionResult::from_json(value,task,request).error->message=="  \r\n",
          "schema-valid error message changed");
    // A request permutation cannot change a canonical-order argmax tie.
    auto permuted=request;permuted.choices={"stop","fallback","retry"};
    value=ok_result_json;value["probabilities"]=arbitrium::Json::array({
      {{"label","stop"},{"probability",0.0}},{{"label","fallback"},{"probability",.5}},
      {{"label","retry"},{"probability",.5}}});value["confidence"]=.5;
    check(arbitrium::DecisionResult::from_json(value,task,permuted).verdict=="retry","canonical tie");
    value["verdict"]="fallback";bool tie_rejected=false;
    try {(void)arbitrium::DecisionResult::from_json(value,task,permuted);}
    catch(const arbitrium::ContractError&) {tie_rejected=true;}
    check(tie_rejected,"request-order tie accepted");
    // Binary and ordinal reuse the fixed task-bound result contract.
    auto binary=task;binary.kind="binary";binary.labels={{"false","False",0},{"true","True",0}};
    auto binary_request=request;binary_request.kind="binary";binary_request.choices={"true","false"};
    value=ok_result_json;value["kind"]="binary";value["verdict"]="true";value["confidence"]=.8;
    value["probabilities"]=arbitrium::Json::array({{{"label","true"},{"probability",.8}},
                                                 {{"label","false"},{"probability",.2}}});
    check(arbitrium::DecisionResult::from_json(value,binary,binary_request).verdict=="true","binary result");
    auto ordinal=task;ordinal.kind="ordinal";ordinal.labels={{"low","Low",0},{"high","High",1}};
    auto ordinal_request=request;ordinal_request.kind="ordinal";ordinal_request.choices={"low","high"};
    value=ok_result_json;value["kind"]="ordinal";value["verdict"]="high";value["confidence"]=.75;
    value["score"]=.75;value["probabilities"]=arbitrium::Json::array({
      {{"label","low"},{"probability",.25}},{{"label","high"},{"probability",.75}}});
    check(arbitrium::DecisionResult::from_json(value,ordinal,ordinal_request).score==.75,"ordinal expectation");
    value["score"]=.5;bool score_rejected=false;
    try {(void)arbitrium::DecisionResult::from_json(value,ordinal,ordinal_request);}
    catch(const arbitrium::ContractError&) {score_rejected=true;}
    check(score_rejected,"incorrect ordinal expectation accepted");
    check(arbitrium::sha256("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256");
    bool rejected = false;
    try { (void)arbitrium::parse_json("{\"a\":1,\"a\":2}"); }
    catch (const arbitrium::ContractError&) { rejected = true; }
    check(rejected, "duplicate keys rejected");
    rejected = false;
    try {
      request_json["choices"] = {"retry", "retry", "stop"};
      (void)arbitrium::DecisionRequest::from_json(request_json, task);
    } catch (const arbitrium::ContractError&) { rejected = true; }
    check(rejected, "duplicate labels rejected");
    rejected=false;try{ok_result_json["probabilities"][0]["probability"]=.91;
      (void)arbitrium::DecisionResult::from_json(ok_result_json,task,request);
    }catch(const arbitrium::ContractError&){rejected=true;}check(rejected,"probability sum rejected");
    std::cout << "contracts passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
