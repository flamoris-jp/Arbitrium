#include "arbitrium/postprocess.h"

#include <cmath>
#include <iostream>
#include <limits>

void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}

int main() {
  try {
    using namespace arbitrium;
    auto task=TaskSpec::from_json(parse_json(read_file(
      std::string(ARBITRIUM_SOURCE_DIR)+"/docs/examples/recovery-task.json",65536)));
    auto request=DecisionRequest::from_json(parse_json(read_file(
      std::string(ARBITRIUM_SOURCE_DIR)+"/docs/examples/recovery-request.json",65536)),task);
    CalibratedGates gates{"fixture","calibration-fixture",1,1,false,.5,.5};
    auto result=postprocess_logits(task,request,{4,0,0},4,gates);
    check(result.status=="ok"&&result.verdict=="retry","choice accepted");
    const auto p=result.probabilities[0].probability;
    auto permutation=request;permutation.choices={"stop","fallback","retry"};
    result=postprocess_logits(task,permutation,{4,0,0},4,gates);
    check(result.verdict=="retry"&&result.probabilities[2].label=="retry"&&
          result.probabilities[2].probability==p,"permutation invariance");
    result=postprocess_logits(task,request,{4,0,0},-4,gates);
    check(result.status=="abstain"&&result.reason_code=="low_answerability"&&
          !result.verdict&&!result.confidence&&result.probabilities.size()==3,"answerability gate");
    result=postprocess_logits(task,request,{0,0,0},4,gates);
    check(result.reason_code=="low_confidence","confidence gate");
    result=postprocess_logits(task,request,{0,0,0},-4,gates);
    check(result.reason_code=="low_answerability","answerability has priority");
    gates.accept_none=true;gates.tau_p.reset();gates.tau_q.reset();
    result=postprocess_logits(task,request,{4,0,0},-4,gates);
    check(result.reason_code=="release_gate_not_met","accept-none has priority");
    gates={"fixture","calibration-fixture",1,1,false,.5,.5};
    auto binary=task;binary.kind="binary";binary.labels={{"false","False",0},{"true","True",0}};
    auto binary_request=request;binary_request.kind="binary";binary_request.choices={"true","false"};
    result=postprocess_logits(binary,binary_request,{0,0},0,gates);
    check(result.status=="ok"&&result.verdict=="false"&&result.confidence==.5&&
          result.answerability==.5,"inclusive thresholds and canonical tie");
    auto ordinal=task;ordinal.kind="ordinal";ordinal.labels={{"low","Low",0},{"high","High",1}};
    auto ordinal_request=request;ordinal_request.kind="ordinal";ordinal_request.choices={"high","low"};
    result=postprocess_logits(ordinal,ordinal_request,{0,std::log(3.)},4,gates);
    check(result.verdict=="high"&&std::abs(*result.score-.75)<1e-12,"ordinal expectation");
    gates.decision_temperature=2;gates.answerability_temperature=2;
    result=postprocess_logits(ordinal,ordinal_request,{0,2*std::log(3.)},2*std::log(3.),gates);
    check(std::abs(*result.score-.75)<1e-12&&std::abs(*result.answerability-.75)<1e-12,
          "independent temperatures");
    for(auto bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected=false;try {(void)postprocess_logits(task,request,{bad,0,0},4,gates);}
      catch(const ContractError& e) {rejected=e.code=="nonfinite_output";}
      check(rejected,"nonfinite output rejected");
    }
    gates.tau_p=.51;bool rejected=false;
    try {(void)postprocess_logits(task,request,{4,0,0},4,gates);}
    catch(const ContractError& e) {rejected=e.code=="artifact_invalid";}
    check(rejected,"off-grid threshold rejected");
    std::cout<<"postprocessing contracts passed\n";return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
