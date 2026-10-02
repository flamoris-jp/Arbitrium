#include "arbitrium/calibration.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

void metric_check(bool value, const char* message) { if(!value)throw std::runtime_error(message); }

int main(){
 try {
  using arbitrium::ScoredExample;
  std::vector<ScoredExample> values{{{.9,.1},0,true,.9,true},{{.2,.8},1,true,.8,true},{{.5,.5},-1,false,.1,false}};
  auto metrics=arbitrium::compute_metrics(values,2);
  metric_check(metrics.accuracy==1&&metrics.macro_f1==1&&metrics.coverage>0.66&&metrics.selective_risk==0,"basic metrics");
  metric_check(std::abs(metrics.brier-.05)<1e-12,"multiclass brier");
  metric_check(arbitrium::ranked_probability_score(values,2)>.02,"RPS");
  metric_check(arbitrium::wilson_upper(0,200)<.05,"Wilson gate");
  std::vector<std::vector<double>> logits{{4,0},{0,4}};std::vector<int64_t> targets{0,1};
  auto decision=arbitrium::fit_decision_temperature(logits,targets);
  metric_check(decision.converged&&decision.fitted_nll<=decision.raw_nll,"decision temperature");
  auto answer=arbitrium::fit_answerability_temperature({4,-4},{true,false});
  metric_check(answer.converged&&answer.fitted_nll<=answer.raw_nll,"answerability temperature");
  std::vector<ScoredExample> gate;
  for(int i=0;i<200;++i)gate.push_back({{.99,.01},0,true,.99,false});
  for(int i=0;i<200;++i)gate.push_back({{.99,.01},-1,false,.01,false});
  auto selected=arbitrium::select_gate(gate);
  metric_check(!selected.accept_none&&selected.accepted==200&&selected.errors==0&&selected.coverage==.5,"gate selection");
  metric_check(arbitrium::select_gate({values[0]}).accept_none,"no valid gate");
  std::cout<<"metrics and calibration passed\n";return EXIT_SUCCESS;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return EXIT_FAILURE;}
}
