#include "arbitrium/model.h"
#include <iostream>
int main(){torch::manual_seed(42);arbitrium::ChoiceModel model(arbitrium::ModelConfig{});model->eval();torch::NoGradGuard guard;auto ids=torch::tensor({{4,5,9,6,10,7,11,8,12,3}},torch::kInt64);auto out=model->forward(ids,ids.ne(0),torch::zeros_like(ids));std::cout<<"Synthetic uncalibrated logits only; not a DecisionResult:\n"<<out.decision_logits<<'\n';}
