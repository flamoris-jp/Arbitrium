#include "arbitrium/training.h"

#include <cstdlib>
#include <iostream>
#include <filesystem>

int main(){
 try {
  arbitrium::ModelConfig model_config;model_config.vocab_size=64;model_config.dropout=.1;
  arbitrium::ChoiceModel model(model_config);
  std::vector<arbitrium::EncodedSample> train,dev;
  for(int i=0;i<48;++i){
    const int label=i%3;const bool answerable=i%4!=0;
    arbitrium::EncodedSample sample{"s"+std::to_string(i),"f"+std::to_string(i),
      {4,5,9,6,10,7,20+label,8,12,3},{0,0,0,1,1,2,2,3,3,3},answerable?label:-1,answerable};
    (i<36?train:dev).push_back(sample);
  }
  arbitrium::TrainingConfig config;config.task_id="fixture";config.batch_size=12;config.epochs=20;
  config.learning_rate=.02;config.weight_decay=.01;config.patience=5;
  // Factory owns initialization regardless of unrelated prior global RNG use.
  auto seeded_a=arbitrium::make_training_model(model_config,config);
  (void)torch::randn({100});
  auto seeded_b=arbitrium::make_training_model(model_config,config);
  for(const auto& item:seeded_a->named_parameters())
    if(!torch::equal(item.value(),seeded_b->named_parameters()[item.key()]))
      throw std::runtime_error("same-seed initialization mismatch");
  auto other_config=config;other_config.seed=43;
  auto other=arbitrium::make_training_model(model_config,other_config);
  if(torch::equal(seeded_a->named_parameters()["embedding.weight"],other->named_parameters()["embedding.weight"]))
    throw std::runtime_error("different seeds produced identical initial weights");
  const auto before=arbitrium::evaluate_objective(model,dev,12);
  const auto result=arbitrium::train_pooled(model,train,dev,config);
  const auto after=arbitrium::evaluate_objective(model,dev,12);
  if(result.history.empty()||!std::isfinite(after)||after>=before||result.best_epoch<0)throw std::runtime_error("fixture did not learn");
  if(!model->named_parameters()["embedding.weight"][0].eq(0).all().item<bool>())throw std::runtime_error("padding embedding moved");
  // Same-build epoch-boundary resume must equal uninterrupted training.
  config.epochs=4;config.patience=20;
  torch::manual_seed(123);arbitrium::ChoiceModel uninterrupted(model_config);
  const auto uninterrupted_result=arbitrium::train_pooled(uninterrupted,train,dev,config);
  const auto checkpoint_root=std::filesystem::temp_directory_path()/"arbitrium-checkpoint-test";
  std::filesystem::remove_all(checkpoint_root);
  torch::manual_seed(123);arbitrium::ChoiceModel resumed(model_config);
  auto partial=arbitrium::train_pooled(resumed,train,dev,config,{checkpoint_root.string(),"fixture-identity",false,2});
  if(!partial.interrupted)throw std::runtime_error("fixture did not interrupt");
  auto resumed_result=arbitrium::train_pooled(resumed,train,dev,config,{checkpoint_root.string(),"fixture-identity",true,0});
  for(const auto& item:uninterrupted->named_parameters())
    if(!torch::equal(item.value(),resumed->named_parameters()[item.key()]))throw std::runtime_error("resume parameter mismatch");
  if(resumed_result.history.size()!=uninterrupted_result.history.size())throw std::runtime_error("resume history mismatch");
  bool mismatch_rejected=false;try{(void)arbitrium::train_pooled(resumed,train,dev,config,{checkpoint_root.string(),"wrong",true,0});}
  catch(const std::exception&){mismatch_rejected=true;}if(!mismatch_rejected)throw std::runtime_error("identity mismatch accepted");
  std::filesystem::remove_all(checkpoint_root);
  // The same native loop trains the registered encoder only when both architecture IDs agree.
  arbitrium::ModelConfig encoder_config=model_config;encoder_config.architecture_id="tiny_encoder_v1";
  torch::manual_seed(321);arbitrium::ChoiceModel encoder(encoder_config);
  config.architecture_id="tiny_encoder_v1";config.epochs=1;config.patience=3;config.batch_size=12;
  const auto encoder_result=arbitrium::train_pooled(encoder,train,dev,config);
  if(encoder_result.history.size()!=1||!std::isfinite(encoder_result.history[0].train_loss))
    throw std::runtime_error("encoder training path failed");
  config.architecture_id="pooled_embedding_v1";bool architecture_rejected=false;
  try{(void)arbitrium::train_pooled(encoder,train,dev,config);}catch(const std::invalid_argument&){architecture_rejected=true;}
  if(!architecture_rejected)throw std::runtime_error("architecture mismatch accepted");
  std::cout<<"training fixture objective "<<before<<" -> "<<after<<'\n';return EXIT_SUCCESS;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return EXIT_FAILURE;}
}
