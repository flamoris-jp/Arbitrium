#include "arbitrium/model.h"
#include <iostream>
#include <sstream>
void check(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
int main() {
 try {
  torch::set_num_threads(1);torch::manual_seed(42);
  arbitrium::ModelConfig c;c.vocab_size=32;
  arbitrium::ChoiceModel m(c);m->eval();
  auto ids=torch::tensor({{4,5,9,6,10,7,11,8,12,3}},torch::kInt64);
  auto mask=ids.ne(0);auto seg=torch::zeros_like(ids);
  auto original=m->forward(ids,mask,seg);
  auto padded=torch::cat({ids,torch::zeros({1,3},torch::kInt64)},1);
  auto other=m->forward(padded,padded.ne(0),torch::zeros_like(padded));
  check(torch::allclose(original.decision_logits,other.decision_logits,1e-4,1e-5),"padding parity");
  auto batch=ids.repeat({2,1});auto together=m->forward(batch,batch.ne(0),torch::zeros_like(batch));
  check(torch::allclose(original.decision_logits,together.decision_logits[0],1e-4,1e-5),"batch parity");
  auto target=torch::tensor({-1},torch::kInt64);auto no=torch::tensor({false},torch::kBool);
  m->zero_grad();auto loss=arbitrium::supervised_loss(original,target,no);loss.backward();
  check(m->named_parameters()["decision.weight"].grad().eq(0).all().item<bool>(),"unanswerable decision gradient");
  torch::optim::AdamW optimizer(m->parameters(),torch::optim::AdamWOptions(0.01));
  target=torch::tensor({1},torch::kInt64);auto yes=~no;
  m->train();for(int i=0;i<50;++i){optimizer.zero_grad();auto out=m->forward(ids,mask,seg);auto l=arbitrium::supervised_loss(out,target,yes);l.backward();for(auto p:m->parameters())if(p.grad().defined())check(torch::isfinite(p.grad()).all().item<bool>(),"finite gradient");optimizer.step();}
  m->eval();auto fitted=m->forward(ids,mask,seg);
  check(fitted.decision_logits.softmax(-1)[0][1].item<float>()>0.99,"tiny fixture overfit");
  check(m->named_parameters()["embedding.weight"][0].eq(0).all().item<bool>(),"padding row zero");
  torch::serialize::OutputArchive archive;m->save(archive);std::stringstream bytes;archive.save_to(bytes);
  arbitrium::ChoiceModel loaded(c);torch::serialize::InputArchive input;input.load_from(bytes);loaded->load(input);loaded->eval();
  check(torch::allclose(fitted.decision_logits,loaded->forward(ids,mask,seg).decision_logits,1e-4,1e-5),"native archive parity");
  bool rejected=false;try{m->forward(torch::tensor({{4,0,9}},torch::kInt64),torch::tensor({{true,false,true}},torch::kBool),torch::zeros({1,3},torch::kInt64));}catch(const c10::Error&){rejected=true;}check(rejected,"hole mask rejected");
  // The v1 encoder is repository-native explicit Q/K/V, not a pretrained module.
  arbitrium::ModelConfig econfig;econfig.vocab_size=32;econfig.architecture_id="tiny_encoder_v1";
  torch::manual_seed(42);arbitrium::ChoiceModel encoder(econfig);encoder->eval();
  auto encoded=encoder->forward(ids,mask,seg);
  auto encoded_padded=encoder->forward(padded,padded.ne(0),torch::zeros_like(padded));
  check(torch::allclose(encoded.decision_logits,encoded_padded.decision_logits,1e-4,1e-5),"encoder padding parity");
  auto encoded_batch=encoder->forward(batch,batch.ne(0),torch::zeros_like(batch));
  check(torch::allclose(encoded.decision_logits,encoded_batch.decision_logits[0],1e-4,1e-5),"encoder batch parity");
  check(torch::equal(encoded.decision_logits,encoder->forward(ids,mask,seg).decision_logits),"encoder eval repeatability");
  encoder->train();encoder->zero_grad();
  auto encoder_loss=arbitrium::supervised_loss(encoder->forward(ids,mask,seg),target,yes);encoder_loss.backward();
  for(const auto* name:{"blocks.0.q.weight","blocks.0.k.weight","blocks.0.v.weight","decision.weight","answerability.weight"}) {
    const auto gradient=encoder->named_parameters()[name].grad();
    check(gradient.defined()&&torch::isfinite(gradient).all().item<bool>()&&gradient.abs().sum().item<float>()>0,name);
  }
  encoder->eval();torch::serialize::OutputArchive encoder_archive;encoder->save(encoder_archive);
  std::stringstream encoder_bytes;encoder_archive.save_to(encoder_bytes);
  arbitrium::ChoiceModel encoder_loaded(econfig);torch::serialize::InputArchive encoder_input;
  encoder_input.load_from(encoder_bytes);encoder_loaded->load(encoder_input);encoder_loaded->eval();
  check(torch::equal(encoder->forward(ids,mask,seg).decision_logits,
                     encoder_loaded->forward(ids,mask,seg).decision_logits),"encoder native archive parity");
  std::cout<<"pooled and encoder gradients, masking, overfit, native archive parity passed\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
