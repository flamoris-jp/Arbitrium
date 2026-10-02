#include "arbitrium/composition.h"
#include "maidionis/training.h"
#include "maidionis/artifact.h"
#include <iostream>
#include <limits>
#include <sys/resource.h>
using namespace maidionis;
int main(int argc,char** argv){try {
  if(argc==2&&std::string(argv[1])=="binding-self-test"){
    auto t=arbitrium::trusted();if(t.size()!=4)throw std::runtime_error("trusted source coverage");
    std::cout<<"four inventoried sources compiled; no Core Decision registration\n";return 0;
  }
  if(argc==4&&std::string(argv[1])=="encode"){
    auto c=arbitrium::composition(argv[2]);auto x=parse_json(read_file(argv[3],65536));auto batch=c.encode({x});
    Json ids=Json::array(),segments=Json::array();for(int64_t i=0;i<batch.inputs.size(1);++i){ids.push_back(batch.inputs[0][i].item<int64_t>());segments.push_back(batch.segments[0][i].item<int64_t>());}
    std::cout<<canonical(Json{{"ids",ids},{"segments",segments}});return 0;
  }
  if(argc==2&&std::string(argv[1])=="build-identity"){std::cout<<canonical(Json{{"build_digest",std::string(ARBITRIUM_BUILD_DIGEST)}});return 0;}
  if(argc==3&&std::string(argv[1])=="composition-self-test"){
    auto c=arbitrium::composition(argv[2]);
    auto logits=torch::zeros({2,4},torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    Batch batch;batch.targets=torch::full({2},-123,torch::kInt64);batch.target_mask=torch::zeros({2,1},torch::kFloat32);
    auto loss=c.objective(logits,batch);loss.backward();auto gradient=logits.grad();
    if(!torch::isfinite(loss).item<bool>()||!gradient.slice(1,0,3).eq(0).all().item<bool>()||!gradient.slice(1,3,4).gt(0).all().item<bool>())throw std::runtime_error("masked Decision/answerability gradient");
    if(c.decode(torch::zeros({4},torch::kFloat32))["diagnostic_label"]!="retry")throw std::runtime_error("canonical Decision tie");
    if(c.decode(torch::tensor({0.f,1e-8f,0.f,0.f}))["diagnostic_label"]!="fallback")throw std::runtime_error("rounded Decision tie");
    for(float value:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),1000001.f}){
      auto bad=torch::zeros({4},torch::kFloat32);bad[0]=value;bool rejected=false;
      try{c.decode(bad);}catch(const std::invalid_argument&){rejected=true;}
      if(!rejected)throw std::runtime_error("unbounded/nonfinite Decision logits accepted");
    }
    std::cout<<canonical(Json{{"masked_gradient",true},{"bounded_decode",true},{"canonical_tie",true},{"near_tie",true}});return 0;
  }
  if(argc==10&&std::string(argv[1])=="infer"){
    auto c=arbitrium::composition(argv[2]);ValidationContext validation{argv[4],true,"offline_evaluation",size_t(std::stoull(argv[5]))};
    auto bundle=validate_bundle(argv[3],c,validation);
    bool admitted=std::string(argv[9])!="expired";
    OfflineContext context{"test:offline-load","linux.cpu.fp32.serial.v1",size_t(std::stoull(argv[6])),size_t(std::stoull(argv[7])),size_t(std::stoull(argv[8])),[&]{return admitted;},
      [&](const std::string& stage){if(std::string(argv[9])==stage)throw std::runtime_error("injected materialization failure");if(std::string(argv[9])=="expire_after_load"&&stage=="after_load")admitted=false;}};
    auto loaded=materialize_model(bundle,c,context);
    auto rows=parse_json(immutable_read(std::filesystem::path(argv[2])/"inference-inputs.json",4*1024*1024),4*1024*1024);
    if(!rows.is_array()||rows.size()>1000000)throw std::invalid_argument("inference input bound");
    Json predictions=Json::array();torch::NoGradGuard no;
    for(const auto& request:rows){c.registry.request(request);auto batch=c.encode({request["payload"]});auto logits=loaded.model->forward(batch).flatten();
      auto output=c.decode(logits);Json ordered=Json::array();for(const auto& label:request["payload"]["choices"])for(const auto& p:output["probabilities"])if(p["label"]==label)ordered.push_back(p);output["probabilities"]=ordered;
      Json result={{"schema_version","maidionis.result.v1"},{"request_id",request["request_id"]},{"artifact_digest",bundle.digest()},
       {"context_ref",request["context_ref"]},{"payload_schema",c.registry.descriptor()["output_schema"]},{"status","abstain"},{"payload",nullptr},{"diagnostics",output},{"error",nullptr}};
      for(const auto& k:{"specialization_id","specialization_version","task_id","task_version"})result[k]=request[k];
      c.registry.result(result,request,bundle.digest());predictions.push_back({{"sample_id",request["request_id"]},{"raw_output",Json::array({logits[0].item<double>(),logits[1].item<double>(),logits[2].item<double>(),logits[3].item<double>()})},{"result",result}});
    }
    std::cout<<canonical(Json{{"predictions",predictions},{"receipt",loaded.receipt},{"component_digest",bundle.manifest()["evidence"]["evaluation"]["evaluated_component_digest"]}});return 0;
  }
  if(argc==6&&std::string(argv[1])=="validate"){
    auto before=model_construction_count();
    auto c=arbitrium::composition(argv[2]);auto bundle=validate_bundle(argv[3],c,ValidationContext{argv[4],true,argv[5],512*1024*1024});
    auto delta=model_construction_count()-before;if(delta)throw std::runtime_error("metadata validation constructed a model");
    std::cout<<canonical(Json{{"manifest_digest",bundle.digest()},{"parameter_bytes",bundle.parameter_bytes()},{"model_constructions",delta}});return 0;
  }
  if(argc!=10||std::string(argv[1])!="train")throw std::invalid_argument("usage: arbitrium_driver train dataset digest checkpoint-root output epochs-this-call resume config-json fault");
  auto c=arbitrium::composition(argv[2]);auto data=training_data(argv[2],argv[3],c);TrainingConfig config;auto cfg=parse_json(read_file(argv[8],65536));
  if(cfg.size()!=8||cfg["schema_version"]!="arbitrium.research-training.v1")throw std::invalid_argument("training config shape");
  config.seed=cfg.at("seed");config.epochs=cfg.at("epochs");config.learning_rate=cfg.at("learning_rate");config.batch_size=cfg.at("batch_size");config.patience=cfg.at("patience");config.weight_decay=cfg.at("weight_decay");config.selection_scope=cfg.at("selection_scope");
  auto initial=make_seeded_model(c.model_config,config.seed);auto initial_bytes=save_model_bytes(initial);
  auto result=train(c,data,config,argv[4],std::string(argv[7])=="1",std::stoll(argv[6]),[&](const std::string& stage){if(std::string(argv[9])!="none"&&stage==argv[9])throw std::runtime_error("injected training publication fault");});
  Files files={{"final.pt",save_model_bytes(result.final_model)},{"best.pt",save_model_bytes(result.best_model)},
    {"state.json",canonical(result.state)},{"model.config.json",canonical(c.model_config.json())},{"training.json",canonical(Json{{"training_run_id","arbitrium:train:"+std::to_string(config.seed)},{"config",config.json()},{"state",result.state},
    {"checkpoint_digest",result.checkpoint_digest},{"dataset_digest",data.manifest_digest()},{"environment_digest",sha256(numerical_environment())},
    {"descriptor_digest",sha256(canonical(c.registry.descriptor()))},{"build_digest",c.registry.build_digest()},{"model_config",c.model_config.json()},
    {"weights_digest",sha256(save_model_bytes(result.best_model))}})}};
  Json summary={{"state",result.state},{"gradient_l1",result.gradient_l1},{"parameters_changed",initial_bytes!=files["final.pt"]},{"environment_digest",sha256(numerical_environment())},
    {"final_sha256",sha256(files["final.pt"])},{"best_sha256",sha256(files["best.pt"])},{"environment",parse_json(numerical_environment())},
    {"tensor_inventory",tensor_inventory(result.best_model)},{"parameter_count",[&]{int64_t n=0;for(const auto& p:result.best_model->parameters())n+=p.numel();return n;}()}};
  result.final_model->eval();result.best_model->eval();Json final_logits=Json::array(),best_logits=Json::array();
  torch::NoGradGuard no;
  for(const auto& row:data.training_rows()){auto batch=c.encode({row});auto a=result.final_model->forward(batch).flatten(),b=result.best_model->forward(batch).flatten();
    final_logits.push_back({a[0].item<double>(),a[1].item<double>(),a[2].item<double>(),a[3].item<double>()});best_logits.push_back({b[0].item<double>(),b[1].item<double>(),b[2].item<double>(),b[3].item<double>()});}
  summary["dev_logits"]=Json::array();for(const auto& row:data.development_rows()){auto batch=c.encode({row});auto logits=result.best_model->forward(batch).flatten();summary["dev_logits"].push_back({logits[0].item<double>(),logits[1].item<double>(),logits[2].item<double>(),logits[3].item<double>()});}
  summary["final_logits"]=final_logits;summary["best_logits"]=best_logits;files["summary.json"]=canonical(summary);publish_tree(argv[5],files);std::cout<<canonical(summary);return 0;
}catch(const std::exception& e){std::cerr<<"native operation rejected: "<<e.what()<<"\n";return 1;}}
