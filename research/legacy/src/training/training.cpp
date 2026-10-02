#include "arbitrium/training.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

namespace arbitrium {
namespace {
double schedule(const TrainingConfig& config, int64_t step, int64_t planned_steps) {
  const auto warmup = static_cast<int64_t>(std::ceil(config.warmup_fraction * planned_steps));
  if (step <= warmup) return config.learning_rate * static_cast<double>(step) / std::max<int64_t>(1, warmup);
  const auto progress = static_cast<double>(step - warmup) / std::max<int64_t>(1, planned_steps - warmup);
  const auto cosine = .5 * (1 + std::cos(std::numbers::pi * std::min(1.0, progress)));
  return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine);
}

void finite_gradients(ChoiceModel& model) {
  for (const auto& parameter : model->named_parameters())
    if (parameter.value().grad().defined() && !torch::isfinite(parameter.value().grad()).all().item<bool>())
      throw std::runtime_error("nonfinite gradient");
}

void decoupled_weight_decay(ChoiceModel& model, double amount) {
  torch::NoGradGuard guard;
  for (const auto& parameter : model->named_parameters()) {
    const auto& name=parameter.key();
    const bool embedding=name=="embedding.weight"||name=="position_embedding.weight"||
                         name=="segment_embedding.weight";
    if (name.ends_with("weight")&&!embedding&&name.find("norm")==std::string::npos)
      parameter.value().mul_(1 - amount);
  }
}

void durable_write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("checkpoint write failure");
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); output.flush();
  if (!output) throw std::runtime_error("checkpoint write failure"); output.close();
  const auto file = ::open(path.c_str(), O_RDONLY);
  if (file < 0 || ::fsync(file) != 0) { if (file >= 0) ::close(file); throw std::runtime_error("checkpoint fsync failure"); }
  ::close(file);
}

Json history_json(const std::vector<EpochMetrics>& history) {
  Json value=Json::array();for(const auto& row:history)value.push_back({
    {"epoch",row.epoch},{"global_step",row.global_step},{"train_loss",row.train_loss},
    {"dev_objective",row.dev_objective},{"learning_rate",row.learning_rate},{"improved",row.improved}});
  return value;
}

void save_checkpoint(const TrainRunOptions& options, ChoiceModel& model, torch::optim::AdamW& optimizer,
                     const TrainResult& result, int64_t completed_epoch, int64_t global_step,
                     int64_t nonimproving) {
  if(options.checkpoint_root.empty())return;
  namespace fs=std::filesystem;const fs::path root=options.checkpoint_root;
  fs::create_directories(root);const auto name="epoch-"+std::to_string(completed_epoch+1);
  const fs::path final=root/name,temp=root/("."+name+".incomplete");
  if(fs::exists(final)||fs::exists(temp))throw std::runtime_error("checkpoint path conflict");
  fs::create_directory(temp);
  try {
    torch::save(model,temp/"model.pt");torch::save(optimizer,temp/"optimizer.pt");
    auto rng=at::detail::getDefaultCPUGenerator().get_state();torch::save(rng,temp/"rng.pt");
    Json state={{"schema_version","arbitrium.checkpoint.v1"},{"identity",options.identity},
      {"completed_epoch",completed_epoch},{"next_epoch",completed_epoch+1},{"global_step",global_step},
      {"best_dev_objective",result.best_dev_objective},{"best_epoch",result.best_epoch},
      {"patience_counter",nonimproving},{"history",history_json(result.history)}};
    durable_write(temp/"state.json",state.dump()+"\n");
    for(const auto& file:{"model.pt","optimizer.pt","rng.pt"}) {
      const auto descriptor=::open((temp/file).c_str(),O_RDONLY);
      if(descriptor<0||::fsync(descriptor)!=0){if(descriptor>=0)::close(descriptor);throw std::runtime_error("checkpoint fsync failure");}
      ::close(descriptor);
    }
    const auto directory=::open(temp.c_str(),O_RDONLY|O_DIRECTORY);if(directory<0||::fsync(directory)!=0){if(directory>=0)::close(directory);throw std::runtime_error("checkpoint directory fsync failure");}::close(directory);
    fs::rename(temp,final);
    const auto state_bytes=read_file((final/"state.json").string(),4*1024*1024);
    const fs::path latest_temp=root/".latest.json.incomplete",latest=root/"latest.json";
    durable_write(latest_temp,Json{{"checkpoint",name},{"state_sha256",sha256(state_bytes)}}.dump()+"\n");
    fs::rename(latest_temp,latest);
    const auto root_fd=::open(root.c_str(),O_RDONLY|O_DIRECTORY);if(root_fd>=0){::fsync(root_fd);::close(root_fd);}
  }catch(...){fs::remove_all(temp);throw;}
}

void load_checkpoint(const TrainRunOptions& options, ChoiceModel& model, torch::optim::AdamW& optimizer,
                     TrainResult& result, int64_t& start_epoch, int64_t& global_step,
                     int64_t& nonimproving) {
  namespace fs=std::filesystem;const fs::path root=options.checkpoint_root;
  auto latest=parse_json(read_file((root/"latest.json").string(),4*1024*1024),4*1024*1024);
  exact_keys(latest,{"checkpoint","state_sha256"});const fs::path checkpoint=root/latest["checkpoint"].get<std::string>();
  auto state_bytes=read_file((checkpoint/"state.json").string(),4*1024*1024);
  if(sha256(state_bytes)!=latest["state_sha256"].get<std::string>())throw std::runtime_error("checkpoint state hash mismatch");
  auto state=parse_json(state_bytes,4*1024*1024);
  exact_keys(state,{"schema_version","identity","completed_epoch","next_epoch","global_step",
                    "best_dev_objective","best_epoch","patience_counter","history"});
  if(state["schema_version"]!="arbitrium.checkpoint.v1"||state["identity"]!=options.identity)
    throw std::runtime_error("checkpoint identity mismatch");
  torch::load(model,checkpoint/"model.pt");torch::load(optimizer,checkpoint/"optimizer.pt");
  torch::Tensor rng;torch::load(rng,checkpoint/"rng.pt");
  auto cpu_generator=at::detail::getDefaultCPUGenerator();cpu_generator.set_state(rng);
  start_epoch=state["next_epoch"];global_step=state["global_step"];nonimproving=state["patience_counter"];
  result.best_dev_objective=state["best_dev_objective"];result.best_epoch=state["best_epoch"];
  for(const auto& row:state["history"])result.history.push_back({row["epoch"],row["global_step"],row["train_loss"],row["dev_objective"],row["learning_rate"],row["improved"]});
}
}  // namespace

void TrainingConfig::validate() const {
  if ((architecture_id != "pooled_embedding_v1" && architecture_id != "tiny_encoder_v1") || task_id.empty() ||
      seed < 0 || batch_size < 1 || batch_size > 64 || epochs < 1 ||
      !(learning_rate > 0) || weight_decay < 0 || beta1 <= 0 || beta1 >= 1 ||
      beta2 <= 0 || beta2 >= 1 || !(epsilon > 0) || warmup_fraction < 0 ||
      warmup_fraction > 1 || min_lr_ratio < 0 || min_lr_ratio > 1 ||
      !(max_grad_norm > 0) || answerability_loss_weight != 1 || eval_every_epochs != 1 ||
      patience < 1 || device != "cpu" || dtype != "float32" || !deterministic ||
      num_workers != 0 || max_sequence_length != 256)
    throw std::invalid_argument("unsupported v1 training configuration");
}

ChoiceModel make_training_model(const ModelConfig& model_config, const TrainingConfig& config) {
  config.validate();
  if (model_config.architecture_id != config.architecture_id)
    throw std::invalid_argument("training architecture mismatch");
  torch::set_num_threads(1);
  torch::globalContext().setDeterministicAlgorithms(true, false);
  torch::manual_seed(config.seed);
  return ChoiceModel(model_config);
}

double evaluate_objective(ChoiceModel& model, const std::vector<EncodedSample>& samples,
                          size_t batch_size) {
  if (samples.empty()) throw std::invalid_argument("empty dev set");
  const auto was_training = model->is_training(); model->eval(); torch::NoGradGuard guard;
  std::vector<size_t> order(samples.size()); std::iota(order.begin(), order.end(), 0);
  double decision_sum = 0, answer_sum = 0; int64_t answerable_count = 0;
  for (size_t begin=0;begin<order.size();begin+=batch_size) {
    const auto batch=make_batch(samples,order,begin,batch_size);
    const auto output=model->forward(batch.token_ids,batch.attention_mask,batch.segment_ids);
    auto selected=batch.answerable.nonzero().squeeze(1);
    if(selected.numel()) {
      decision_sum += torch::nn::functional::cross_entropy(
        output.decision_logits.index_select(0,selected),batch.targets.index_select(0,selected),
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)).item<double>();
      answerable_count+=selected.numel();
    }
    answer_sum += torch::nn::functional::binary_cross_entropy_with_logits(
      output.answerability_logit,batch.answerable.to(torch::kFloat32),
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(
        torch::kSum)).item<double>();
  }
  if(was_training)model->train();
  return (answerable_count ? decision_sum/answerable_count : 0) + answer_sum/samples.size();
}

TrainResult train_pooled(ChoiceModel& model,const std::vector<EncodedSample>& train,
                         const std::vector<EncodedSample>& dev,const TrainingConfig& config,
                         const TrainRunOptions& options) {
  config.validate();if(train.empty()||dev.empty())throw std::invalid_argument("empty split");
  if(model->architecture_id()!=config.architecture_id)throw std::invalid_argument("training architecture mismatch");
  if((!options.checkpoint_root.empty()&&options.identity.empty())||options.max_epochs_this_call<0)
    throw std::invalid_argument("invalid run options");
  torch::set_num_threads(1);torch::manual_seed(config.seed);
  torch::globalContext().setDeterministicAlgorithms(true,false);
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(config.learning_rate)
    .betas({config.beta1,config.beta2}).eps(config.epsilon).weight_decay(0));
  const auto steps_per_epoch=static_cast<int64_t>((train.size()+config.batch_size-1)/config.batch_size);
  const auto planned_steps=steps_per_epoch*config.epochs;
  TrainResult result;result.best_dev_objective=std::numeric_limits<double>::infinity();
  int64_t global_step=0,nonimproving=0,start_epoch=0,epochs_this_call=0;
  if(options.resume)load_checkpoint(options,model,optimizer,result,start_epoch,global_step,nonimproving);
  for(int64_t epoch=start_epoch;epoch<config.epochs;++epoch) {
    model->train();const auto order=epoch_order(train.size(),config.seed,epoch);double loss_sum=0;int64_t batches=0;
    for(size_t begin=0;begin<order.size();begin+=config.batch_size) {
      optimizer.zero_grad();const auto batch=make_batch(train,order,begin,config.batch_size);
      const auto output=model->forward(batch.token_ids,batch.attention_mask,batch.segment_ids);
      const auto loss=supervised_loss(output,batch.targets,batch.answerable);
      if(!torch::isfinite(loss).item<bool>())throw std::runtime_error("nonfinite loss");
      loss.backward();finite_gradients(model);
      torch::nn::utils::clip_grad_norm_(model->parameters(),config.max_grad_norm);
      ++global_step;const auto lr=schedule(config,global_step,planned_steps);
      static_cast<torch::optim::AdamWOptions&>(optimizer.param_groups()[0].options()).lr(lr);
      decoupled_weight_decay(model,lr*config.weight_decay);optimizer.step();
      // Keep the fixed padding embedding invariant even with optimizer state.
      {torch::NoGradGuard no_grad;model->named_parameters()["embedding.weight"][0].zero_();}
      loss_sum+=loss.item<double>();++batches;
    }
    const auto dev_objective=evaluate_objective(model,dev,config.batch_size);
    const bool improved=dev_objective<=result.best_dev_objective-1e-4;
    if(improved){result.best_dev_objective=dev_objective;result.best_epoch=epoch;nonimproving=0;}else ++nonimproving;
    result.history.push_back({epoch,global_step,loss_sum/batches,dev_objective,
                              schedule(config,global_step,planned_steps),improved});
    save_checkpoint(options,model,optimizer,result,epoch,global_step,nonimproving);
    ++epochs_this_call;
    if(nonimproving>=config.patience){result.early_stopped=true;break;}
    if(options.max_epochs_this_call&&epochs_this_call>=options.max_epochs_this_call&&epoch+1<config.epochs){result.interrupted=true;break;}
  }
  return result;
}

}  // namespace arbitrium
