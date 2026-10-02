// Experiment-specific research runner; this is not the release CLI/schema.
#include "arbitrium/training.h"
#include "arbitrium/calibration.h"
#include <torch/version.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

using namespace arbitrium;
namespace fs = std::filesystem;

Json score(ChoiceModel& model, const std::vector<EncodedSample>& samples) {
  model->eval(); torch::NoGradGuard guard;
  std::vector<size_t> order(samples.size()); std::iota(order.begin(), order.end(), 0);
  std::vector<ScoredExample> scored;
  size_t correct_answerability = 0, answerable_predictions = 0;
  size_t confusion[3][3] = {{0}};
  size_t answerability_confusion[2][2] = {{0}};
  for (size_t begin = 0; begin < samples.size(); begin += 64) {
    const auto batch = make_batch(samples, order, begin, 64);
    const auto output = model->forward(batch.token_ids, batch.attention_mask, batch.segment_ids);
    const auto probabilities = torch::softmax(output.decision_logits, -1).to(torch::kFloat64);
    const auto q = torch::sigmoid(output.answerability_logit).to(torch::kFloat64);
    for (int64_t i = 0; i < probabilities.size(0); ++i) {
      const auto& sample = samples[begin + i];
      std::vector<double> p;
      for (int64_t k=0; k<probabilities.size(1); ++k) p.push_back(probabilities[i][k].item<double>());
      const auto answerability = q[i].item<double>();
      correct_answerability += (answerability >= .5) == sample.answerable;
      answerable_predictions += answerability >= .5;
      ++answerability_confusion[sample.answerable][answerability >= .5];
      if (sample.answerable) {
        const auto predicted = std::max_element(p.begin(), p.end()) - p.begin();
        ++confusion[sample.target][predicted];
      }
      scored.push_back({p, sample.target, sample.answerable, answerability, false});
    }
  }
  const auto m = compute_metrics(scored, 3);
  Json matrix = Json::array(), support = Json::array(), recall = Json::array();
  for (size_t k=0; k<3; ++k) {
    const auto n = confusion[k][0]+confusion[k][1]+confusion[k][2];
    matrix.push_back({confusion[k][0],confusion[k][1],confusion[k][2]});
    support.push_back(n);
    recall.push_back(n ? Json(double(confusion[k][k])/n) : Json(nullptr));
  }
  return {{"total",m.total},{"answerable_count",m.answerable_count},
    {"accuracy",m.accuracy},{"macro_f1",m.macro_f1},{"nll",m.nll},
    {"brier",m.brier},{"ece",m.ece},{"answerability_bce",m.answerability_bce},
    {"answerability_brier",m.answerability_brier},
    {"answerability_accuracy_at_0_5",double(correct_answerability)/samples.size()},
    {"predicted_answerable_at_0_5",answerable_predictions},
    {"answerability_label_order",{"unanswerable","answerable"}},
    {"answerability_confusion_matrix",{{answerability_confusion[0][0],answerability_confusion[0][1]},
      {answerability_confusion[1][0],answerability_confusion[1][1]}}},
    {"confusion_matrix",matrix},{"label_order",{"retry","fallback","stop"}},
    {"label_support",support},{"label_recall",recall}};
}

int main(int argc, char** argv) {
 try {
  if (argc != 5 && argc != 6)
    throw std::invalid_argument("usage: arbitrium_experiment DATASET TOKENIZER OUTPUT SEED [RESEARCH_CONFIG]");
  const fs::path dataset=argv[1], tokenizer_dir=argv[2], out=argv[3];
  if (fs::exists(out)) throw std::invalid_argument("output already exists");
  const auto manifest_bytes = read_file((dataset/"manifest.json").string(), 1024*1024);
  const auto manifest = parse_json(manifest_bytes, 1024*1024);
  if (manifest.at("schema_version") != "arbitrium.dataset.v1" || !manifest.at("fixture_only").get<bool>())
    throw std::invalid_argument("research experiment requires a fixture");
  const auto task_bytes=read_file((dataset/"task.json").string(),65536);
  if (sha256(task_bytes)!=manifest.at("task_sha256").get<std::string>()) throw std::invalid_argument("task hash mismatch");
  const auto task=TaskSpec::from_json(parse_json(task_bytes));
  if (task.id!="runtime.recovery.v1" || task.labels.size()!=3) throw std::invalid_argument("unsupported experiment task");
  Tokenizer tokenizer((tokenizer_dir/"tokenizer.model").string(), (tokenizer_dir/"tokenizer-config.json").string());
  const auto load = [&](const std::string& split) {
    return load_split((dataset/(split+".jsonl")).string(), split,
      manifest.at("files").at(split+".jsonl").at("sha256"), task, tokenizer);
  };
  const auto train=load("train");
  TrainingConfig config; config.task_id=task.id;
  bool diagnostic=false;
  Json research_config=nullptr;
  if (argc==6) {
    research_config=parse_json(read_file(argv[5],65536));
    exact_keys(research_config,{"schema_version","selection_scope","epochs","learning_rate",
      "batch_size","patience","weight_decay"});
    if (research_config.at("schema_version")!="arbitrium.research-training.v1" ||
        (research_config.at("selection_scope")!="train_diagnostic" && research_config.at("selection_scope")!="dev"))
      throw std::invalid_argument("unsupported research configuration");
    for (const auto* k:{"epochs","batch_size","patience"})
      if (!research_config.at(k).is_number_integer()) throw std::invalid_argument("integer config value required");
    for (const auto* k:{"learning_rate","weight_decay"})
      if (!research_config.at(k).is_number() || !std::isfinite(research_config.at(k).get<double>()))
        throw std::invalid_argument("finite numeric config value required");
    diagnostic=research_config.at("selection_scope")=="train_diagnostic";
    config.epochs=research_config.at("epochs"); config.learning_rate=research_config.at("learning_rate");
    config.batch_size=research_config.at("batch_size"); config.patience=research_config.at("patience");
    config.weight_decay=research_config.at("weight_decay");
    if (config.epochs>200 || config.patience>200 || config.learning_rate>.02 || config.weight_decay>1)
      throw std::invalid_argument("research budget exceeded");
  }
  // Train-only memorization is a diagnostic and never called dev generalization.
  const auto dev=diagnostic ? train : load("dev");
  size_t parsed=0; config.seed=std::stoll(argv[4], &parsed);
  if (parsed!=std::string(argv[4]).size()) throw std::invalid_argument("invalid seed");
  config.validate();
  ModelConfig mc; mc.vocab_size=tokenizer.vocab_size(); mc.controls=tokenizer.control_ids();
  auto model=make_training_model(mc, config);
  const auto initial=score(model,dev);
  const auto initial_train=score(model,train);
  fs::create_directories(out);
  const auto result=train_pooled(model,train,dev,config,
    {(out/"checkpoints").string(), sha256(manifest_bytes)+":"+sha256(research_config.dump())+":"+std::to_string(config.seed), false, 0});
  // Report the selected epoch; train-only selection is explicitly diagnostic.
  torch::load(model, out/"checkpoints"/("epoch-"+std::to_string(result.best_epoch+1))/"model.pt");
  Json history=Json::array();
  for (const auto& h:result.history) {
    Json row={{"epoch",h.epoch},{"train_loss",h.train_loss},
      {"selection_objective",h.dev_objective},{"improved",h.improved}};
    if (!diagnostic) row["dev_objective"]=h.dev_objective;
    history.push_back(row);
  }
  size_t support[3]={0,0,0}, dev_support[3]={0,0,0};
  for (const auto& s:train) if(s.answerable) ++support[s.target];
  for (const auto& s:dev) if(s.answerable) ++dev_support[s.target];
  const auto majority=std::max_element(support,support+3)-support;
  size_t parameters=0; for(const auto& p:model->parameters()) parameters+=p.numel();
  const auto majority_accuracy=double(dev_support[majority])/std::max<size_t>(1,dev_support[0]+dev_support[1]+dev_support[2]);
  Json report={{"format",argc==6?"experiment-002-current-v1":"experiment-001-dev-v1"},{"research_only",true},
    {"seed",config.seed},{"architecture",config.architecture_id},{"dataset_digest",sha256(manifest_bytes)},
    {"tokenizer_sha256",sha256(read_file((tokenizer_dir/"tokenizer.model").string(),16*1024*1024))},
    {"toolchain",{{"libtorch",TORCH_VERSION},{"compiler",__VERSION__},{"device","cpu"},{"threads",1}}},
    {"config",{{"epochs",config.epochs},{"batch_size",config.batch_size},{"learning_rate",config.learning_rate},
      {"weight_decay",config.weight_decay},{"patience",config.patience},{"hidden_size",mc.hidden_size},{"dropout",mc.dropout}}},
    {"train_count",train.size()},{"dev_count",diagnostic?0:dev.size()},{"parameter_count",parameters},
    {"best_epoch",result.best_epoch},{"early_stopped",result.early_stopped},
    {"initial_train",initial_train},{"selected_train",score(model,train)},
    {"initial_dev",diagnostic?Json(nullptr):initial},
    {"selected_dev",diagnostic?Json(nullptr):score(model,dev)},{"history",history},
    {"selection_scope",diagnostic?"train_diagnostic":"dev"},
    {"research_config",research_config},
    {"uniform_random_accuracy",1.0/3},
    {"train_majority_selection_accuracy",majority_accuracy},
    {"train_majority_dev_accuracy",diagnostic?Json(nullptr):Json(majority_accuracy)},
    {"calibration","not fitted; reported probabilities are raw"},{"test","not read"}};
  std::ofstream file(out/"report.json"); file << report.dump(2) << '\n';
  if (!file) throw std::runtime_error("report write failed");
  std::cout << "seed " << config.seed << (diagnostic?" train diagnostic accuracy ":" dev accuracy ") << report[diagnostic?"selected_train":"selected_dev"]["accuracy"] << '\n';
  return 0;
 } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
