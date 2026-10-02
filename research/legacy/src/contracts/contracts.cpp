#include "arbitrium/contracts.h"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace arbitrium {
namespace {
void require(bool yes, const char* message) { if (!yes) throw ContractError(message); }
void finite(const Json& j) {
  if(j.is_number_float()) require(std::isfinite(j.get<double>()), "nonfinite JSON number");
  if(j.is_structured()) for(const auto& v:j) finite(v);
}
bool whitespace(const std::string& s) {
  return std::all_of(s.begin(),s.end(),[](unsigned char c){return c==' '||c=='\t'||c=='\r'||c=='\n';});
}
}
Json parse_json(const std::string& bytes, size_t limit) {
  if(bytes.size()>limit) throw ContractError("JSON byte limit exceeded","input_too_long");
  std::vector<std::set<std::string>> stack;
  try {
    auto callback=[&](int, Json::parse_event_t event, Json& parsed) {
      if(event==Json::parse_event_t::object_start) stack.emplace_back();
      if(event==Json::parse_event_t::key && !stack.back().insert(parsed.get<std::string>()).second)
        throw ContractError("duplicate JSON key");
      if(event==Json::parse_event_t::object_end) stack.pop_back();
      return true;
    };
    auto result=Json::parse(bytes,callback); finite(result); return result;
  } catch(const Json::exception&) { throw ContractError("invalid UTF-8 JSON"); }
}
void exact_keys(const Json& j,const std::vector<std::string>& keys) {
  require(j.is_object() && j.size()==keys.size(),"object field set mismatch");
  for(const auto& k:keys) require(j.contains(k),"missing field");
}
void valid_id(const Json& j) {
  static const std::regex pattern("^[A-Za-z0-9._:-]{1,128}$");
  require(j.is_string() && std::regex_match(j.get<std::string>(),pattern),"invalid opaque ID");
}
std::string normalized_text(const Json& j,size_t limit) {
  require(j.is_string(),"text must be a string");
  const auto s=j.get<std::string>();
  // dump validates UTF-8 for typed callers too (not only the wire parser).
  try { (void)j.dump(); } catch(const Json::exception&) {throw ContractError("invalid UTF-8");}
  if(s.size()>limit) throw ContractError("field byte limit exceeded","input_too_long");
  require(!s.empty()&&!whitespace(s),"empty text");
  std::string out;
  for(size_t i=0;i<s.size();++i) {
    unsigned char c=s[i];
    require(!(c<32&&c!='\t'&&c!='\n'&&c!='\r')&&c!=127,"prohibited control character");
    if(c=='\r') {out+='\n'; if(i+1<s.size()&&s[i+1]=='\n') ++i;}
    else out+=s[i];
  }
  return out;
}
TaskSpec TaskSpec::from_json(const Json& j) {
  exact_keys(j,{"schema_version","task_id","kind","question","description","policy_scope","canonical_policy","labels","positive_label","max_sequence_length","assembly_version"});
  require(j["schema_version"]=="arbitrium.task.v1","unsupported task schema");
  valid_id(j["task_id"]);require(j["kind"]=="choice"||j["kind"]=="binary"||j["kind"]=="ordinal","invalid kind");
  require(j["max_sequence_length"].is_number_integer()&&j["max_sequence_length"]==256&&j["assembly_version"]=="fields.v1","unsupported assembly");
  normalized_text(j["description"],4*1024*1024);normalized_text(j["policy_scope"],4*1024*1024);
  TaskSpec t{j["task_id"],j["kind"],normalized_text(j["question"],2048),normalized_text(j["canonical_policy"],8192),{}};
  require(j["labels"].is_array()&&j["labels"].size()>=2&&j["labels"].size()<=16,"invalid label count");
  std::set<std::string> seen;
  for(const auto& l:j["labels"]) {
    exact_keys(l,{"id","description","anchor"});valid_id(l["id"]);
    Label label{l["id"],normalized_text(l["description"],4*1024*1024),0};
    require(seen.insert(label.id).second,"duplicate label");
    if(t.kind=="ordinal") {
      require(l["anchor"].is_number(),"ordinal anchor required");label.anchor=l["anchor"].get<double>();
      require(std::isfinite(label.anchor)&&(t.labels.empty()||label.anchor>t.labels.back().anchor),"invalid ordinal anchors");
    } else require(l["anchor"].is_null(),"unexpected anchor");
    t.labels.push_back(label);
  }
  if(t.kind=="binary") require(t.labels.size()==2&&t.labels[0].id=="false"&&t.labels[1].id=="true"&&j["positive_label"]=="true","binary labels mismatch");
  else require(j["positive_label"].is_null(),"unexpected positive label");
  return t;
}
DecisionRequest DecisionRequest::from_json(const Json& j,const TaskSpec& t) {
  exact_keys(j,{"schema_version","request_id","task_id","kind","language","state","policy","question","choices"});
  if(j["schema_version"]!="arbitrium.request.v1")throw ContractError("unsupported request schema","unsupported_schema");
  valid_id(j["request_id"]);valid_id(j["task_id"]);
  if(j["language"]!="en")throw ContractError("English declaration required","unsupported_language");
  if(j["task_id"]!=t.id||j["kind"]!=t.kind)throw ContractError("task mismatch","unsupported_task");
  DecisionRequest r{j["request_id"],t.id,t.kind,normalized_text(j["state"],16384),normalized_text(j["policy"],8192),normalized_text(j["question"],2048),{}};
  if(r.question!=t.question)throw ContractError("canonical question mismatch","question_mismatch");
  require(r.policy==t.policy,"canonical policy mismatch");
  std::set<std::string> expected,actual;for(const auto& l:t.labels)expected.insert(l.id);
  if(!j["choices"].is_array()||j["choices"].size()!=t.labels.size())throw ContractError("invalid choices","invalid_choices");
  for(const auto& c:j["choices"]) {valid_id(c);r.choices.push_back(c);actual.insert(c);}
  if(actual!=expected)throw ContractError("invalid choices","invalid_choices");
  return r;
}

DecisionResult DecisionResult::from_json(const Json& j,const TaskSpec& task,
                                         const DecisionRequest& request) {
  finite(j); // Typed callers can bypass parse_json; reject NaN/infinity here too.
  exact_keys(j,{"schema_version","request_id","task_id","artifact_id","kind","status",
                "verdict","probabilities","score","confidence","answerability",
                "reason_code","calibration_id","error"});
  require(j["schema_version"]=="arbitrium.result.v1","unsupported result schema");
  auto optional_id=[&](const char* key)->std::optional<std::string>{
    if(j[key].is_null())return std::nullopt;valid_id(j[key]);return j[key].get<std::string>();};
  DecisionResult result;
  result.request_id=optional_id("request_id");result.task_id=optional_id("task_id");
  result.artifact_id=optional_id("artifact_id");result.verdict=optional_id("verdict");
  result.calibration_id=optional_id("calibration_id");
  if(!j["kind"].is_null()) {
    require(j["kind"]=="choice"||j["kind"]=="binary"||j["kind"]=="ordinal","invalid result kind");
    result.kind=j["kind"].get<std::string>();
  }
  require(j["status"]=="ok"||j["status"]=="abstain"||j["status"]=="error","invalid result status");
  result.status=j["status"].get<std::string>();
  require(j["probabilities"].is_array()&&j["probabilities"].size()<=16,"invalid probabilities");
  for(const auto& value:j["probabilities"]) {
    exact_keys(value,{"label","probability"});valid_id(value["label"]);
    require(value["probability"].is_number(),"invalid probability");
    const auto probability=value["probability"].get<double>();
    require(probability>=0&&probability<=1,"invalid probability");
    result.probabilities.push_back({value["label"],probability});
  }
  auto optional_number=[&](const char* key)->std::optional<double>{
    if(j[key].is_null())return std::nullopt;require(j[key].is_number(),"invalid result number");
    return j[key].get<double>();};
  result.score=optional_number("score");result.confidence=optional_number("confidence");
  result.answerability=optional_number("answerability");
  for(const auto value:{result.confidence,result.answerability})
    require(!value||(value.value()>=0&&value.value()<=1),"result probability out of range");
  if(!j["reason_code"].is_null()) {
    require(j["reason_code"]=="low_answerability"||j["reason_code"]=="low_confidence"||
            j["reason_code"]=="release_gate_not_met","invalid reason code");
    result.reason_code=j["reason_code"].get<std::string>();
  }
  if(!j["error"].is_null()) {
    const auto& error=j["error"];exact_keys(error,{"code","retryable","message"});
    static const std::set<std::string> codes={"invalid_request","unsupported_schema","unsupported_task",
      "unsupported_language","invalid_choices","question_mismatch","input_too_long","artifact_invalid",
      "artifact_incompatible","calibration_missing","resource_exhausted","nonfinite_output","cancelled",
      "deadline_exceeded","io_error","internal_error"};
    require(error["code"].is_string()&&codes.contains(error["code"].get<std::string>()),"invalid error code");
    require(error["retryable"].is_boolean(),"invalid retryable flag");
    require(error["message"].is_string(),"invalid error message");
    const auto message=error["message"].get<std::string>();
    try {(void)error["message"].dump();}
    catch(const Json::exception&) {throw ContractError("invalid UTF-8 error message");}
    // JSON Schema maxLength counts Unicode code points, not UTF-8 bytes.
    const auto length=std::count_if(message.begin(),message.end(),[](unsigned char c){
      return (c&0xc0)!=0x80;
    });
    require(length>=1&&length<=1024,"invalid error message length");
    result.error=ResultError{error["code"].get<std::string>(),error["retryable"].get<bool>(),message};
  }
  if(result.status=="error") {
    require(!result.verdict&&result.probabilities.empty()&&!result.score&&!result.confidence&&
            !result.answerability&&!result.reason_code&&result.error.has_value(),"invalid error result");
    return result;
  }
  require(result.request_id==request.id&&result.task_id==task.id&&result.kind==task.kind&&
          result.artifact_id&&result.calibration_id&&!result.error,"result identity mismatch");
  require(request.task_id==task.id&&request.kind==task.kind&&request.question==task.question&&
          request.policy==task.policy,"request/task mismatch");
  std::set<std::string> expected_labels,request_labels(request.choices.begin(),request.choices.end());
  for(const auto& label:task.labels)expected_labels.insert(label.id);
  require(expected_labels.size()>=2&&request_labels==expected_labels&&
          request.choices.size()==task.labels.size(),"request label set mismatch");
  require(result.probabilities.size()>=2&&result.probabilities.size()==request.choices.size(),
          "result probability count mismatch");
  double total=0;std::map<std::string,double> by_id;
  for(size_t i=0;i<result.probabilities.size();++i) {
    require(result.probabilities[i].label==request.choices[i],"result label order mismatch");
    require(by_id.emplace(result.probabilities[i].label,result.probabilities[i].probability).second,
            "duplicate result label");total+=result.probabilities[i].probability;
  }
  require(std::abs(total-1)<=1e-5,"probabilities must sum to one");
  require(result.answerability.has_value(),"missing answerability");
  if(result.status=="abstain") {
    require(!result.verdict&&!result.score&&!result.confidence&&result.reason_code,
            "invalid abstain result");return result;
  }
  require(result.verdict&&result.confidence&&!result.reason_code,"invalid ok result");
  auto winner=task.labels.front().id;
  for(const auto& label:task.labels)if(by_id.at(label.id)>by_id.at(winner))winner=label.id;
  require(result.verdict==winner&&std::abs(*result.confidence-by_id.at(winner))<=1e-5,
          "verdict/confidence mismatch");
  if(task.kind=="ordinal") {
    require(result.score.has_value(),"ordinal score missing");double expected=0;
    for(const auto& label:task.labels)expected+=by_id.at(label.id)*label.anchor;
    require(std::abs(*result.score-expected)<=1e-5,"ordinal expectation mismatch");
  } else require(!result.score,"unexpected score");
  return result;
}

Sample Sample::from_json(const Json& j,const TaskSpec& t) {
  exact_keys(j,{"schema_version","sample_id","family_id","task_id","split","input","target","concepts","difficulty","provenance_id","review_status","verification_kind","supersedes"});
  require(j["schema_version"]=="arbitrium.sample.v1"&&j["review_status"]=="verified","unverified sample");
  for(auto k:{"sample_id","family_id","task_id","provenance_id"})valid_id(j[k]);
  require(j["task_id"]==t.id,"sample task mismatch");
  require(j["split"]=="train"||j["split"]=="dev"||j["split"]=="calibration_fit"||j["split"]=="calibration_select"||j["split"]=="test","invalid split");
  require(j["verification_kind"]=="controlled_oracle"||j["verification_kind"]=="audited_paraphrase"||j["verification_kind"]=="human"||j["verification_kind"]=="execution_adjudicated","invalid verification");
  require(j["difficulty"].is_number_integer()&&j["difficulty"]>=1&&j["difficulty"]<=5,"invalid difficulty");
  require(j["concepts"].is_array()&&!j["concepts"].empty(),"invalid concepts");
  std::set<std::string> concepts;for(const auto& c:j["concepts"]) {valid_id(c);require(concepts.insert(c).second,"duplicate concept");}
  if(!j["supersedes"].is_null()) {valid_id(j["supersedes"]);require(j["supersedes"]!=j["sample_id"],"self supersession");}
  auto input=DecisionRequest::from_json(j["input"],t);
  require(input.id==j["sample_id"].get<std::string>(),"sample ID mismatch");
  exact_keys(j["target"],{"answerable","label"});require(j["target"]["answerable"].is_boolean(),"answerability must be boolean");
  bool answerable=j["target"]["answerable"];int64_t target=-1;
  if(answerable) {for(size_t i=0;i<t.labels.size();++i)if(j["target"]["label"]==t.labels[i].id)target=i;require(target>=0,"unknown target");}
  else require(j["target"]["label"].is_null(),"unanswerable label must be null");
  return {j["sample_id"],j["family_id"],j["split"],j["provenance_id"],std::move(input),answerable,target};
}
std::string sha256(const std::string& bytes) {
  unsigned char hash[EVP_MAX_MD_SIZE];unsigned int size=0;
  if(EVP_Digest(bytes.data(),bytes.size(),hash,&size,EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 failure");
  std::ostringstream out;for(unsigned int i=0;i<size;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned int>(hash[i]);return out.str();
}
std::string read_file(const std::string& path,size_t limit) {
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open file");
  std::string result;char block[8192];
  while(f) {f.read(block,sizeof block);auto n=f.gcount();if(result.size()+n>limit)throw std::runtime_error("file limit exceeded");result.append(block,n);}
  if(!f.eof())throw std::runtime_error("file read failure");return result;
}
}
