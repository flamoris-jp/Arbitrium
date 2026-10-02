#include "arbitrium/composition.h"
#include "trusted_legacy.h"
#include <regex>
#include <set>
#include <algorithm>
using namespace maidionis;
namespace arbitrium {
Json trusted(){static const auto value=parse_json(trusted_legacy,4*1024*1024);return value;}
namespace {
void require(bool ok,const char* msg){if(!ok)throw std::invalid_argument(msg);}
std::string legacy_split(const std::string& family){auto h=sha256("arbitrium-split-v1\n42\n"+family);auto n=std::stoull(h.substr(0,16),nullptr,16)%10000;return n<6000?"train":n<7500?"dev":n<8500?"calibration_fit":n<9000?"calibration_select":"test";}
Json component(const std::string& name,const Json& value){return {{"id","arbitrium.decision."+name},{"version","1"},{"config_digest",sha256(canonical(value))}};}
std::vector<int64_t> tokenize(const std::string& text,const std::map<std::string,int64_t>& vocabulary){
  static const std::regex re("[A-Za-z0-9]+|[^\\s]");std::vector<int64_t> ids;
  for(auto it=std::sregex_iterator(text.begin(),text.end(),re);it!=std::sregex_iterator();++it){auto word=it->str();auto p=vocabulary.find(word);ids.push_back(p==vocabulary.end()?1:p->second);}return ids;
}
}
NumericalComposition composition(const std::filesystem::path& root){
  auto read=[&](const std::string& p){return parse_json(read_file(root/p,4*1024*1024),4*1024*1024);};
  auto codec=read("input_codec.config.json");auto name=codec.at("legacy_dataset").get<std::string>();auto sources=trusted();require(sources.contains(name),"unregistered legacy source");auto source=sources.at(name);
  require(codec==source["configs"]["input_codec"],"compiled codec pin");auto descriptor=read("descriptor.json");require(descriptor==source["descriptor"],"compiled descriptor pin");
  require(read("semantic.json")==source["task"],"TaskSpec binding");auto schemas=parse_json(compiled_schemas,65536);RegistryBuilder b(ARBITRIUM_BUILD_DIGEST);
  for(const auto& k:{"input","target","output","raw"}){
    auto s=read(std::string(k)+".schema.json");require(s==schemas[k],"compiled schema pin");
    auto ref=Json{{"id","arbitrium.decision."+std::string(k)},{"version","1"},{"sha256",sha256(canonical(s))}};b.add_schema(ref,s);
  }
  for(const auto& k:{"input_codec","output_codec","architecture","objective","head","numerical_compatibility"}){
    auto config=read(std::string(k)+".config.json");require(config==source["configs"][k],"compiled operation binding");auto ref=component(k,config);b.add_operation(ref,config,[config](const Json& v){require(v==config,"operation configuration");});
  }
  NumericalComposition c;c.registry=b.freeze(descriptor);c.model_config.kind="pooled";c.model_config.hidden_width=256;c.model_config.output_width=4;c.model_config.vocabulary=9+codec["vocabulary"].size();c.model_config.controls={2,3,4,5,6,7,8};
  std::map<std::string,int64_t> vocab;int64_t vi=9;for(const auto& w:codec["vocabulary"])require(vocab.emplace(w.get<std::string>(),vi++).second,"duplicate vocabulary");
  c.encode=[r=c.registry,vocab,source](const std::vector<Json>& rows){
    require(!rows.empty()&&rows.size()<=64,"Decision batch bound");std::vector<std::vector<int64_t>> tokens,segments;std::vector<int64_t> labels;std::vector<float> answers;size_t width=0;bool samples=rows[0].contains("input");
    const std::vector<std::string> names={"retry","fallback","stop"};
    for(const auto& row:rows){require(row.contains("input")==samples,"mixed batch");Json x=samples?row["input"]:row;r.payload("input_schema",x);if(samples)r.sample(row);
      require(x["policy"]==source["task"]["canonical_policy"]&&x["question"]==source["task"]["question"],"canonical policy/question");std::set<std::string> choices;for(const auto& a:x["choices"])choices.insert(a.get<std::string>());require(choices==std::set<std::string>(names.begin(),names.end()),"choice set");
      std::vector<int64_t> ids={4,5},seg={0,0};
      auto field=[&](const std::string& text,int64_t s){auto v=tokenize(text,vocab);ids.insert(ids.end(),v.begin(),v.end());seg.insert(seg.end(),v.size(),s);};
      field("runtime.recovery.v1",0);for(const auto& [key,control,s]:std::vector<std::tuple<std::string,int64_t,int64_t>>{{"policy",6,1},{"state",7,2},{"question",8,3}}){ids.push_back(control);seg.push_back(s);field(x[key],s);}ids.push_back(3);seg.push_back(3);
      require(ids.size()<=256,"assembled token bound");width=std::max(width,ids.size());tokens.push_back(ids);segments.push_back(seg);
      if(samples){bool answerable=row["target"]["answerable"];auto target=row["target"]["label"];require(answerable?!target.is_null():target.is_null(),"answerability target invariant");int64_t label=0;if(answerable){auto p=std::find(names.begin(),names.end(),target.get<std::string>());require(p!=names.end(),"target label");label=std::distance(names.begin(),p);}labels.push_back(label);answers.push_back(answerable?1.f:0.f);}
    }
    Batch out;out.inputs=torch::zeros({int64_t(rows.size()),int64_t(width)},torch::kInt64);out.segments=torch::zeros_like(out.inputs);
    for(size_t i=0;i<tokens.size();++i)for(size_t j=0;j<tokens[i].size();++j){out.inputs[i][j]=tokens[i][j];out.segments[i][j]=segments[i][j];}
    out.mask=out.inputs.ne(0);if(samples){out.targets=torch::from_blob(labels.data(),{int64_t(labels.size())},torch::kInt64).clone();out.target_mask=torch::from_blob(answers.data(),{int64_t(answers.size()),1},torch::kFloat32).clone();}return out;
  };
  c.objective=[](const torch::Tensor& logits,const Batch& batch){return categorical_loss(logits.slice(1,0,3),batch.targets,batch.target_mask.flatten().to(torch::kBool))+bernoulli_loss(logits.slice(1,3,4),batch.target_mask);};
  c.decode=[](const torch::Tensor& logits){require(logits.numel()==4&&torch::isfinite(logits).all().item<bool>()&&logits.abs().le(1e6).all().item<bool>(),"bounded finite Decision output");auto x=logits.flatten().to(torch::kFloat64);auto p=torch::softmax(x.slice(0,0,3),0);std::vector<std::string> names={"retry","fallback","stop"};auto i=p.argmax().item<int64_t>();return Json{{"diagnostic_label",names[i]},{"advisory_label",nullptr},{"probabilities",Json::array({Json{{"label","retry"},{"probability",p[0].item<double>()}},Json{{"label","fallback"},{"probability",p[1].item<double>()}},Json{{"label","stop"},{"probability",p[2].item<double>()}}})},{"answerability",torch::sigmoid(x[3]).item<double>()},{"reason_code","release_gate_not_met"},{"calibration_status","uncalibrated"}};};
  auto manifest_digest=source["manifest"].get<std::string>();auto group=component("legacy-group",Json{{"algorithm","manifest-family-ancestry.v1"}});
  auto hook=component("legacy-hooks",Json{{"build_digest",ARBITRIUM_BUILD_DIGEST},{"legacy_manifest",manifest_digest}});auto verify=component("legacy-oracle",Json{{"oracle","finite-recovery.v1"},{"legacy_manifest",manifest_digest}});
  auto dedup=component("dedup",Json{{"projection","state-casefold-space.v1"}});
  c.split_algorithm="arbitrium.legacy-partition.v1";
  c.assign_family=[manifest_digest,group,hook,source,name](const Json& f,const Json& m,const Json& cfg){
    require(cfg==Json{{"algorithm","arbitrium.legacy-partition.v1"},{"seed",42},{"grouping",group},{"hook",hook}}&&m["purpose"]=="research_fixture","legacy partition binding");
    require(f["roots"].size()==1,"legacy family roots");auto root=f["roots"][0]["content"];require(root.size()==2&&root["legacy_manifest"]==manifest_digest&&root.contains("legacy_family"),"legacy ancestry");
    require(m["dataset_id"]==name+".maidionis.v1","legacy derivative dataset identity");
    auto family=root["legacy_family"].get<std::string>();std::map<std::string,size_t> counts;std::map<std::string,std::set<std::string>> known;
    Json expected=Json::array();for(const auto& row:source["rows"]){auto key=row["family"].get<std::string>(),split=legacy_split(key);++counts[split];known[split].insert(key);if(key==family)expected.push_back(row["id"]);}
    std::sort(expected.begin(),expected.end());require(!expected.empty()&&f["members"]==expected&&f["aliases"]==Json::array({family})&&f["audit_members"].empty(),"complete original family membership");
    for(const auto& split:{"train","dev","calibration_fit","calibration_select","test"})require(m["counts"][split]["records"]==counts[split]&&m["counts"][split]["families"]==known[split].size(),"complete original split coverage");
require(std::find(f["aliases"].begin(),f["aliases"].end(),family)!=f["aliases"].end(),"legacy alias binding");return legacy_split(family);
  };
  c.verify_dataset_row=[source,manifest_digest,verify,dedup,group,hook](const Json& row,const Json& f,const Json& m,const Json& cfg){
    require(m["verification_profile"]==verify&&m["dedup_profile"]==dedup&&row["verification_profile"]==verify&&row["supersedes"].is_null()&&cfg["grouping"]==group&&cfg["hook"]==hook,"legacy verification profiles");
    const Json* old=nullptr;for(const auto& r:source["rows"])if(r["id"]==row["sample_id"]){old=&r;break;}
    require(old&&(*old)["input"]==row["input"]&&(*old)["target"]==row["target"]&&(*old)["family"]==row["family_id"]&&(*old)["provenance_id"]==row["provenance_id"],"inventoried Decision row binding");
    auto root=Json{{"legacy_manifest",manifest_digest},{"legacy_family",(*old)["family"]}};require(f["roots"]==Json::array({Json{{"digest",sha256(canonical(root))},{"content",root}}}),"legacy root projection");
  };
  c.validate();return c;
}
}
