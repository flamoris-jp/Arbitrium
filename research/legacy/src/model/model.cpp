#include "arbitrium/model.h"

#include <cmath>
#include <limits>

namespace arbitrium {
void validate_tensor_input(const torch::Tensor& ids,const torch::Tensor& mask,const torch::Tensor& segments,int64_t vocab) {
  TORCH_CHECK(ids.dim()==2&&ids.size(0)>=1&&ids.size(0)<=64&&ids.size(1)>=1&&ids.size(1)<=256,"invalid batch/sequence dimensions");
  TORCH_CHECK(ids.scalar_type()==torch::kInt64&&segments.scalar_type()==torch::kInt64&&mask.scalar_type()==torch::kBool,"invalid input dtypes");
  TORCH_CHECK(ids.sizes()==mask.sizes()&&ids.sizes()==segments.sizes(),"input shape mismatch");
  TORCH_CHECK(ids.device()==mask.device()&&ids.device()==segments.device(),"input device mismatch");
  TORCH_CHECK(ids.min().item<int64_t>()>=0&&ids.max().item<int64_t>()<vocab,"token out of range");
  TORCH_CHECK(segments.min().item<int64_t>()>=0&&segments.max().item<int64_t>()<4,"segment out of range");
  TORCH_CHECK(mask.select(1,0).all().item<bool>(),"empty mask row");
  TORCH_CHECK(torch::equal(ids.ne(0),mask),"padding mask mismatch");
  TORCH_CHECK(segments.masked_select(~mask).eq(0).all().item<bool>(),"padding segment must be zero");
  if(ids.size(1)>1)TORCH_CHECK((mask.slice(1,1).to(torch::kInt64)-mask.slice(1,0,-1).to(torch::kInt64)).le(0).all().item<bool>(),"mask must be right padded");
}

EncoderBlockImpl::EncoderBlockImpl(const ModelConfig& c)
    : heads_(c.num_heads), head_dimension_(c.hidden_size/c.num_heads) {
  norm1_=register_module("norm1",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.hidden_size}).eps(c.norm_epsilon)));
  norm2_=register_module("norm2",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.hidden_size}).eps(c.norm_epsilon)));
  q_=register_module("q",torch::nn::Linear(c.hidden_size,c.hidden_size));
  k_=register_module("k",torch::nn::Linear(c.hidden_size,c.hidden_size));
  v_=register_module("v",torch::nn::Linear(c.hidden_size,c.hidden_size));
  output_=register_module("output",torch::nn::Linear(c.hidden_size,c.hidden_size));
  ffn1_=register_module("ffn1",torch::nn::Linear(c.hidden_size,c.ffn_size));
  ffn2_=register_module("ffn2",torch::nn::Linear(c.ffn_size,c.hidden_size));
  attention_dropout_=register_module("attention_dropout",torch::nn::Dropout(c.dropout));
  output_dropout_=register_module("output_dropout",torch::nn::Dropout(c.dropout));
  ffn_dropout_=register_module("ffn_dropout",torch::nn::Dropout(c.dropout));
}

torch::Tensor EncoderBlockImpl::forward(const torch::Tensor& input,const torch::Tensor& mask) {
  const auto batch=input.size(0),length=input.size(1),width=input.size(2);
  const auto normalized=norm1_->forward(input);
  const auto split=[&](torch::nn::Linear& layer) {
    return layer->forward(normalized).view({batch,length,heads_,head_dimension_}).transpose(1,2);
  };
  const auto query=split(q_),key=split(k_),value=split(v_);
  auto scores=torch::matmul(query,key.transpose(-2,-1))/std::sqrt(static_cast<double>(head_dimension_));
  scores=scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2),-std::numeric_limits<float>::infinity());
  auto probabilities=attention_dropout_->forward(torch::softmax(scores,-1));
  auto attended=torch::matmul(probabilities,value).transpose(1,2).contiguous().view({batch,length,width});
  auto hidden=(input+output_dropout_->forward(output_->forward(attended)))*mask.unsqueeze(-1);
  auto ffn=ffn2_->forward(torch::gelu(ffn1_->forward(norm2_->forward(hidden)),"none"));
  return (hidden+ffn_dropout_->forward(ffn))*mask.unsqueeze(-1);
}

ChoiceModelImpl::ChoiceModelImpl(const ModelConfig& c):config_(c) {
  const bool pooled=c.architecture_id=="pooled_embedding_v1",encoder=c.architecture_id=="tiny_encoder_v1";
  TORCH_CHECK((pooled||encoder)&&c.vocab_size>=9&&c.vocab_size<=8192&&c.hidden_size==256&&
              c.num_choices>=2&&c.num_choices<=16&&c.dropout==0.1,"invalid model config");
  if(encoder)TORCH_CHECK(c.num_layers==4&&c.num_heads==4&&c.ffn_size==1024&&
                         c.max_sequence_length==256&&c.norm_epsilon==1e-5,"invalid encoder config");
  embedding_=register_module("embedding",torch::nn::Embedding(torch::nn::EmbeddingOptions(c.vocab_size,c.hidden_size).padding_idx(0)));
  if(pooled)projection_=register_module("projection",torch::nn::Linear(c.hidden_size,c.hidden_size));
  else {
    position_embedding_=register_module("position_embedding",torch::nn::Embedding(c.max_sequence_length,c.hidden_size));
    segment_embedding_=register_module("segment_embedding",torch::nn::Embedding(4,c.hidden_size));
    blocks_=register_module("blocks",torch::nn::ModuleList());
    for(int64_t layer=0;layer<c.num_layers;++layer)blocks_->push_back(EncoderBlock(c));
    final_norm_=register_module("final_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.hidden_size}).eps(c.norm_epsilon)));
  }
  decision_=register_module("decision",torch::nn::Linear(c.hidden_size,c.num_choices));
  answerability_=register_module("answerability",torch::nn::Linear(c.hidden_size,1));
  dropout_=register_module("dropout",torch::nn::Dropout(c.dropout));
  torch::NoGradGuard guard;
  for(auto& parameter:named_parameters()) {
    if(parameter.key().ends_with("weight")&&parameter.key().find("norm")!=std::string::npos)
      torch::nn::init::ones_(parameter.value());
    else if(parameter.key().ends_with("weight"))torch::nn::init::normal_(parameter.value(),0,0.02);
    else torch::nn::init::zeros_(parameter.value());
  }
  embedding_->weight[0].zero_();
}
ModelOutput ChoiceModelImpl::forward(const torch::Tensor& ids,const torch::Tensor& mask,const torch::Tensor& segments) {
  validate_tensor_input(ids,mask,segments,config_.vocab_size);
  torch::Tensor h;
  if(config_.architecture_id=="pooled_embedding_v1") {
    auto content=mask.clone();for(auto id:config_.controls)content=content&ids.ne(id);
    auto count=content.sum(1,true);TORCH_CHECK(count.gt(0).all().item<bool>(),"no content tokens");
    h=(embedding_->forward(ids)*content.unsqueeze(-1)).sum(1)/count.to(torch::kFloat32);
    h=dropout_->forward(torch::gelu(projection_->forward(h),"none"));
  } else {
    auto positions=torch::arange(ids.size(1),ids.options()).unsqueeze(0);
    auto hidden=dropout_->forward(embedding_->forward(ids)+position_embedding_->forward(positions)+
                                  segment_embedding_->forward(segments));
    hidden=hidden*mask.unsqueeze(-1);
    for(auto& module:*blocks_)hidden=module->as<EncoderBlock>()->forward(hidden,mask);
    h=final_norm_->forward(hidden).select(1,0);
  }
  return {decision_->forward(h),answerability_->forward(h).squeeze(-1)};
}
torch::Tensor supervised_loss(const ModelOutput& out,const torch::Tensor& targets,const torch::Tensor& answerable) {
  TORCH_CHECK(targets.dim()==1&&answerable.dim()==1&&targets.size(0)==out.decision_logits.size(0)&&targets.sizes()==answerable.sizes(),"target shape mismatch");
  TORCH_CHECK(targets.scalar_type()==torch::kInt64&&answerable.scalar_type()==torch::kBool,"target dtype mismatch");
  auto selected=answerable.nonzero().squeeze(1);
  auto decision_loss=out.decision_logits.sum()*0;
  if(selected.numel())decision_loss=torch::nn::functional::cross_entropy(out.decision_logits.index_select(0,selected),targets.index_select(0,selected));
  auto answer_loss=torch::nn::functional::binary_cross_entropy_with_logits(out.answerability_logit,answerable.to(torch::kFloat32));
  return decision_loss+answer_loss;
}
}
