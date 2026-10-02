#include "arbitrium/calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace arbitrium {
namespace {
TemperatureFit fit(size_t support, const std::function<double(double)>& objective) {
  if (!support) throw std::invalid_argument("empty temperature support");
  constexpr double ratio = 0.6180339887498948482;
  double left = -4, right = 4;
  double x1 = right - ratio * (right - left), x2 = left + ratio * (right - left);
  double y1 = objective(x1), y2 = objective(x2);
  int iterations = 0;
  for (; iterations < 200 && right - left >= 1e-6; ++iterations) {
    if (y1 < y2) { right = x2; x2 = x1; y2 = y1; x1 = right - ratio * (right-left); y1 = objective(x1); }
    else { left = x1; x1 = x2; y1 = y2; x2 = left + ratio * (right-left); y2 = objective(x2); }
  }
  const auto log_temperature = (left + right) / 2;
  const auto candidate_temperature = std::exp(log_temperature);
  const auto candidate = objective(log_temperature);
  const auto raw = objective(0);
  const bool use_fit = std::isfinite(candidate) && candidate < raw;
  return {use_fit ? candidate_temperature : 1.0, raw, use_fit ? candidate : raw, support,
          right-left < 1e-6, use_fit && (std::abs(log_temperature+4)<1e-4 || std::abs(log_temperature-4)<1e-4)};
}
}

std::vector<double> calibrated_softmax(const std::vector<double>& logits, double temperature) {
  if (logits.size() < 2 || !std::isfinite(temperature) || temperature < std::exp(-4) || temperature > std::exp(4))
    throw std::invalid_argument("invalid calibrated softmax input");
  const auto maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<double> result; result.reserve(logits.size()); double total = 0;
  for (const auto value : logits) {
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite logit");
    result.push_back(std::exp((value-maximum)/temperature)); total += result.back();
  }
  for (auto& value : result) value /= total;
  return result;
}

double calibrated_sigmoid(double logit, double temperature) {
  if (!std::isfinite(logit) || !std::isfinite(temperature) || temperature < std::exp(-4) || temperature > std::exp(4))
    throw std::invalid_argument("invalid calibrated sigmoid input");
  const auto scaled = logit / temperature;
  return scaled >= 0 ? 1/(1+std::exp(-scaled)) : std::exp(scaled)/(1+std::exp(scaled));
}

TemperatureFit fit_decision_temperature(const std::vector<std::vector<double>>& logits,
                                        const std::vector<int64_t>& targets) {
  if (logits.size()!=targets.size()) throw std::invalid_argument("decision fit shape");
  return fit(logits.size(), [&](double log_temperature) {
    const auto temperature=std::exp(log_temperature); double nll=0;
    for(size_t i=0;i<logits.size();++i) {
      if(targets[i]<0||targets[i]>=static_cast<int64_t>(logits[i].size()))throw std::invalid_argument("decision target");
      nll-=std::log(std::max(calibrated_softmax(logits[i],temperature)[targets[i]],1e-12));
    }
    return nll/logits.size();
  });
}

TemperatureFit fit_answerability_temperature(const std::vector<double>& logits,
                                             const std::vector<bool>& targets) {
  if(logits.size()!=targets.size())throw std::invalid_argument("answerability fit shape");
  return fit(logits.size(),[&](double log_temperature) {
    const auto temperature=std::exp(log_temperature);double nll=0;
    for(size_t i=0;i<logits.size();++i){const auto p=calibrated_sigmoid(logits[i],temperature);
      nll-=targets[i]?std::log(std::max(p,1e-12)):std::log(std::max(1-p,1e-12));}
    return nll/logits.size();
  });
}

GateSelection select_gate(const std::vector<ScoredExample>& values) {
  if(values.empty())throw std::invalid_argument("empty gate support");
  const std::vector<double> grid{.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99};
  GateSelection best;
  for(const auto q:grid)for(const auto p:grid) {
    size_t accepted=0,errors=0;
    for(const auto& value:values) {
      const auto confidence=*std::max_element(value.probabilities.begin(),value.probabilities.end());
      if(value.answerability<q||confidence<p)continue;
      ++accepted;
      const auto prediction=std::distance(value.probabilities.begin(),std::max_element(value.probabilities.begin(),value.probabilities.end()));
      errors+=!value.answerable||prediction!=value.target;
    }
    const auto coverage=static_cast<double>(accepted)/values.size();
    if(accepted<200||coverage<.50)continue;
    const auto upper=wilson_upper(errors,accepted);if(upper>.05)continue;
    const bool better=best.accept_none||coverage>best.coverage||
      (coverage==best.coverage&&(upper<best.upper_risk||
       (upper==best.upper_risk&&(q>best.tau_q||(q==best.tau_q&&p>best.tau_p)))));
    if(better)best={false,q,p,coverage,upper,accepted,errors};
  }
  return best;
}

}  // namespace arbitrium
