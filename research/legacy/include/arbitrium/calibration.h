#pragma once

#include "arbitrium/metrics.h"

#include <functional>

namespace arbitrium {

struct TemperatureFit {
  double temperature{1}, raw_nll{0}, fitted_nll{0};
  size_t support{0};
  bool converged{false}, boundary_hit{false};
};

TemperatureFit fit_decision_temperature(const std::vector<std::vector<double>>& logits,
                                        const std::vector<int64_t>& targets);
TemperatureFit fit_answerability_temperature(const std::vector<double>& logits,
                                             const std::vector<bool>& targets);
std::vector<double> calibrated_softmax(const std::vector<double>& logits, double temperature);
double calibrated_sigmoid(double logit, double temperature);

struct GateSelection {
  bool accept_none{true};
  double tau_q{0}, tau_p{0}, coverage{0}, upper_risk{0};
  size_t accepted{0}, errors{0};
};

GateSelection select_gate(const std::vector<ScoredExample>& calibration_select);

}  // namespace arbitrium
