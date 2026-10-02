#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arbitrium {

struct ScoredExample {
  std::vector<double> probabilities;
  int64_t target{-1};
  bool answerable{false};
  double answerability{0};
  bool accepted{false};
};

struct Metrics {
  size_t total{0}, answerable_count{0}, accepted_count{0}, accepted_errors{0};
  double accuracy{0}, macro_f1{0}, nll{0}, brier{0}, ece{0};
  double answerability_bce{0}, answerability_brier{0};
  double coverage{0}, selective_risk{0}, false_acceptance{0};
};

Metrics compute_metrics(const std::vector<ScoredExample>&, size_t classes);
double ranked_probability_score(const std::vector<ScoredExample>&, size_t classes);
double wilson_upper(size_t errors, size_t count);

}  // namespace arbitrium
