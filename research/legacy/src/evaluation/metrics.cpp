#include "arbitrium/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace arbitrium {
namespace {
constexpr double kEpsilon = 1e-12;
int64_t modal(const std::vector<double>& probabilities) {
  // std::max_element preserves the canonical-order winner on ties.
  return std::distance(probabilities.begin(),
                       std::max_element(probabilities.begin(), probabilities.end()));
}
void validate(const ScoredExample& value, size_t classes) {
  if (value.probabilities.size() != classes || !std::isfinite(value.answerability) ||
      value.answerability < 0 || value.answerability > 1)
    throw std::invalid_argument("invalid scored example");
  double sum = 0;
  for (const auto probability : value.probabilities) {
    if (!std::isfinite(probability) || probability < 0 || probability > 1)
      throw std::invalid_argument("invalid probability");
    sum += probability;
  }
  if (std::abs(sum - 1) > 1e-5 ||
      (value.answerable && (value.target < 0 || value.target >= static_cast<int64_t>(classes))) ||
      (!value.answerable && value.target != -1))
    throw std::invalid_argument("invalid score target");
}
}  // namespace

Metrics compute_metrics(const std::vector<ScoredExample>& values, size_t classes) {
  if (values.empty() || classes < 2 || classes > 16) throw std::invalid_argument("metric support");
  Metrics result;
  result.total = values.size();
  std::vector<size_t> true_positive(classes), false_positive(classes), false_negative(classes);
  std::array<size_t, 15> bin_count{}, bin_correct{};
  std::array<double, 15> bin_confidence{};
  size_t raw_correct = 0, unanswerable = 0, accepted_unanswerable = 0;
  for (const auto& value : values) {
    validate(value, classes);
    const auto prediction = modal(value.probabilities);
    const auto confidence = value.probabilities[prediction];
    const auto actual = value.answerable ? 1.0 : 0.0;
    result.answerability_bce += -(actual * std::log(std::max(value.answerability, kEpsilon)) +
                                  (1 - actual) * std::log(std::max(1 - value.answerability, kEpsilon)));
    result.answerability_brier += std::pow(value.answerability - actual, 2);
    if (value.answerable) {
      ++result.answerable_count;
      const bool correct = prediction == value.target;
      raw_correct += correct;
      if (correct) ++true_positive[prediction];
      else { ++false_positive[prediction]; ++false_negative[value.target]; }
      result.nll += -std::log(std::max(value.probabilities[value.target], kEpsilon));
      for (size_t index = 0; index < classes; ++index)
        result.brier += std::pow(value.probabilities[index] - (index == static_cast<size_t>(value.target)), 2);
      const auto bin = std::min<size_t>(14, static_cast<size_t>(confidence * 15));
      ++bin_count[bin]; bin_correct[bin] += correct; bin_confidence[bin] += confidence;
    } else {
      ++unanswerable;
    }
    if (value.accepted) {
      ++result.accepted_count;
      const bool error = !value.answerable || prediction != value.target;
      result.accepted_errors += error;
      if (!value.answerable) ++accepted_unanswerable;
    }
  }
  if (!result.answerable_count) throw std::invalid_argument("no answerable metric support");
  result.accuracy = static_cast<double>(raw_correct) / result.answerable_count;
  result.nll /= result.answerable_count;
  result.brier /= result.answerable_count;
  for (size_t index = 0; index < classes; ++index) {
    const auto denominator = 2 * true_positive[index] + false_positive[index] + false_negative[index];
    result.macro_f1 += denominator ? 2.0 * true_positive[index] / denominator : 0;
  }
  result.macro_f1 /= classes;
  for (size_t bin = 0; bin < 15; ++bin) if (bin_count[bin]) {
    const auto accuracy = static_cast<double>(bin_correct[bin]) / bin_count[bin];
    const auto confidence = bin_confidence[bin] / bin_count[bin];
    result.ece += static_cast<double>(bin_count[bin]) / result.answerable_count *
                  std::abs(accuracy - confidence);
  }
  result.answerability_bce /= result.total;
  result.answerability_brier /= result.total;
  result.coverage = static_cast<double>(result.accepted_count) / result.total;
  result.selective_risk = result.accepted_count
                              ? static_cast<double>(result.accepted_errors) / result.accepted_count
                              : std::numeric_limits<double>::quiet_NaN();
  result.false_acceptance = unanswerable
                                ? static_cast<double>(accepted_unanswerable) / unanswerable
                                : std::numeric_limits<double>::quiet_NaN();
  return result;
}

double ranked_probability_score(const std::vector<ScoredExample>& values, size_t classes) {
  if (classes < 2) throw std::invalid_argument("RPS classes");
  double total = 0; size_t support = 0;
  for (const auto& value : values) {
    validate(value, classes);
    if (!value.answerable) continue;
    ++support; double predicted = 0;
    for (size_t index = 0; index + 1 < classes; ++index) {
      predicted += value.probabilities[index];
      const auto observed = index >= static_cast<size_t>(value.target) ? 1.0 : 0.0;
      total += std::pow(predicted - observed, 2);
    }
  }
  if (!support) throw std::invalid_argument("no RPS support");
  return total / (support * (classes - 1));
}

double wilson_upper(size_t errors, size_t count) {
  if (!count || errors > count) throw std::invalid_argument("Wilson support");
  constexpr double z = 1.6448536269514722;
  const auto rate = static_cast<double>(errors) / count;
  const auto z2 = z * z;
  return (rate + z2 / (2 * count) +
          z * std::sqrt(rate * (1 - rate) / count + z2 / (4.0 * count * count))) /
         (1 + z2 / count);
}

}  // namespace arbitrium
