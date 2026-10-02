#pragma once

#include "arbitrium/contracts.h"

namespace arbitrium {

// In-memory projection of an already verified calibration. This is NOT a
// persisted schema or a substitute for bundle/hash/provenance validation.
struct CalibratedGates {
  std::string artifact_id, calibration_id;
  double decision_temperature{1}, answerability_temperature{1};
  bool accept_none{true};
  std::optional<double> tau_p, tau_q;
};

// Internal postprocessing only: callers must first validate the bundle and
// request. The serving engine/CLI stays unavailable until bundle loading exists.
DecisionResult postprocess_logits(const TaskSpec&, const DecisionRequest&,
                                  const std::vector<double>& decision_logits,
                                  double answerability_logit, const CalibratedGates&);

}  // namespace arbitrium
