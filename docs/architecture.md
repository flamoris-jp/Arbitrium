# Decision composition and authority

`python/arbitrium/composition.py` and `src/composition.cpp` independently register
closed schemas and approved operations. Native registration rejects descriptors,
codec/configs or TaskSpecs that differ from compiled approved fixtures. The
compiled approved fixture data is generated only after the source inventory pin
and every original byte/digest has been verified. CMake independently recomputes
the source identity, rather than accepting an arbitrary supplied digest.

The specialization combines three categorical logits with one answerability
logit. Decision cross-entropy is masked on answerable rows; Bernoulli
answerability loss uses every admitted row. An all-unanswerable batch retains a
differentiable zero Decision term. No null target becomes a categorical label.

Inference accepts only bounded supplied evidence under the canonical
`runtime.recovery.v1` policy/question. Canonical retry/fallback/stop order breaks
ties; diagnostics are returned in request choice order. Since calibration/release
support is absent, envelopes abstain, payloads are null, and diagnostics carry
raw advisory research information. There is no action executor.

`DecisionProvider` is separate from Core: an offline host supplies actual admitted
execution and owns process/resource cleanup. The CLI's disposable inference host
caps address space/CPU/deadline, binds the trusted artifact digest and waits for
process exit. It does not claim a Runtime receipt/holder transfer. Host permission
and Workflow state remain outside the model.

The experimental constructor is `DecisionProvider(decision, artifact_digest,
host_infer)`, with an explicit source-backed `Decision`, a trusted SHA-256 bundle
identity and the host callable. It checks canonical policy/question and token
bounds before invoking the host, protects request identity against host mutation,
and validates response binding, request-order probability coverage/normalization
and research-only abstention. A generic schema registry alone is not a TaskSpec
or an admission capability. Successful envelopes cannot claim `ok` in this
uncalibrated reference profile. Logits are bounded to ±1,000,000; native and Python
probability postprocessing use FP64 so near ties are not manufactured by FP32
probability rounding. Model training and weights remain CPU FP32.

The retained legacy partition is explicitly compiled, source-pinned, and confined
to research fixtures. Native and Python reject unknown profiles, forged source
rows/targets and incomplete family membership/counts. Core's normal fingerprint
partition remains the default for new non-legacy datasets.
