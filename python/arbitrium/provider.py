"""Separate offline provider boundary with host-supplied admitted execution."""
import copy,math,re
class DecisionProvider:
    def __init__(self,decision,artifact_digest,host_infer):
        if not callable(host_infer):raise ValueError('explicit admitted host required')
        if type(artifact_digest) is not str or re.fullmatch('[a-f0-9]{64}',artifact_digest) is None:raise ValueError('trusted artifact digest required')
        self.decision=decision;self.registry=decision.registry;self.artifact_digest=artifact_digest;self.host_infer=host_infer
    def infer(self,request):
        # Do not let a host mutate the request used to bind its response.
        bound=copy.deepcopy(request);self.registry.request(bound);self.decision.assemble(bound['payload'])
        result=self.host_infer(copy.deepcopy(bound))
        self.registry.result(result,bound,self.artifact_digest)
        if result['status']=='ok':raise ValueError('uncalibrated Decision cannot return actionable output')
        if result['status']=='abstain':
            probabilities=result['diagnostics']['probabilities']
            if [p['label'] for p in probabilities]!=bound['payload']['choices']:raise ValueError('Decision probability order/coverage')
            if not math.isclose(sum(p['probability'] for p in probabilities),1.,rel_tol=0.,abs_tol=1e-6):raise ValueError('Decision probability normalization')
        return result
