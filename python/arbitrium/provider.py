"""Separate offline provider boundary with host-supplied admitted execution."""
class DecisionProvider:
    def __init__(self,registry,artifact_digest,host_infer):
        if not callable(host_infer):raise ValueError('explicit admitted host required')
        self.registry=registry;self.artifact_digest=artifact_digest;self.host_infer=host_infer
    def infer(self,request):
        self.registry.request(request);result=self.host_infer(request)
        self.registry.result(result,request,self.artifact_digest);return result
