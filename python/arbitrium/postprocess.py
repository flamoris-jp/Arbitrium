"""Decision gates are advisory policy; never workflow execution."""
import math
LABELS=('retry','fallback','stop')
GRID=(.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99)
def postprocess(raw,choices=LABELS,*,temperature=1.,answer_temperature=1.,tau_p=None,tau_q=None,accept_none=True):
    if len(raw)!=4 or any(type(v) not in (int,float) or not -1e6<=v<=1e6 for v in raw):raise ValueError('bounded finite four logits')
    if len(choices)!=3 or set(choices)!=set(LABELS):raise ValueError('exact choices')
    if any(type(t) not in (int,float) or not math.exp(-4)<=t<=math.exp(4) for t in (temperature,answer_temperature)):raise ValueError('temperature range')
    if type(accept_none) is not bool:raise ValueError('gate type')
    if accept_none:
        if tau_p is not None or tau_q is not None:raise ValueError('accept-none gates')
    elif tau_p not in GRID or tau_q not in GRID:raise ValueError('registered gate grid')
    x=[v/temperature for v in raw[:3]];offset=max(x);exp=[math.exp(v-offset) for v in x];p=[v/sum(exp) for v in exp]
    z=raw[3]/answer_temperature;q=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
    winner=max(range(3),key=lambda i:p[i]);reason='release_gate_not_met' if accept_none else 'low_answerability' if q<tau_q else 'low_confidence' if p[winner]<tau_p else None
    return dict(status='abstain' if reason else 'ok',verdict=None if reason else LABELS[winner],diagnostic_label=LABELS[winner],probabilities=[dict(label=k,probability=p[LABELS.index(k)]) for k in choices],answerability=q,reason_code=reason)
