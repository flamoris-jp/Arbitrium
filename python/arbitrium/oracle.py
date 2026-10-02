"""Structured finite oracle for education/audit, never text inference."""
from itertools import product
DOMAINS=dict(repeat_safe=(False,True),provider_state=('healthy','transient','permanent'),retry_budget=('zero','positive'),fallback_available=(False,True),fallback_permitted=(False,True),fatal_violation=(False,True))
def decide(facts,contradictory=False):
    if type(contradictory) is not bool or type(facts) is not dict or set(facts)!=set(DOMAINS):raise ValueError('finite facts shape')
    for k,v in facts.items():
        if v!='unknown' and not any(type(v) is type(a) and v==a for a in DOMAINS[k]):raise ValueError('finite facts value')
    def complete(f):
        if f['fatal_violation']:return 'stop'
        if f['repeat_safe'] and f['provider_state'] in ('healthy','transient') and f['retry_budget']=='positive':return 'retry'
        return 'fallback' if f['fallback_available'] and f['fallback_permitted'] else 'stop'
    outcomes={complete(dict(zip(DOMAINS,x))) for x in product(*(DOMAINS[k] if facts[k]=='unknown' else (facts[k],) for k in DOMAINS))}
    answerable=not contradictory and len(outcomes)==1
    return dict(answerable=answerable,label=next(iter(outcomes)) if answerable else None)
