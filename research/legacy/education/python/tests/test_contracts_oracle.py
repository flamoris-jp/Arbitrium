from pathlib import Path
from itertools import product
import copy
import json
import pytest
from arbitrium_education import contracts as c
from arbitrium_education.oracle import DOMAINS, decide, render, resolve_events

ROOT = Path(__file__).resolve().parents[3]

def fixture(name):
    return json.loads((ROOT/'docs/examples'/name).read_text())


def test_shared_fixtures_and_packaged_schemas():
    spec = fixture('recovery-task.json')
    req = fixture('recovery-request.json')
    c.request(req, spec)
    c.sample(fixture('recovery-sample.json'), spec)
    for name in ['recovery-result.json', 'abstain-result.json', 'error-result.json']:
        c.result(fixture(name), spec, req)
    for path in (ROOT/'docs/contracts').glob('*.json'):
        assert path.read_bytes() == (ROOT/'education/python/src/arbitrium_education/schemas'/path.name).read_bytes()


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":{"b":1,"b":2}}',
                                     'NaN', '1e999', '"\\ud800"', b'"\xff"'])
def test_strict_json(raw):
    with pytest.raises(c.ContractError): c.loads(raw)


@pytest.mark.parametrize('field,value', [('state',' '), ('state','a\x00b'), ('policy','new policy'),
    ('question','new question'), ('language','ja'), ('choices',['retry','stop']),
    ('choices',['retry','retry','stop']), ('extra', True)])
def test_request_mutations(field, value):
    req=fixture('recovery-request.json');req[field]=value
    with pytest.raises(c.ContractError): c.request(req, fixture('recovery-task.json'))


def test_byte_limit_and_newline_normalization():
    spec=fixture('recovery-task.json');req=fixture('recovery-request.json')
    req['state']='é'*8193
    with pytest.raises(c.ContractError): c.request(req,spec)
    req['state']='before\r\nafter\rfinal'
    assert c.request(req,spec)['state']=='before\nafter\nfinal'


def test_permuted_results_and_ties():
    spec=fixture('recovery-task.json');req=fixture('recovery-request.json')
    out=fixture('recovery-result.json');req['choices'].reverse();out['probabilities'].reverse()
    c.result(out,spec,req)
    out['probabilities'][0]['probability']=0.9
    with pytest.raises(c.ContractError):c.result(out,spec,req)


def test_oracle_complete_truth_table():
    for values in product(*DOMAINS.values()):
        f=dict(zip(DOMAINS,values));t=decide(f)
        # Independent precedence formulation.
        retry=(values[0] is True and values[1]!='permanent' and values[2]!='zero')
        fallback=values[3] and values[4]
        expected='stop' if values[5] else ('retry' if retry else 'fallback' if fallback else 'stop')
        assert t.answerable and t.label==expected
        assert render(f)


def test_all_partial_oracles_equal_refinement_union():
    for values in product(*(tuple(d)+('unknown',) for d in DOMAINS.values())):
        f=dict(zip(DOMAINS,values));t=decide(f)
        missing=[k for k,v in f.items() if v=='unknown']
        outcomes=set()
        for completion in product(*(DOMAINS[k] for k in missing)):
            complete=f|dict(zip(missing,completion));outcomes.add(decide(complete).label)
        assert t.answerable==(len(outcomes)==1)
        assert t.label==(next(iter(outcomes)) if len(outcomes)==1 else None)


def test_irrelevant_unknown_and_temporal_override():
    f={k:'unknown' for k in DOMAINS};f['fatal_violation']=True
    assert decide(f).label=='stop'
    assert not decide(f, contradictory=True).answerable
    events=[{'time':1,'fact':'fatal_violation','value':True}, {'time':1,'fact':'fatal_violation','value':False}]
    assert resolve_events(events)[1]
    events.append({'time':2,'fact':'fatal_violation','value':False})
    assert resolve_events(events)[0]['fatal_violation'] is False
    assert not resolve_events(events)[1]


def test_no_bool_coercion():
    f={k:'unknown' for k in DOMAINS};f['repeat_safe']=1
    with pytest.raises(ValueError):decide(f)
