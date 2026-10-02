import json
from pathlib import Path
from dataclasses import asdict
import httpx
import pytest
from arbitrium_education.journal import Journal
from arbitrium_education.gpt_oss import OpenAICompatibleGptOssTeacher, ModelIdentity, Budget
from arbitrium_education.teacher import CurriculumRequest, CandidateLesson, ReviewResult
from arbitrium_education.orchestrator import EducationOrchestrator
from arbitrium_education.oracle import DOMAINS

IDENTITY=ModelIdentity('gpt-oss','a'*64,'Q4','test-version','b'*64)
TASK=json.loads((Path(__file__).resolve().parents[3]/'docs/examples/recovery-task.json').read_text())
FACTS={k:'unknown' for k in DOMAINS}|{'fatal_violation':True}
SPEC=CurriculumRequest('negation',1,'scenario1',FACTS,TASK,1)

def envelope(value,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(value)}}]}


def test_blind_review_never_overwrites_or_verifies(tmp_path):
    candidate=CandidateLesson('A fatal policy violation occurred.','retry',True,'secret teacher rationale')
    class Teacher:
        def generate(self,spec):return [candidate]
    class Reviewer:
        def review(self,request):
            serialized=json.dumps(asdict(request))
            assert 'secret teacher rationale' not in serialized
            assert 'proposed_target' not in serialized
            return ReviewResult('correct','stop',True,FACTS,(),'correct proposal')
    with Journal(tmp_path) as journal:
        rows=EducationOrchestrator(Teacher(),Reviewer(),journal).generate_and_review(SPEC)
    assert rows[0].disposition=='quarantine'
    assert candidate.proposed_target=='retry'


def test_retries_replay_and_temperatures(tmp_path):
    calls=[]; sleeps=[]
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:return httpx.Response(429,headers={'Retry-After':'0'},json={'error':'busy'})
        return httpx.Response(200,json=envelope({'lessons':[{'state':'Fatal violation confirmed.',
            'proposed_target':'stop','proposed_answerable':True}]}))
    with Journal(tmp_path) as j:
        with OpenAICompatibleGptOssTeacher(base_url='http://localhost:8080',identity=IDENTITY,journal=j,
                    transport=httpx.MockTransport(handler),sleep=sleeps.append) as provider:
            assert provider.generate(SPEC)[0].proposed_target=='stop'
            assert calls[-1]['temperature']==.8
            # First failed attempt is reattempted, successful attempt is immutable/replayed.
            provider.generate(SPEC)
            assert provider.budget.attempts==2
    assert len(sleeps)==1

@pytest.mark.parametrize('url',['http://localhost/v1','http://user:pw@localhost','https://example.org','http://localhost/?x=1'])
def test_origin_validation(tmp_path,url):
    with Journal(tmp_path) as j:
        with pytest.raises(ValueError):OpenAICompatibleGptOssTeacher(base_url=url,identity=IDENTITY,journal=j)

@pytest.mark.parametrize('kind',['string_bool','truncated','401','redirect','timeout'])
def test_failures_do_not_accept(tmp_path,kind):
    calls=[]
    def handler(req):
        calls.append(req)
        if kind=='timeout':raise httpx.ReadTimeout('timeout')
        if kind=='401':return httpx.Response(401,json={})
        if kind=='redirect':return httpx.Response(302,headers={'Location':'http://other'})
        row={'state':'Fatal violation.','proposed_target':'stop','proposed_answerable':'false' if kind=='string_bool' else True}
        return httpx.Response(200,json=envelope({'lessons':[row]},'length' if kind=='truncated' else 'stop'))
    with Journal(tmp_path) as j:
        with OpenAICompatibleGptOssTeacher(base_url='http://localhost',identity=IDENTITY,journal=j,
                    transport=httpx.MockTransport(handler),sleep=lambda _:None) as p:
            with pytest.raises((ValueError,httpx.HTTPError)):p.generate(SPEC)
    assert len(calls)==(3 if kind=='timeout' else 1)


def test_journal_lock_corruption_and_budget(tmp_path):
    with Journal(tmp_path) as j:
        with pytest.raises(BlockingIOError):
            with Journal(tmp_path):pass
        j.save_response('abc',b'content')
        assert j.replay('abc')==b'content'
        with pytest.raises(ValueError):j.save_response('abc',b'other')
        (tmp_path/'responses/abc').write_bytes(b'bad')
        with pytest.raises(ValueError):j.replay('abc')
    b=Budget(max_attempts=1);b.take(0)
    with pytest.raises(RuntimeError):b.take(1)
