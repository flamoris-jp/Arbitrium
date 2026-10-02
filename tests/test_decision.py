import copy,unittest,tempfile
from pathlib import Path
from arbitrium.composition import Decision
from arbitrium.legacy import verify_archive,SPLITS
from arbitrium.oracle import decide
from arbitrium.metrics import diagnostic
from arbitrium.postprocess import postprocess
from arbitrium.provider import DecisionProvider
from arbitrium.composition import IDENTITY,ref,payload
from maidionis_education.datasets import validate_dataset,freeze
from maidionis_education.contracts import loads
class DecisionTests(unittest.TestCase):
    def test_archive_complete_byte_integrity(self):self.assertEqual(len(verify_archive()['files']),130)
    def test_legacy_name_is_not_a_path_discovery_mechanism(self):
        for name in ('/tmp/outside','../../../outside','unknown',None,{'dataset':'recovery-current-tiny-v1'}):
            with self.assertRaises(ValueError):Decision(name)
    def test_oracle_unknown_conflict_and_precedence(self):
        f=dict(repeat_safe=True,provider_state='transient',retry_budget='positive',fallback_available=True,fallback_permitted=True,fatal_violation=False)
        self.assertEqual(decide(f)['label'],'retry');self.assertEqual(decide(dict(f,fatal_violation=True))['label'],'stop')
        self.assertFalse(decide(dict(f,fatal_violation='unknown'))['answerable']);self.assertFalse(decide(f,True)['answerable'])
        with self.assertRaises(ValueError):decide(dict(f,repeat_safe=1))
    def test_all_derivatives_preserve_splits_and_bind_legacy_rows(self):
        for name in ('recovery-current-tiny-v1','recovery-current-diagnostic-v1','recovery-current-grouped-v1','controlled-recovery-1000-v1'):
            d=Decision(name)
            with tempfile.TemporaryDirectory() as td:
                h=d.freeze(Path(td)/'data');m,rows=d.validate(Path(td)/'data',h)
                self.assertNotEqual(h,d.legacy_digest)
                for s in SPLITS:self.assertEqual(m['counts'][s]['records'],d.manifest['counts'][s]['records'])
                byid={r['sample_id']:r for r in d.legacy_rows}
                self.assertTrue(all(r['target']==byid[r['sample_id']]['target'] and r['split']==byid[r['sample_id']]['split'] for r in rows))
                row=copy.deepcopy(rows[0]);row['target']['label']='stop' if row['target']['label']!='stop' else 'retry'
                p=dict(ancestry=[d.legacy_digest,row['sample_id'],row['family_id']]);self.assertFalse(d.verify_row(row,p))
    def test_gate_ties_reordering_and_absent_release(self):
        out=postprocess([0,0,0,0],['stop','retry','fallback']);self.assertEqual(out['diagnostic_label'],'retry');self.assertEqual(out['status'],'abstain')
        self.assertEqual([x['label'] for x in out['probabilities']],['stop','retry','fallback'])
        self.assertEqual(postprocess([20,0,0,-20],tau_p=.5,tau_q=.5,accept_none=False)['reason_code'],'low_answerability')
        self.assertEqual(postprocess([20,0,0,20],tau_p=.5,tau_q=.5,accept_none=False)['verdict'],'retry')
        for raw in ([0,0,0],[float('nan'),0,0,0]):
            with self.assertRaises(ValueError):postprocess(raw)
        with self.assertRaises(ValueError):postprocess([0]*4,tau_p=.2,tau_q=.5,accept_none=False)
    def test_original_family_and_provenance_aliases_cannot_be_rewritten(self):
        d=Decision('recovery-current-tiny-v1')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'data';d.freeze(root)
            rows=[loads(line) for line in (root/'train.jsonl').read_bytes().splitlines()]
            proofs=[loads(line) for line in (root/'provenance.jsonl').read_bytes().splitlines()]
            byid={p['sample_id']:p for p in proofs}
            for key in ('family_id','provenance_id'):
                row=copy.deepcopy(rows[0]);row[key]='forged-alias'
                self.assertFalse(d.verify_row(row,byid[row['sample_id']]))
            for row in rows:row['family_id']='forged-family'
            with self.assertRaises(ValueError):
                freeze(Path(td)/'forged',rows,proofs,d.registry,d.hooks,d.files(),dataset_id=d.name+'.maidionis.v1',seed=42,created_at='2026-10-02T00:00:00Z',generator=dict(code_digest='a'*64,config_digest='b'*64),license_summary='Apache-2.0',limitations=['research'])
    def test_legacy_dataset_cannot_claim_registered_evaluation(self):
        d=Decision('recovery-current-tiny-v1')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'data';d.freeze(root)
            rows=[loads(line) for line in (root/'train.jsonl').read_bytes().splitlines()]
            proofs=[loads(line) for line in (root/'provenance.jsonl').read_bytes().splitlines()]
            promoted=Path(td)/'promoted'
            h=freeze(promoted,rows,proofs,d.registry,d.hooks,d.files(),dataset_id=d.name+'.maidionis.v1',seed=42,created_at='2026-10-02T00:00:00Z',generator=dict(code_digest='a'*64,config_digest='b'*64),license_summary='Apache-2.0',limitations=['research'],purpose='registered_evaluation')
            with self.assertRaises(ValueError):d.validate(promoted,h)
    def test_undefined_support_and_hand_counted_metrics(self):
        m=diagnostic([dict(target=dict(answerable=True,label='retry')),dict(target=dict(answerable=False,label=None))],[[20,0,0,20],[0,0,0,-20]])
        self.assertEqual(m['decision_accuracy'],1);self.assertEqual(m['unanswerable_recall'],1);self.assertIsNone(m['label_recall']['fallback']);self.assertIsNone(m['macro_f1'])
    def test_extreme_logits_and_invalid_temperature_fail_closed(self):
        for raw in ([1e308,0,0,0],[10**400,0,0,0],[float('inf'),0,0,0],[float('nan'),0,0,0],[-1000001,0,0,0]):
            with self.assertRaises(ValueError):postprocess(raw)
        for t in (True,None,'1',float('nan'),float('inf'),10**400):
            with self.assertRaises(ValueError):postprocess([0]*4,temperature=t)
        out=postprocess([1e6,-1e6,0,-1e6],temperature=.02,answer_temperature=.02)
        self.assertEqual([p['probability'] for p in out['probabilities']],[1.,0.,0.]);self.assertEqual(out['answerability'],0.)
    def test_codec_fixed_policy_bounds(self):
        d=Decision('recovery-current-tiny-v1');x={k:d.legacy_rows[0]['input'][k] for k in ('state','policy','question','choices')}
        ids,seg=d.assemble(x);self.assertEqual(ids[0],4);self.assertEqual(ids[-1],3);self.assertEqual(len(ids),len(seg))
        with self.assertRaises(ValueError):d.assemble(dict(x,policy='arbitrary policy'))
        with self.assertRaises(ValueError):d.assemble(dict(x,state='word '*400))

class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=Decision('recovery-current-tiny-v1')
    def setUp(self):
        x=payload(self.d.legacy_rows[0]);x['choices']=['stop','retry','fallback']
        self.request=dict(schema_version='maidionis.request.v1',request_id='provider:test',**IDENTITY,payload_schema=ref('input'),payload=x,context_ref=None)
        raw=postprocess([2,0,0,2],x['choices'])
        diagnostics=dict(diagnostic_label=raw['diagnostic_label'],advisory_label=None,probabilities=raw['probabilities'],answerability=raw['answerability'],reason_code='release_gate_not_met',calibration_status='uncalibrated')
        self.result=dict(schema_version='maidionis.result.v1',request_id=self.request['request_id'],**IDENTITY,artifact_digest='a'*64,context_ref=None,payload_schema=ref('output'),status='abstain',payload=None,diagnostics=diagnostics,error=None)
    def test_codec_rejects_before_host_and_valid_abstention_passes(self):
        calls=[];provider=DecisionProvider(self.d,'a'*64,lambda r:calls.append(r) or copy.deepcopy(self.result))
        self.assertEqual(provider.infer(self.request),self.result);self.assertEqual(len(calls),1)
        for x in (dict(self.request['payload'],policy='arbitrary'),dict(self.request['payload'],state='word '*400)):
            with self.assertRaises(ValueError):provider.infer(dict(self.request,payload=x))
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):DecisionProvider(self.d,'untrusted',lambda r:self.result)
    def test_actionable_duplicate_unordered_and_unnormalized_results_rejected(self):
        bad=[]
        ok=copy.deepcopy(self.result);ok.update(status='ok',payload=ok['diagnostics']);bad.append(ok)
        duplicate=copy.deepcopy(self.result);duplicate['diagnostics']['probabilities'][0]['label']='retry';bad.append(duplicate)
        unordered=copy.deepcopy(self.result);unordered['diagnostics']['probabilities'].reverse();bad.append(unordered)
        unnormalized=copy.deepcopy(self.result)
        for p in unnormalized['diagnostics']['probabilities']:p['probability']=.1
        bad.append(unnormalized)
        for result in bad:
            with self.assertRaises(ValueError):DecisionProvider(self.d,'a'*64,lambda r:result).infer(self.request)
    def test_host_cannot_rebind_request_by_mutating_it(self):
        def host(request):
            request['request_id']='forged';result=copy.deepcopy(self.result);result['request_id']=request['request_id'];return result
        with self.assertRaises(ValueError):DecisionProvider(self.d,'a'*64,host).infer(self.request)
        self.assertEqual(self.request['request_id'],'provider:test')
if __name__=='__main__':unittest.main()
