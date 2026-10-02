import copy,unittest,tempfile
from pathlib import Path
from arbitrium.composition import Decision
from arbitrium.legacy import verify_archive,SPLITS
from arbitrium.oracle import decide
from arbitrium.metrics import diagnostic
from arbitrium.postprocess import postprocess
from maidionis_education.datasets import validate_dataset,freeze
from maidionis_education.contracts import loads
class DecisionTests(unittest.TestCase):
    def test_archive_complete_byte_integrity(self):self.assertEqual(len(verify_archive()['files']),130)
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
    def test_codec_fixed_policy_bounds(self):
        d=Decision('recovery-current-tiny-v1');x={k:d.legacy_rows[0]['input'][k] for k in ('state','policy','question','choices')}
        ids,seg=d.assemble(x);self.assertEqual(ids[0],4);self.assertEqual(ids[-1],3);self.assertEqual(len(ids),len(seg))
        with self.assertRaises(ValueError):d.assemble(dict(x,policy='arbitrary policy'))
        with self.assertRaises(ValueError):d.assemble(dict(x,state='word '*400))
if __name__=='__main__':unittest.main()
