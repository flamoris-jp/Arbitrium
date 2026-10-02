import copy,os,shutil,tempfile,unittest
from pathlib import Path
from arbitrium.composition import Decision,BUILD,payload
from maidionis_education.contracts import canonical,digest,loads
from maidionis_education.storage import inventory
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_experiments import run,configs
DRIVER=os.environ.get('ARBITRIUM_DRIVER')
@unittest.skipUnless(DRIVER,'tools/verify.py supplies the actual native driver')
class NativeDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.d=Decision('recovery-current-tiny-v1');cls.data=cls.root/'data';cls.dh=cls.d.freeze(cls.data)
        cls.config=dict(configs()[0][1],seed=42,epochs=2,patience=2);cls.cfg=cls.root/'config.json';cls.cfg.write_bytes(canonical(cls.config))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def train(self,checkpoints,out,n,resume):return run(DRIVER,'train',self.data,self.dh,checkpoints,out,n,resume,self.cfg,'none')
    def test_actual_composition_masking_gradients_ties_and_bounds(self):
        self.assertEqual(run(DRIVER,'composition-self-test',self.data),dict(masked_gradient=True,bounded_decode=True,canonical_tie=True,near_tie=True))
    def test_codec_native_python_parity_reordering_and_bounds(self):
        self.assertEqual(run(DRIVER,'build-identity')['build_digest'],BUILD)
        x=payload(self.d.legacy_rows[0]);x['choices']=['stop','retry','fallback'];p=self.root/'request.json';p.write_bytes(canonical(x));out=run(DRIVER,'encode',self.data,p)
        ids,segments=self.d.assemble(x);self.assertEqual(out,dict(ids=ids,segments=segments))
        x['policy']='unsupported policy';p.write_bytes(canonical(x));run(DRIVER,'encode',self.data,p,success=False)
        x=payload(self.d.legacy_rows[0]);x['state']='word '*400;p.write_bytes(canonical(x));run(DRIVER,'encode',self.data,p,success=False)
    def test_fresh_process_resume_bit_exact_and_source_substitution_rejected(self):
        full=self.train(self.root/'full-checkpoints',self.root/'full',0,0)
        first=self.train(self.root/'resumed-checkpoints',self.root/'first',1,0)
        resumed=self.train(self.root/'resumed-checkpoints',self.root/'resumed',0,1)
        self.assertGreater(resumed['gradient_l1'],0);self.assertTrue(full['parameters_changed'])
        for k in ('state','final_sha256','best_sha256','best_logits'):self.assertEqual(full[k],resumed[k],k)
        bad=self.root/'bad';shutil.copytree(self.data,bad);line=loads((bad/'train.jsonl').read_bytes().splitlines()[0]);line['target']['label']='stop' if line['target']['label']!='stop' else 'retry'
        raw=(bad/'train.jsonl').read_bytes().splitlines();raw[0]=canonical(line).rstrip(b'\n');(bad/'train.jsonl').write_bytes(b'\n'.join(raw)+b'\n')
        m=loads((bad/'manifest.json').read_bytes(),4*2**20);files={p.relative_to(bad).as_posix():p.read_bytes() for p in bad.rglob('*') if p.is_file() and p.name!='manifest.json'};m['files']=inventory(files);mraw=canonical(m);(bad/'manifest.json').write_bytes(mraw)
        run(DRIVER,'train',bad,digest(mraw),self.root/'bad-checkpoints',self.root/'bad-output',0,0,self.cfg,'none',success=False)
    def test_compiled_profile_cannot_be_installed_by_metadata(self):
        bad=self.root/'neutral-with-legacy';shutil.copytree(self.data,bad)
        cfg=loads((bad/'split.config.json').read_bytes());cfg['algorithm']='maidionis-split-v1';(bad/'split.config.json').write_bytes(canonical(cfg))
        m=loads((bad/'manifest.json').read_bytes(),4*2**20);files={p.relative_to(bad).as_posix():p.read_bytes() for p in bad.rglob('*') if p.is_file() and p.name!='manifest.json'};m['files']=inventory(files);m['split_profile']['id']='maidionis-split';m['split_profile']['config_digest']=digest(files['split.config.json']);raw=canonical(m);(bad/'manifest.json').write_bytes(raw)
        run(DRIVER,'train',bad,digest(raw),self.root/'neutral-checkpoints',self.root/'neutral-output',0,0,self.cfg,'none',success=False)
    def test_rehashed_unknown_partition_is_rejected(self):
        bad=self.root/'bad-partition';shutil.copytree(self.data,bad);cfg=loads((bad/'split.config.json').read_bytes());cfg['algorithm']='unknown.partition';(bad/'split.config.json').write_bytes(canonical(cfg))
        m=loads((bad/'manifest.json').read_bytes(),4*2**20);files={p.relative_to(bad).as_posix():p.read_bytes() for p in bad.rglob('*') if p.is_file() and p.name!='manifest.json'};m['files']=inventory(files);m['split_profile']['config_digest']=digest(files['split.config.json']);raw=canonical(m);(bad/'manifest.json').write_bytes(raw)
        run(DRIVER,'train',bad,digest(raw),self.root/'partition-checkpoints',self.root/'partition-output',0,0,self.cfg,'none',success=False)
    def test_rehashed_provenance_alias_is_rejected(self):
        bad=self.root/'bad-provenance';shutil.copytree(self.data,bad)
        lines=(bad/'train.jsonl').read_bytes().splitlines();row=loads(lines[0]);row['provenance_id']='forged-proof';lines[0]=canonical(row).rstrip(b'\n');(bad/'train.jsonl').write_bytes(b'\n'.join(lines)+b'\n')
        m=loads((bad/'manifest.json').read_bytes(),4*2**20);files={p.relative_to(bad).as_posix():p.read_bytes() for p in bad.rglob('*') if p.is_file() and p.name!='manifest.json'};m['files']=inventory(files);raw=canonical(m);(bad/'manifest.json').write_bytes(raw)
        run(DRIVER,'train',bad,digest(raw),self.root/'provenance-checkpoints',self.root/'provenance-output',0,0,self.cfg,'none',success=False)
