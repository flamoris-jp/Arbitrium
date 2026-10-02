"""Explicit Decision registries/codec/dataset adapters; no Core defaults."""
from pathlib import Path
import re,sys,json
from maidionis_education.contracts import canonical,digest,loads,RegistryBuilder
from maidionis_education.datasets import DatasetHooks,prepare_sample,freeze,validate_dataset
from .legacy import ROOT,ARCHIVE,dataset,split_for,SPLITS
from .oracle import decide
sys.path.insert(0,str(ROOT/'tools'))
from source_identity import build_digest
LOCK=loads((ROOT/'dependency.lock.json').read_bytes())
BUILD=build_digest(LOCK['core_build_digest'])
IDENTITY=dict(specialization_id='arbitrium.decision',specialization_version='1',task_id='runtime.recovery.v1',task_version='1')
LABELS=['retry','fallback','stop']
TIME='2026-10-02T00:00:00Z'
def obj(fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
TEXT=dict(type='string',minLength=1,maxLength=16384,pattern=r'^[\x20-\x7e\t\r\n]+$')
INPUT=obj(dict(state=TEXT,policy=TEXT,question=TEXT,choices=dict(type='array',minItems=3,maxItems=3,items=dict(type='string',enum=LABELS),uniqueItems=True)))
TARGET=obj(dict(answerable=dict(type='boolean'),label=dict(anyOf=[dict(type='null'),dict(type='string',enum=LABELS)])))
OUTPUT=obj(dict(diagnostic_label=dict(type='string',enum=LABELS),advisory_label=dict(type='null'),probabilities=dict(type='array',minItems=3,maxItems=3,items=obj(dict(label=dict(type='string',enum=LABELS),probability=dict(type='number',minimum=0,maximum=1)))),answerability=dict(type='number',minimum=0,maximum=1),reason_code=dict(const='release_gate_not_met'),calibration_status=dict(const='uncalibrated')))
RAW=dict(type='array',minItems=4,maxItems=4,items=dict(type='number',minimum=-1e6,maximum=1e6))
SCHEMAS=dict(input=INPUT,target=TARGET,output=OUTPUT,raw=RAW)
def ref(name):return dict(id='arbitrium.decision.'+name,version='1',sha256=digest(canonical(SCHEMAS[name])))
def component(name,value):return dict(id='arbitrium.decision.'+name,version='1',config_digest=digest(canonical(value)))
def words(text):return re.findall(r'[A-Za-z0-9]+|[^\s]',text,flags=re.ASCII)
def payload(row):return {k:row['input'][k] for k in ('state','policy','question','choices')}
class Decision:
    def __init__(self,name):
        self.manifest,self.legacy_digest,self.legacy_rows,self.proofs=dataset(name)
        self.name=name;self.rows_by_state={r['input']['state']:r for r in self.legacy_rows}
        if len(self.rows_by_state)!=len(self.legacy_rows):raise ValueError('duplicate legacy state')
        self.task=loads((ARCHIVE/'examples/education'/name/'task.json').read_bytes())
        self.vocabulary=sorted({w for r in self.legacy_rows if r['split']=='train' for k in ('task_id','policy','state','question') for w in words(r['input'][k])})
        self.configs=dict(input_codec=dict(kind='ascii-word-fields.v1',vocabulary=self.vocabulary,legacy_dataset=name,legacy_manifest=self.legacy_digest,assembly='fields.v1',max_length=256),
            output_codec=dict(calibration='absent',accept_none=True,threshold_milli=500,tie='canonical_first'),
            architecture=dict(kind='pooled',hidden_width=256,output_width=4,dropout_milli=100),
            objective=dict(kind='masked_categorical_plus_answerability',decision_outputs=3,answerability_index=3),
            head=dict(labels=LABELS,answerability=True),numerical_compatibility=dict(profile='linux.cpu.fp32.serial.v1',libtorch='2.5.1',archive=1))
        self.semantic=canonical(self.task)
        self.descriptor=dict(schema_version='maidionis.specialization.v1',**IDENTITY,input_schema=ref('input'),target_schema=ref('target'),output_schema=ref('output'),diagnostics_schema=ref('output'),
            semantic_spec=dict(path='semantic.json',sha256=digest(self.semantic)),heads=[component('head',self.configs['head'])],**{k:component(k,v) for k,v in self.configs.items() if k!='head'})
        b=RegistryBuilder(BUILD)
        for k,s in SCHEMAS.items():b.add_schema(ref(k),s)
        for k,c in self.configs.items():b.add_operation(component(k,c),c,lambda v:None)
        self.registry=b.freeze(self.descriptor)
        self.group=component('legacy-group',dict(algorithm='manifest-family-ancestry.v1'))
        self.verify=component('legacy-oracle',dict(oracle='finite-recovery.v1',legacy_manifest=self.legacy_digest))
        self.hook=component('legacy-hooks',dict(build_digest=BUILD,legacy_manifest=self.legacy_digest))
        self.generator=component('legacy-conversion',dict(legacy_manifest=self.legacy_digest,adapter='neutral-derivative.v1'))
        self.hooks=DatasetHooks(self.hook,self.group,component('dedup',dict(projection='state-casefold-space.v1')),self.verify,
            self.roots,lambda x:' '.join(x['state'].casefold().split()),self.verify_row,
            split_algorithm='arbitrium.legacy-partition.v1',assign_split=self.partition)
    def source(self,x):
        row=self.rows_by_state.get(x['state'])
        if row is None or x!=payload(row):raise ValueError('not an inventoried legacy input')
        return row
    def roots(self,x):return [dict(legacy_manifest=self.legacy_digest,legacy_family=self.source(x)['family_id'])]
    def partition(self,x,fp,seed):
        if seed!=42:raise ValueError('legacy split seed')
        return split_for(self.source(x)['family_id'])
    def verify_row(self,row,p):
        old=self.source(row['input']);proof=self.proofs[old['provenance_id']]
        expected=decide(proof['facts'],proof.get('contradictory',False))
        return row['target']==old['target']==expected and row['sample_id']==old['sample_id'] and p['ancestry']==[self.legacy_digest,old['sample_id'],old['family_id']] and row['supersedes'] is None
    def files(self):return dict({'descriptor.json':canonical(self.descriptor),'semantic.json':self.semantic},**{k+'.schema.json':canonical(v) for k,v in SCHEMAS.items()},**{k+'.config.json':canonical(v) for k,v in self.configs.items()})
    def assemble(self,x):
        self.registry.payload('input_schema',x)
        if set(x['choices'])!=set(LABELS) or x['policy']!=self.task['canonical_policy'] or x['question']!=self.task['question']:raise ValueError('Decision TaskSpec binding')
        lookup={w:i+9 for i,w in enumerate(self.vocabulary)}
        tokens=[4,5];segments=[0,0]
        for k,control,segment in [('task_id',None,0),('policy',6,1),('state',7,2),('question',8,3)]:
            if control is not None:tokens.append(control);segments.append(segment)
            text=IDENTITY['task_id'] if k=='task_id' else x[k]
            ids=[lookup.get(w,1) for w in words(text)];tokens.extend(ids);segments.extend([segment]*len(ids))
        tokens.append(3);segments.append(3)
        if len(tokens)>256:raise ValueError('assembled token bound')
        return tokens,segments
    def validate(self,root,trusted_digest):
        m,rows=validate_dataset(root,trusted_digest,self.registry,self.hooks)
        if m['dataset_id']!=self.name+'.maidionis.v1' or len(rows)!=len(self.legacy_rows):raise ValueError('complete legacy conversion identity')
        for split in SPLITS:
            if m['counts'][split]['records']!=self.manifest['counts'][split]['records'] or m['counts'][split]['families']!=self.manifest['counts'][split]['families']:raise ValueError('original split coverage')
        if {r['sample_id'] for r in rows}!={r['sample_id'] for r in self.legacy_rows}:raise ValueError('complete source record coverage')
        return m,rows
    def freeze(self,root):
        rows=[];proofs=[];dataset_id=self.name+'.maidionis.v1'
        for old in self.legacy_rows:
            x=payload(old);self.assemble(x)
            row=dict(schema_version='maidionis.sample.v1',sample_id=old['sample_id'],family_id=old['family_id'],dataset_id=dataset_id,**IDENTITY,input_schema=ref('input'),input=x,target_schema=ref('target'),target=old['target'],provenance_id=old['provenance_id'],verification_profile=self.verify,verification_status='verified',tags=['legacy-derived'],supersedes=None)
            row=prepare_sample(row,self.registry,self.hooks,42);rows.append(row)
            proofs.append(dict(schema_version='maidionis.provenance.v1',provenance_id=row['provenance_id'],sample_id=row['sample_id'],ancestry=[self.legacy_digest,row['sample_id'],row['family_id']],source_digest=digest(canonical(x)),source_license='Apache-2.0',generator=self.generator,teacher=None,reviewer=None,non_model_source_reason='Inventoried synthetic fixture; finite oracle. No LLM/human audit.',prompt_digest=None,sampling_digest=None,raw_response_digest=None,proposed_target=row['target'],reviewed_target=row['target'],final_target=row['target'],verification_profile=self.verify,outcome='verified',disposition='accepted',human_audit_ref=None,created_at=TIME,supersedes=None))
        return freeze(root,rows,proofs,self.registry,self.hooks,self.files(),dataset_id=dataset_id,seed=42,created_at=TIME,generator=dict(code_digest=BUILD,config_digest=self.generator['config_digest']),license_summary='Apache-2.0 synthetic derived research fixture',limitations=['Explicit legacy partition preserves original allocations. New format/digest; not historical bytes.','Public fixture, not sealed generalization or release support.'],purpose='research_fixture')
