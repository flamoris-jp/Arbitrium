"""Finite preregistered offline public-architecture comparison; no promotion."""
import argparse,copy,json,os,resource,subprocess,sys
from pathlib import Path
from source_identity import ROOT
sys.path.insert(0,str(ROOT/'python'))
from arbitrium.composition import Decision,BUILD,IDENTITY,ref,component,TIME
from arbitrium.legacy import SPLITS,ARCHIVE
from arbitrium.metrics import diagnostic,reduce_predictions
from arbitrium.provider import DecisionProvider
from maidionis_education.contracts import canonical,digest,loads
from maidionis_education.datasets import validate_dataset,evaluation_data
from maidionis_education.artifacts import export_components,export_candidate,finalize,component_digest
from maidionis_education.evaluation import evaluate,EvaluationLedger

CAP=2*2**30
POLICY=digest(b'Arbitrium migration comparison v1; complete diagnostic accounting; every artifact is research_only; no quality approval.\n')
METRIC=component('metrics',dict(reducer='decision-diagnostic.v1',labels=['retry','fallback','stop'],answerability_threshold_milli=500))
CARD=dict(license='Apache-2.0',intended_use='Offline migration research comparison',limitations=['Public synthetic fixtures, not sealed; changed codec/init/optimizer/order/toolchain.','Absent calibration; no production readiness or execution authority.'])
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(CAP,CAP));resource.setrlimit(resource.RLIMIT_CPU,(300,300))
def run(driver,*args,success=True):
    p=subprocess.run([str(driver),*map(str,args)],capture_output=True,timeout=330,preexec_fn=limits)
    if not success:
        if p.returncode==0:raise AssertionError('expected rejection')
        return p
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
    if len(p.stdout)>4*2**20:raise ValueError('native output bound')
    return loads(p.stdout,4*2**20)
def configs():
    result=[]
    for name,file in [('recovery-current-tiny-v1','current-tiny.json'),('recovery-current-diagnostic-v1','current-diagnostic.json'),('recovery-current-grouped-v1','current-generalization.json')]:
        c=loads((ARCHIVE/'configs/experiments'/file).read_bytes());result.append((name,c))
    result.append(('controlled-recovery-1000-v1',dict(schema_version='arbitrium.research-training.v1',selection_scope='dev',epochs=20,learning_rate=.0003,batch_size=64,patience=5,weight_decay=.01)))
    return result

def plan():
    return dict(format='arbitrium.public-comparison-plan.v1',seeds=[42,43,44],runs=[dict(dataset=n,legacy_manifest=Decision(n).legacy_digest,config=c) for n,c in configs()],
        evaluation_policy_digest=POLICY,split_seed=42,scope='Exact legacy inputs/targets/family allocations; changed neutral format and numerical implementation.',
        differences=['Train-only ASCII word tokenizer replaces SentencePiece; fixed fields/control tokens retained.','Core seeded default initialization, one joint four-output layer, explicit module decay roles, SHA-256 epoch ordering and linear-decay schedule replace legacy implementations.','LibTorch 2.5.1+cpu replaces recorded 2.14.0+cpu. No bitwise or score equality claimed.'],
        selection_tie='first strict improvement',all_seeds_required=True,production_promotion=False)

def one(driver,d,data,dh,rows,root,config):
    root.mkdir();(root/'config.json').write_bytes(canonical(config))
    trained=run(driver,'train',data,dh,root/'checkpoints',root/'trained',0,0,root/'config.json','none')
    if not trained['parameters_changed'] or not trained['gradient_l1']>0:raise AssertionError('real training required')
    components=export_components(d.files(),root/'trained');scope=config['selection_scope'];split='train' if scope=='train_diagnostic' else 'dev'
    registration=dict(schema_version='maidionis.evaluation-registration.v1',id='arbitrium:evaluation:'+str(config['seed']),version='1',experiment_id='arbitrium:public:v1',policy_digest=POLICY,
        descriptor_digest=digest(canonical(d.descriptor)),evaluated_component_digest=component_digest(components),dataset_digest=dh,split=split,selection_scope=scope,metrics=[METRIC],baselines=[],calibration='not_applicable',minimum_samples=1,tolerance=1e-6,environment_digest=trained['environment_digest'])
    m=loads((data/'manifest.json').read_bytes(),4*2**20)
    edu=dict(schema_version='maidionis.education-plan.v1',experiment_id=registration['experiment_id'],descriptor_digest=registration['descriptor_digest'],hook_identity=d.hook,native_build_digest=BUILD,curriculum_digest=d.legacy_digest,prompt_digest=POLICY,providers=[d.hook,d.hook],verification_profile=d.verify,dataset_references=[dict(id=m['dataset_id'],digest=dh)],training_config_digest=digest(canonical(config)),selection_config_digest=digest(canonical(dict(scope=scope,tie='first_strict_improvement'))),calibration_config_digest=None,evaluation_policy_digest=POLICY,max_cycles=1,max_attempts=1,max_examples=len(rows),max_elapsed_seconds=300,max_output_bytes=4*2**20)
    ledger=EvaluationLedger(root/'evaluation-ledger',edu);ledger.register(registration,edu);ledger.admit(registration,'Public fixture migration comparison, no sealed test access.')
    candidate=root/'candidate';cd=export_candidate(candidate,components,root/'trained',registration,artifact_id='arbitrium:candidate:'+str(config['seed']),created_at=TIME,model_card=CARD)
    requests=[dict(schema_version='maidionis.request.v1',request_id=r['sample_id'],**IDENTITY,payload_schema=ref('input'),payload=r['input'],context_ref=None) for r in rows if r['split']==split]
    # Keep transient inference framing outside the immutable dataset.
    composition=root/'composition';composition.mkdir()
    for p,b in d.files().items():(composition/p).write_bytes(b)
    (composition/'inference-inputs.json').write_bytes(canonical(requests))
    before=run(driver,'validate',composition,candidate,cd,'offline_evaluation')
    if before['model_constructions']!=0:raise AssertionError('metadata validation allocation')
    loaded=run(driver,'infer',composition,candidate,cd,64*2**20,CAP,32*2**20,CAP,'none')
    # Validate the actual admitted native envelopes across the separate provider
    # boundary after the disposable host has exited; this grants no holder.
    host_results={p['sample_id']:p['result'] for p in loaded['predictions']}
    provider=DecisionProvider(d,cd,lambda request:host_results[request['request_id']])
    for request in requests:provider.infer(request)
    selected=[r for r in rows if r['split']==split];native_logits=trained['best_logits'] if split=='train' else trained['dev_logits']
    if len(loaded['predictions'])!=len(selected):raise AssertionError('inference coverage')
    for x,y in zip(loaded['predictions'],native_logits):
        if max(abs(a-b) for a,b in zip(x['raw_output'],y))>1e-6:raise AssertionError('selected-best loaded parity')
    predictions=[];rh=digest(canonical(registration));run_id='arbitrium:seed:'+str(config['seed'])
    for row,p in zip(selected,loaded['predictions']):
        predictions.append(dict(schema_version='maidionis.prediction.v1',experiment_id=registration['experiment_id'],run_id=run_id,sample_id=row['sample_id'],dataset_id=row['dataset_id'],dataset_digest=dh,split=split,descriptor_digest=registration['descriptor_digest'],artifact_digest=cd,evaluated_component_digest=registration['evaluated_component_digest'],evaluation_registration_digest=rh,selection_scope=scope,calibration_status='uncalibrated',result=p['result'],target=row['target'],raw_output_schema=ref('raw'),raw_output=p['raw_output'],timing=dict(elapsed_ns=0,profile='offline.mechanics'),error_status=None))
    sealed=evaluation_data(data,dh,d.registry,d.hooks,split=split)
    kw=dict(run_id=run_id,artifact_digest=cd,reducers={(METRIC['id'],METRIC['version'],METRIC['config_digest']):reduce_predictions},passing_policy=lambda report:False)
    report,summary=evaluate(sealed,predictions,registration,d.registry,**kw)
    if report['status']!='complete' or summary['passing']:raise AssertionError('research complete accounting')
    bad,_=evaluate(sealed,predictions[:-1],registration,d.registry,**kw)
    if bad['status']!='invalid_run':raise AssertionError('missing prediction rejection')
    fd=finalize(candidate,cd,root/'final',report,summary,artifact_id='arbitrium:research:'+str(config['seed']),created_at=TIME,model_card=CARD)
    final_loaded=run(driver,'infer',composition,root/'final',fd,64*2**20,CAP,32*2**20,CAP,'none')
    if [p['raw_output'] for p in final_loaded['predictions']]!=[p['raw_output'] for p in loaded['predictions']]:raise AssertionError('final reload parity')
    run(driver,'validate',composition,candidate,cd,'serving',success=False)
    trainrows=[r for r in rows if r['split']=='train'];devrows=[r for r in rows if r['split']=='dev']
    out=dict(seed=config['seed'],config=config,build_digest=BUILD,descriptor_digest=registration['descriptor_digest'],legacy_dataset_digest=d.legacy_digest,derived_dataset_digest=dh,environment=trained['environment'],environment_digest=trained['environment_digest'],codec_digest=d.descriptor['input_codec']['config_digest'],best_epoch=trained['state']['best_epoch'],history=trained['state']['history'],gradient_l1=trained['gradient_l1'],parameters_changed=trained['parameters_changed'],train=diagnostic(trainrows,trained['best_logits']),dev=diagnostic(devrows,trained['dev_logits']) if devrows else None,evaluation_report=report,evaluation_summary=summary,final_artifact_digest=fd,candidate_digest=cd,load_receipt=loaded['receipt'],selected_best_loaded_parity=True,missing_prediction_rejected=True,pending_serving_rejected=True)
    (root/'report.json').write_bytes(canonical(out));return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',default='build/arbitrium_driver');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    driver=Path(a.driver).resolve();root=Path(a.output).resolve()
    if root.exists():raise FileExistsError('immutable experiment output')
    root.mkdir(parents=True);pr=plan();(root/'plan.json').write_bytes(canonical(pr))
    if run(driver,'build-identity')['build_digest']!=BUILD:raise ValueError('native/Python build identity mismatch')
    reports=[]
    for name,config in configs():
        d=Decision(name);data=root/name;dh=d.freeze(data);_,rows=d.validate(data,dh)
        for seed in ([42] if a.smoke else pr['seeds']):
            cfg=dict(config,seed=seed)
            if a.smoke:cfg.update(epochs=2,patience=2)
            report=one(driver,d,data,dh,rows,root/(name+'-seed-'+str(seed)),cfg);reports.append(dict(dataset=name,report=report))
            print(name,seed,'train',report['train']['decision_accuracy'],'dev',None if report['dev'] is None else report['dev']['decision_accuracy'],flush=True)
        if a.smoke:break
    (root/'results.json').write_bytes(canonical(dict(format='arbitrium.public-comparison-results.v1',plan_digest=digest(canonical(pr)),plan=pr,smoke=a.smoke,reports=reports)))
if __name__=='__main__':main()
