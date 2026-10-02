"""Current-state curriculum and explicit memorization/generalization experiments.

The historical 1,000-record fixture and its generator remain immutable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import subprocess

from .contracts import loads, task as validate_task
from .dataset import Freezer, split_for
from .experiment import CANONICAL_POLICY, render as grouped_render
from .journal import canonical, digest
from .oracle import DOMAINS, decide, render as simple_render
from .tokenizer import train as train_tokenizer

DATASETS = {'diagnostic': 'recovery-current-diagnostic-v1',
            'grouped': 'recovery-current-grouped-v1'}
DIAGNOSTIC_FAMILY = 'simple-recovery-diagnostic-v1'


def generate(task, kind):
    validate_task(task)
    if (task['task_id'] != 'runtime.recovery.v1' or task['canonical_policy'] != CANONICAL_POLICY
            or [v['id'] for v in task['labels']] != ['retry', 'fallback', 'stop']):
        raise ValueError('canonical recovery TaskSpec required')
    if kind not in DATASETS:
        raise ValueError('unknown curriculum')
    if kind == 'diagnostic':
        values = list(product(*DOMAINS.values()))
    else:
        values = list(product(*(v + ('unknown',) for v in DOMAINS.values())))
    rng = random.Random(73)
    rng.shuffle(values)
    selected = list(enumerate(values))
    if kind == 'grouped':
        # 18 distinct current fact combinations per outcome, no replicated rows.
        groups = {label: [] for label in ('retry', 'fallback', 'stop', None)}
        for i, v in selected:
            groups[decide(dict(zip(DOMAINS, v))).label].append((i, v))
        selected = [row for label in groups for row in groups[label][:18]]
    code_hash = digest(Path(__file__).read_bytes())
    oracle_hash = digest(Path(__file__).with_name('oracle.py').read_bytes())
    renderer_hash = digest(Path(__file__).with_name('experiment.py').read_bytes())
    records, provenance = [], {}
    for index, (scenario, values) in enumerate(selected):
        facts = dict(zip(DOMAINS, values))
        template = scenario % 24
        family = DIAGNOSTIC_FAMILY if kind == 'diagnostic' else f'simple-current-template-{template:02d}'
        state = simple_render(facts) if kind == 'diagnostic' else grouped_render(facts, template)
        sid = f'current-{kind}-{index:04d}'
        target = asdict(decide(facts))
        record = dict(schema_version='arbitrium.sample.v1', sample_id=sid,
            family_id=family, task_id=task['task_id'], split=split_for(family),
            input=dict(schema_version='arbitrium.request.v1', request_id=sid,
                task_id=task['task_id'], kind='choice', language='en', state=state,
                policy=task['canonical_policy'], question=task['question'],
                choices=[v['id'] for v in task['labels']]),
            target=target, concepts=['conditionals','negation'] +
                (['uncertainty','evidence_sufficiency'] if 'unknown' in facts.values() else []),
            difficulty=1 if kind == 'diagnostic' else 2, provenance_id=sid+'.audit',
            review_status='verified', verification_kind='controlled_oracle', supersedes=None)
        records.append(record)
        provenance[record['provenance_id']] = dict(schema_version='arbitrium.provenance.v1',
            provenance_id=record['provenance_id'], sample_id=sid,
            scenario_id=f'current-{kind}-scenario-{scenario:04d}', template_id=family,
            facts=facts, source_kind='deterministic_controlled',
            permission_basis='Original synthetic current-state curriculum requested by repository owner.',
            source_content_sha256=digest(canonical(facts)), generator_sha256=code_hash,
            renderer_sha256=oracle_hash if kind == 'diagnostic' else renderer_hash,
            verifier_sha256=oracle_hash, generator_seed=73,
            teacher=None, reviewer=None, human_audit=None,
            non_model_reason='Value-preserving controlled clauses; finite-state oracle; no LLM or human audit claimed.',
            proposed_target=target, expected_outcome=target, final_label_source='finite_recovery_oracle',
            review_outcome='controlled_verified', supersedes=None)
    return records, provenance


def freeze(task_path, output):
    task = loads(Path(task_path).read_bytes())
    paths = {}
    for kind, dataset_id in DATASETS.items():
        records, audits = generate(task, kind)
        if kind == 'diagnostic' and {r['split'] for r in records} != {'train'}:
            raise ValueError('the registered diagnostic family must be train-only')
        path, sha = Freezer(output, dataset_id, task, fixture_only=True).freeze(records, audits,
            curriculum_sha256=digest(Path(__file__).read_bytes()),
            generator_git_sha='see provenance generator_sha256',
            known_limitations=['Public research fixture; no sealed test or release support.',
                'Current facts only. No historical overrides or contradictions.',
                'Diagnostic uses training data for selection; this is not generalization evidence.'
                    if kind == 'diagnostic' else
                'Small family-heldout dev has low per-label support. Train a fresh model; no diagnostic weights are reused.'])
        paths[kind] = {'path':str(path), 'dataset_digest':sha}
    return paths


def run(dataset, output, trainer, config_path, seeds=(42,43,44)):
    dataset, output = Path(dataset), Path(output)
    config_bytes = Path(config_path).read_bytes()
    config = loads(config_bytes)
    if config.get('selection_scope') not in ('train_diagnostic', 'dev'):
        raise ValueError('explicit research selection scope required')
    manifest_bytes = (dataset/'manifest.json').read_bytes()
    manifest = loads(manifest_bytes)
    task_bytes = (dataset/'task.json').read_bytes()
    if digest(task_bytes) != manifest['task_sha256']:
        raise ValueError('task hash mismatch')
    required = ('train',) if config['selection_scope']=='train_diagnostic' else ('train','dev')
    for split in required:
        raw = (dataset/f'{split}.jsonl').read_bytes()
        if digest(raw) != manifest['files'][f'{split}.jsonl']['sha256']:
            raise ValueError('split hash mismatch')
        if not raw:
            raise ValueError('empty required split')
    if output.exists():
        raise FileExistsError('experiment output is immutable')
    output.mkdir(parents=True)
    rows = [loads(line,128*1024) for line in (dataset/'train.jsonl').read_bytes().splitlines()]
    train_tokenizer(rows, loads(task_bytes), output/'tokenizer', vocab_size=2048)
    (output/'resolved-research-config.json').write_bytes(config_bytes)
    reports = []
    for seed in seeds:
        target = output/f'seed-{seed}'
        subprocess.run([str(Path(trainer).resolve()), str(dataset.resolve()),
            str((output/'tokenizer').resolve()), str(target.resolve()), str(seed),
            str((output/'resolved-research-config.json').resolve())], check=True)
        reports.append(loads((target/'report.json').read_bytes(),1024*1024))
    summary = dict(format='experiment-002-summary-v1', research_only=True,
        dataset_digest=digest(manifest_bytes), research_config=config,
        config_sha256=digest(config_bytes), trainer_sha256=digest(Path(trainer).read_bytes()),
        seeds=list(seeds), reports=reports,
        calibration='not fitted; raw Brier/ECE only', test='not read; public fixture is not sealed')
    (output/'summary.json').write_bytes(canonical(summary))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    gen=sub.add_parser('generate')
    gen.add_argument('--task',required=True)
    gen.add_argument('--output',required=True)
    trial=sub.add_parser('run')
    trial.add_argument('--dataset',required=True)
    trial.add_argument('--output',required=True)
    trial.add_argument('--trainer',required=True)
    trial.add_argument('--config',required=True)
    trial.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=parser.parse_args()
    if args.command=='generate':
        print(json.dumps(freeze(args.task,args.output),indent=2))
    else:
        print(json.dumps(run(args.dataset,args.output,args.trainer,args.config,args.seeds),indent=2))


if __name__=='__main__':
    main()
