"""Offline controlled Experiment 001; no teacher, release or serving authority."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import subprocess

from .contracts import loads
from .dataset import Freezer, split_for
from .journal import canonical, digest
from .oracle import DOMAINS, decide
from .tokenizer import train as train_tokenizer

CANONICAL_POLICY = ('Stop on a fatal policy violation. Otherwise retry a repeat-safe operation when the provider is healthy or its failure is transient, while attempts remain. If retry is not permitted, use an available allowed alternative. Stop if neither option is permitted.')

DATASET_ID = 'controlled-recovery-1000-v1'
# Explicit value-preserving English clauses, not model-generated paraphrases.
SUBJECTS = {
 'repeat_safe': ('Repeating the operation', 'Another execution', 'Running this operation again', 'A repeat of this operation'),
 'provider_state': ('The provider', 'The current provider', 'The service provider', 'This provider'),
 'retry_budget': ('The retry budget', 'The remaining attempt budget', 'The budget for retries', 'The available retry allowance'),
 'fallback_available': ('An alternative', 'A fallback provider', 'An alternative provider', 'A fallback option'),
 'fallback_permitted': ('Using the alternative', 'Switching to the fallback', 'Use of an alternative', 'Selecting the fallback'),
 'fatal_violation': ('A fatal policy violation', 'A fatal violation of policy', 'A policy violation classified as fatal', 'A fatal breach of policy'),
}
PREDICATES = {
 'repeat_safe': {False:'is unsafe.', True:'is safe.', 'unknown':'has unknown safety.'},
 'provider_state': {'healthy':'is healthy.', 'transient':'has a transient failure.', 'permanent':'has a permanent failure.', 'unknown':'has an unknown health state.'},
 'retry_budget': {'zero':'is exhausted.', 'positive':'allows at least one more attempt.', 'unknown':'is unknown.'},
 'fallback_available': {False:'is unavailable.', True:'is available.', 'unknown':'has unknown availability.'},
 'fallback_permitted': {False:'is forbidden.', True:'is permitted.', 'unknown':'has unknown permission.'},
 'fatal_violation': {False:'has not occurred.', True:'has occurred.', 'unknown':'may or may not have occurred.'},
}


def render(facts, template, contradictory=False):
    keys = list(DOMAINS)
    random.Random(100 + template).shuffle(keys)
    # Each family has a fixed phrasing/order skeleton. Never assign a new family
    # per row merely to evade template-family grouping.
    rng = random.Random(template)
    forms = {key: rng.randrange(4) for key in DOMAINS}
    clauses = [SUBJECTS[k][forms[k]] + ' ' + PREDICATES[k][facts[k]] for k in keys]
    if contradictory:
        clauses += ['At the same current time, one report confirms a fatal policy violation and another denies it. Neither report supersedes the other.']
    return ' '.join(clauses)


def generate(task):
    """1,000 balanced lessons; repeated scenarios and temporal variants stay in one family."""
    if (task['task_id'] != 'runtime.recovery.v1' or task['canonical_policy'] != CANONICAL_POLICY
            or [v['id'] for v in task['labels']] != ['retry', 'fallback', 'stop']):
        raise ValueError('controlled generator requires the canonical recovery TaskSpec')
    facts = [dict(zip(DOMAINS, values)) for values in product(*(v + ('unknown',) for v in DOMAINS.values()))]
    random.Random(42).shuffle(facts)
    source_hash = digest(Path(__file__).read_bytes())
    oracle_hash = digest(Path(__file__).with_name('oracle.py').read_bytes())
    records, provenance = [], {}
    groups = {label: [] for label in ('retry', 'fallback', 'stop', None)}
    for base, f in enumerate(facts):
        groups[decide(f).label].append(base)
    for index in range(1000):
        label = ('retry', 'fallback', 'stop', None)[index // 250]
        pool = groups[label]
        offset = index % 250
        base = pool[offset % len(pool)]
        variant = offset // len(pool) + 1
        f = facts[base]
        template = base % 40
        family = f'controlled-recovery-template-{template:02d}'
        sid = f'controlled-recovery-{index:04d}'
        conflict = index >= 950
        target = asdict(decide(f, contradictory=conflict))
        state = render(f, template, conflict)
        # Explicitly superseded old facts create meaningful temporal variants.
        # Values in the current observation remain the sole oracle inputs.
        if variant:
            prior = dict(f)
            for k, value in zip(DOMAINS, list(product(*DOMAINS.values()))[(base * 31 + variant * 17) % 96]):
                prior[k] = value
            state = ('Earlier observation: ' + render(prior, template) +
                     ' Current observation, which supersedes the earlier observation: ' + state)
        record = dict(schema_version='arbitrium.sample.v1', sample_id=sid,
            family_id=family, task_id=task['task_id'], split=split_for(family),
            input=dict(schema_version='arbitrium.request.v1', request_id=sid,
                task_id=task['task_id'], kind='choice', language='en', state=state,
                policy=task['canonical_policy'], question=task['question'],
                choices=[label['id'] for label in task['labels']]),
            target=target, concepts=['conditionals', 'evidence_sufficiency'] +
                (['contradiction'] if conflict else ['uncertainty'] if 'unknown' in f.values() else ['negation']) + (['temporal_order'] if variant else []),
            difficulty=3 if conflict else 2, provenance_id=sid+'.audit',
            review_status='verified', verification_kind='controlled_oracle', supersedes=None)
        records.append(record)
        provenance[record['provenance_id']] = dict(schema_version='arbitrium.provenance.v1',
            provenance_id=record['provenance_id'], sample_id=sid, scenario_id=f'finite-{base:04d}',
            template_id=family, facts=f, contradictory=conflict, temporal_variant=variant,
            source_kind='deterministic_controlled', permission_basis='Original synthetic examples authored for this repository at owner request.',
            source_content_sha256=digest(canonical({'facts':f, 'contradictory':conflict})),
            generator_sha256=source_hash, generator_seed=42, teacher=None, reviewer=None,
            non_model_reason='Deterministic value-preserving renderer and finite-state oracle; no LLM or human audit claimed.',
            verifier='finite-recovery-v1', verifier_sha256=oracle_hash,
            proposed_target=target, expected_outcome=target, final_label_source='finite_recovery_oracle',
            review_outcome='controlled_verified', human_audit=None, supersedes=None)
    return records, provenance


def freeze_sample(task_path, output):
    task = loads(Path(task_path).read_bytes())
    records, provenance = generate(task)
    return Freezer(output, DATASET_ID, task, fixture_only=True).freeze(
        records, provenance, curriculum_sha256=digest(Path(__file__).read_bytes()),
        generator_git_sha='see provenance generator_sha256',
        known_limitations=['Public research fixture, not a sealed benchmark or release dataset.',
            '250 examples per decision label and 250 unanswerable; 50 explicit contradictions. Repeated finite scenarios are grouped, not independent lessons.',
            '40 related controlled phrasing skeletons; broad natural-language generalization is untested.',
            'Temporal scope is explicit earlier/current override only. No live teacher, human semantic audit or production support-count gate.'])


def run(dataset, output, trainer):
    dataset, output = Path(dataset), Path(output)
    manifest_bytes = (dataset/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    task_bytes = (dataset/'task.json').read_bytes()
    if digest(task_bytes) != manifest['task_sha256']:
        raise ValueError('task hash mismatch')
    task = loads(task_bytes)
    # Deliberately read only train/dev, never calibration or test payloads.
    for split in ('train', 'dev'):
        if digest((dataset/f'{split}.jsonl').read_bytes()) != manifest['files'][f'{split}.jsonl']['sha256']:
            raise ValueError('split hash mismatch')
    if output.exists():
        raise FileExistsError('experiment output is immutable; choose a new directory')
    output.mkdir(parents=True)
    records = [loads(line, 128*1024) for line in (dataset/'train.jsonl').read_bytes().splitlines()]
    train_tokenizer(records, task, output/'tokenizer', vocab_size=2048)
    reports = []
    for seed in (42, 43, 44):
        destination = output/f'seed-{seed}'
        subprocess.run([str(Path(trainer).resolve()), str(dataset.resolve()),
            str((output/'tokenizer').resolve()), str(destination.resolve()), str(seed)], check=True)
        reports.append(json.loads((destination/'report.json').read_text()))
    summary = dict(format='experiment-001-summary-v1', research_only=True,
        dataset_digest=digest(manifest_bytes), seeds=[42,43,44], reports=reports,
        calibration='not fitted; calibration splits reserved', test='not read; public fixture is not sealed')
    (output/'summary.json').write_bytes(canonical(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--task', required=True)
    gen.add_argument('--output', required=True)
    cycle = sub.add_parser('run')
    cycle.add_argument('--dataset', required=True)
    cycle.add_argument('--output', required=True)
    cycle.add_argument('--trainer', required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        path, sha = freeze_sample(args.task, args.output)
        print(json.dumps({'path':str(path), 'dataset_digest':sha}))
    else:
        print(json.dumps(run(args.dataset, args.output, args.trainer), indent=2))


if __name__ == '__main__':
    main()
