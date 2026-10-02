"""Balanced, train-only first lesson; no claim of held-out language ability."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import loads
from .current_curriculum import generate as complete_curriculum
from .dataset import Freezer, split_for
from .journal import digest

DATASET_ID = 'recovery-current-tiny-v1'
FAMILY_ID = 'simple-recovery-tiny-v1'


def generate(task):
    records, audits = complete_curriculum(task, 'diagnostic')
    retry = [r for r in records if r['target']['label'] == 'retry']
    fallback = [r for r in records if r['target']['label'] == 'fallback'][:8]
    # Eight direct counterexamples: changing only fatal_violation defeats retry.
    stop = []
    for positive in retry:
        facts = dict(audits[positive['provenance_id']]['facts'], fatal_violation=True)
        stop.append(next(r for r in records if audits[r['provenance_id']]['facts'] == facts))
    selected, provenance = [], {}
    generator_hash = digest(Path(__file__).read_bytes())
    for index, record in enumerate(retry + fallback + stop):
        parent_id = record['sample_id']
        audit = audits[record['provenance_id']]
        sid = f'current-tiny-{index:04d}'
        record.update(sample_id=sid, family_id=FAMILY_ID, split=split_for(FAMILY_ID),
                      provenance_id=sid+'.audit')
        record['input']['request_id'] = sid
        audit.update(provenance_id=record['provenance_id'], sample_id=sid,
                     template_id=FAMILY_ID, parent_dataset_id='recovery-current-diagnostic-v1',
                     parent_sample_id=parent_id, base_generator_sha256=audit['generator_sha256'],
                     generator_sha256=generator_hash)
        selected.append(record)
        provenance[record['provenance_id']] = audit
    if len(selected) != 24 or {r['split'] for r in selected} != {'train'}:
        raise ValueError('registered tiny curriculum must have 24 train-only records')
    return selected, provenance


def freeze(task_path, output):
    task = loads(Path(task_path).read_bytes())
    records, audits = generate(task)
    return Freezer(output, DATASET_ID, task, fixture_only=True).freeze(records, audits,
        parent_dataset_id='recovery-current-diagnostic-v1',
        curriculum_sha256=digest(Path(__file__).read_bytes()),
        generator_git_sha='see provenance generator_sha256', known_limitations=[
            'Public train-only memorization diagnostic; no generalization or release claim.',
            '8 retry, 8 fallback and 8 fatal-stop contrasts. Other stop cases and unknown facts are absent.',
            'Subset of the 96-state diagnostic. These datasets are not independent evaluations.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path, sha = freeze(args.task, args.output)
    print(f'{path}: {sha}')


if __name__ == '__main__':
    main()
