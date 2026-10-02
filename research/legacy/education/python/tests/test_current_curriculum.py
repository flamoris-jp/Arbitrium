from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from arbitrium_education import current_curriculum, tiny_curriculum
from arbitrium_education.dataset import Freezer, SPLITS
from arbitrium_education.journal import digest
from arbitrium_education.oracle import decide

ROOT = Path(__file__).resolve().parents[3]
SAMPLES = ROOT/'examples/education'
TASK = json.loads((ROOT/'docs/examples/recovery-task.json').read_text())


@pytest.mark.parametrize('kind,count', [('diagnostic',96), ('grouped',72), ('tiny',24)])
def test_frozen_current_curricula_are_exactly_reproducible(kind, count):
    records, audits = (tiny_curriculum.generate(TASK) if kind == 'tiny' else
                      current_curriculum.generate(TASK, kind))
    dataset = SAMPLES/(tiny_curriculum.DATASET_ID if kind == 'tiny' else
                       current_curriculum.DATASETS[kind])
    stored = [json.loads(line) for split in SPLITS
              for line in (dataset/f'{split}.jsonl').read_text().splitlines()]
    assert len(stored) == count
    assert sorted(stored, key=lambda r:r['sample_id']) == records
    stored_audits = [json.loads(line) for line in
                     (dataset/'provenance-index.jsonl').read_text().splitlines()]
    assert {a['provenance_id']:a for a in stored_audits} == audits
    Freezer(dataset.parent, 'validate-current', TASK, fixture_only=True).validate(records, audits)
    manifest = json.loads((dataset/'manifest.json').read_text())
    for filename, entry in manifest['files'].items():
        assert digest((dataset/filename).read_bytes()) == entry['sha256']
    # Each current fact combination appears once: no history/replicas inflate support.
    assert len({json.dumps(a['facts'],sort_keys=True) for a in audits.values()}) == count
    for r in records:
        assert r['target'] == asdict(decide(audits[r['provenance_id']]['facts']))
        assert 'Earlier observation' not in r['input']['state']
        assert 'supersedes' not in r['input']['state']
    if kind == 'grouped':
        assert Counter(r['target']['label'] for r in records) == {
            'retry':18, 'fallback':18, 'stop':18, None:18}
        assert {r['target']['label'] for r in records if r['split']=='dev'} == {
            'retry','fallback','stop',None}
    else:
        assert {r['split'] for r in records} == {'train'}


def test_tiny_curriculum_balances_labels_and_contains_fatal_minimal_pairs():
    records, audits = tiny_curriculum.generate(TASK)
    assert Counter(r['target']['label'] for r in records) == {'retry':8,'fallback':8,'stop':8}
    by_facts = {json.dumps(audits[r['provenance_id']]['facts'],sort_keys=True):r for r in records}
    for r in records:
        if r['target']['label']=='retry':
            changed = dict(audits[r['provenance_id']]['facts'],fatal_violation=True)
            opposite = by_facts[json.dumps(changed,sort_keys=True)]
            assert opposite['target'] == {'answerable':True,'label':'stop'}
            assert opposite['family_id'] == r['family_id']
            assert opposite['input']['state'] != r['input']['state']


def test_diagnostic_runner_does_not_read_reserved_payloads(tmp_path, monkeypatch):
    dataset = tmp_path/'data'
    shutil.copytree(SAMPLES/tiny_curriculum.DATASET_ID, dataset)
    for split in SPLITS[1:]:
        (dataset/f'{split}.jsonl').unlink()
    def tokenizer(records, task, output, vocab_size):
        assert len(records)==24
        raise RuntimeError('reached train-only tokenizer')
    monkeypatch.setattr(current_curriculum, 'train_tokenizer', tokenizer)
    with pytest.raises(RuntimeError, match='reached train-only tokenizer'):
        current_curriculum.run(dataset, tmp_path/'run', '/nonexistent/trainer',
                               ROOT/'configs/experiments/current-tiny.json')


def test_current_runner_rejects_corrupt_train_before_creating_output(tmp_path):
    dataset = tmp_path/'data'
    shutil.copytree(SAMPLES/tiny_curriculum.DATASET_ID, dataset)
    with (dataset/'train.jsonl').open('ab') as file:
        file.write(b'\n')
    with pytest.raises(ValueError, match='split hash mismatch'):
        current_curriculum.run(dataset, tmp_path/'run', '/nonexistent/trainer',
                               ROOT/'configs/experiments/current-tiny.json')
    assert not (tmp_path/'run').exists()
