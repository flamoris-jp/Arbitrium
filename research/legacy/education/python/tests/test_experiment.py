from collections import Counter
import json
from pathlib import Path

import pytest

from arbitrium_education.experiment import generate, render, run
from arbitrium_education.dataset import Freezer
from arbitrium_education.oracle import decide

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT/'examples/education/controlled-recovery-1000-v1'


def test_committed_sample_is_reproducible_and_balanced():
    task = json.loads((DATASET/'task.json').read_text())
    records, provenance = generate(task)
    stored = [json.loads(line) for p in DATASET.glob('*.jsonl')
              if p.stem in ('train', 'dev', 'calibration_fit', 'calibration_select', 'test')
              for line in p.read_text().splitlines()]
    assert sorted(stored, key=lambda r:r['sample_id']) == records
    audits = [json.loads(line) for line in (DATASET/'provenance-index.jsonl').read_text().splitlines()]
    assert {a['provenance_id']:a for a in audits} == provenance
    assert Counter(r['target']['label'] for r in records) == {'retry':250, 'fallback':250, 'stop':250, None:250}
    assert sum(a['contradictory'] for a in audits) == 50
    Freezer(DATASET.parent, 'test-validate', task, fixture_only=True).validate(records, provenance)
    scenario_splits = {}
    for r in records:
        audit = provenance[r['provenance_id']]
        expected = decide(audit['facts'], contradictory=audit['contradictory'])
        assert r['target'] == {'answerable':expected.answerable, 'label':expected.label}
        assert scenario_splits.setdefault(audit['scenario_id'], r['split']) == r['split']
    assert {r['split'] for r in records} == {'train','dev','calibration_fit','calibration_select','test'}


def test_healthy_retry_contract():
    task = json.loads((DATASET/'task.json').read_text())
    assert 'healthy or its failure is transient' in task['canonical_policy']
    facts = dict(repeat_safe=True, provider_state='healthy', retry_budget='positive',
        fallback_available=True, fallback_permitted=True, fatal_violation=False)
    assert decide(facts).label == 'retry'
    facts['fatal_violation'] = True
    assert decide(facts).label == 'stop'
    facts['fatal_violation'] = 'unknown'
    assert not decide(facts).answerable


def test_runner_rejects_changed_training_data(tmp_path):
    import shutil
    dataset = tmp_path/'data'
    shutil.copytree(DATASET, dataset)
    with (dataset/'train.jsonl').open('ab') as f:
        f.write(b'\n')
    with pytest.raises(ValueError, match='split hash mismatch'):
        run(dataset, tmp_path/'run', '/nonexistent/trainer')
    assert not (tmp_path/'run').exists()
