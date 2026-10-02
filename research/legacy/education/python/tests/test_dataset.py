import copy
import json
from pathlib import Path

import pytest

from arbitrium_education.dataset import DatasetError, Freezer, split_for

ROOT = Path(__file__).resolve().parents[3]
TASK = json.loads((ROOT / "docs/examples/recovery-task.json").read_text())
BASE = json.loads((ROOT / "docs/examples/recovery-sample.json").read_text())


def record(sample_id, family_id, *, kind="controlled_oracle"):
    value = copy.deepcopy(BASE)
    value["sample_id"] = value["input"]["request_id"] = sample_id
    value["family_id"] = family_id
    value["split"] = split_for(family_id)
    value["provenance_id"] = "prov." + sample_id
    value["verification_kind"] = kind
    value["input"]["state"] += " Unique " + sample_id
    return value


def provenance(value, audit=None):
    result = {"schema_version": "arbitrium.provenance.v1",
              "provenance_id": value["provenance_id"], "sample_id": value["sample_id"]}
    if audit:
        result["human_audit"] = audit
    return result


def test_split_is_stable():
    assert split_for("family-a") == split_for("family-a")
    assert split_for("family-b") in {"train", "dev", "calibration_fit", "calibration_select", "test"}


def test_atomic_freeze_and_resume_identity(tmp_path, monkeypatch):
    values = [record("s1", "f1"), record("s2", "f2")]
    values[1]["input"]["state"] = "A wholly separate controlled fixture about permanent failure."
    audits = {v["provenance_id"]: provenance(v) for v in values}
    freezer = Freezer(tmp_path, "dataset-1", TASK, fixture_only=True)
    created = freezer._created_at()
    assert freezer._created_at() == created
    path, identity = freezer.freeze(values, audits, known_limitations=("fixture",))
    assert identity == __import__("hashlib").sha256((path / "manifest.json").read_bytes()).hexdigest()
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["dataset_id"] == "dataset-1" and manifest["fixture_only"] is True
    assert manifest["created_at"] == created
    with pytest.raises(FileExistsError):
        freezer.freeze(values, audits)


def test_rejects_bad_split_duplicate_and_missing_audit(tmp_path):
    value = record("s1", "f1")
    audit = {value["provenance_id"]: provenance(value)}
    bad = copy.deepcopy(value)
    bad["split"] = next(name for name in ("train", "dev") if name != value["split"])
    with pytest.raises(DatasetError):
        Freezer(tmp_path, "bad", TASK).validate([bad], audit)
    with pytest.raises(DatasetError):
        Freezer(tmp_path, "bad", TASK).validate([value, copy.deepcopy(value)], audit)
    paraphrase = record("s2", "f2", kind="audited_paraphrase")
    with pytest.raises(DatasetError):
        Freezer(tmp_path, "bad", TASK).validate([paraphrase],
            {paraphrase["provenance_id"]: provenance(paraphrase)})


def test_near_duplicate_different_families_blocked(tmp_path):
    one = record("s1", "f1")
    two = record("s2", "f2")
    one["input"]["state"] = "one two three four five six seven eight nine ten"
    two["input"]["state"] = "one two three four five six seven eight nine ten eleven"
    audits = {v["provenance_id"]: provenance(v) for v in (one, two)}
    with pytest.raises(DatasetError):
        Freezer(tmp_path, "bad", TASK).validate([one, two], audits)
