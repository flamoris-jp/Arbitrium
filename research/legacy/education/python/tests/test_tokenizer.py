import copy
import json
from pathlib import Path

import pytest

from arbitrium_education.dataset import split_for
from arbitrium_education.tokenizer import Tokenizer, train

ROOT = Path(__file__).resolve().parents[3]
TASK = json.loads((ROOT / "docs/examples/recovery-task.json").read_text())
BASE = json.loads((ROOT / "docs/examples/recovery-sample.json").read_text())


def train_record(number):
    value = copy.deepcopy(BASE)
    value["sample_id"] = value["input"]["request_id"] = f"tok.{number}"
    family = next(f"train-family-{number}-{i}" for i in range(1000)
                  if split_for(f"train-family-{number}-{i}") == "train")
    value["family_id"] = family
    value["split"] = "train"
    value["input"]["state"] = (
        f"Attempt {number}. Repeating isn't safe; cost is {number}.25.\n"
        "Literal <STATE> is data — fallback may be allowed. café ∀ x ≤ 3."
    )
    return value


def test_train_and_assemble_is_deterministic(tmp_path):
    records = [train_record(i) for i in range(30)]
    config = train(records, TASK, tmp_path, vocab_size=512)
    assert 300 < config["actual_vocab_size"] <= 512
    tokenizer = Tokenizer(tmp_path / "tokenizer.model", tmp_path / "tokenizer-config.json")
    request = records[0]["input"]
    first = tokenizer.assemble(request, TASK)
    second = tokenizer.assemble(request, TASK)
    assert first == second and len(first[0]) == len(first[1]) <= 256
    assert first[0][0] == config["special_ids"]["CLS"]
    # A literal spelling inside state must not turn into an inserted delimiter.
    state_control = config["special_ids"]["STATE"]
    assert first[0].count(state_control) == 1
    changed = copy.deepcopy(request)
    changed["state"] += " changed"
    assert tokenizer.cache_key(request, TASK) != tokenizer.cache_key(changed, TASK)


def test_train_rejects_nontrain_and_load_rejects_tamper(tmp_path):
    value = train_record(1)
    value["split"] = "dev"
    with pytest.raises(ValueError):
        train([value], TASK, tmp_path, vocab_size=512)
    value["split"] = "train"
    train([value], TASK, tmp_path, vocab_size=512)
    (tmp_path / "tokenizer.model").write_bytes((tmp_path / "tokenizer.model").read_bytes() + b"x")
    with pytest.raises(ValueError):
        Tokenizer(tmp_path / "tokenizer.model", tmp_path / "tokenizer-config.json")
