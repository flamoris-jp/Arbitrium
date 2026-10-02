"""SentencePiece training and reference input assembly for parity fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sentencepiece as spm

from .contracts import request as validate_request, task as validate_task
from .journal import atomic_write, canonical, digest

CONTROL_NAMES = ("CLS", "TASK", "POLICY", "STATE", "QUESTION")
CONTROL_SYMBOLS = tuple(f"<{name}>" for name in CONTROL_NAMES)


def corpus_lines(records: list[dict], task: dict) -> list[str]:
    validate_task(task)
    result: list[str] = []
    for record in sorted(records, key=lambda value: value["sample_id"]):
        if record["split"] != "train":
            raise ValueError("tokenizer corpus must contain train records only")
        request = validate_request(record["input"], task)
        result.extend((request["task_id"], request["policy"], request["state"], request["question"]))
    if not result:
        raise ValueError("empty tokenizer corpus")
    return result


def train(records: list[dict], task: dict, output: Path | str, *, vocab_size=8192) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = corpus_lines(records, task)
    corpus = "".join(line + "\n" for line in lines).encode("utf-8")
    corpus_path = output / "train-fields.txt"
    atomic_write(corpus_path, corpus)
    prefix = output / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        hard_vocab_limit=False,
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_whitespace=True,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        num_threads=1,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        control_symbols=list(CONTROL_SYMBOLS),
    )
    model_path = output / "tokenizer.model"
    Path(str(prefix) + ".model").replace(model_path)
    Path(str(prefix) + ".vocab").unlink()
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    controls = {name: processor.piece_to_id(symbol)
                for name, symbol in zip(CONTROL_NAMES, CONTROL_SYMBOLS)}
    if processor.pad_id() != 0 or processor.unk_id() != 1 or processor.bos_id() != 2 or processor.eos_id() != 3:
        raise RuntimeError("special token IDs do not match v1")
    if len(set(controls.values())) != len(controls) or any(value < 0 for value in controls.values()):
        raise RuntimeError("missing or colliding controls")
    options = {
        "model_type": "bpe", "requested_vocab_size": vocab_size,
        "hard_vocab_limit": False, "character_coverage": 1.0,
        "byte_fallback": True, "normalization_rule_name": "identity",
        "add_dummy_prefix": False, "remove_extra_whitespaces": False,
        "split_by_whitespace": True, "shuffle_input_sentence": False,
        "input_sentence_size": 0, "num_threads": 1,
    }
    config = {
        "schema_version": "arbitrium.tokenizer.v1",
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "assembly_version": task["assembly_version"],
        "corpus_sha256": digest(corpus),
        "model_sha256": digest(model_path.read_bytes()),
        "actual_vocab_size": processor.vocab_size(),
        "special_ids": {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3} | controls,
        "trainer_options": options,
    }
    atomic_write(output / "tokenizer-config.json", canonical(config))
    return config


class Tokenizer:
    def __init__(self, model_path: Path | str, config_path: Path | str):
        self.model_path = Path(model_path)
        self.config = json.loads(Path(config_path).read_text())
        if set(self.config) != {"schema_version", "sentencepiece_version", "assembly_version",
                               "corpus_sha256", "model_sha256", "actual_vocab_size",
                               "special_ids", "trainer_options"}:
            raise ValueError("unexpected tokenizer config fields")
        if self.config["schema_version"] != "arbitrium.tokenizer.v1" or self.config["assembly_version"] != "fields.v1":
            raise ValueError("unsupported tokenizer config")
        if digest(self.model_path.read_bytes()) != self.config["model_sha256"]:
            raise ValueError("tokenizer hash mismatch")
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        if self.processor.vocab_size() != self.config["actual_vocab_size"]:
            raise ValueError("tokenizer vocabulary mismatch")
        for name, value in self.config["special_ids"].items():
            actual = {"PAD": self.processor.pad_id(), "UNK": self.processor.unk_id(),
                      "BOS": self.processor.bos_id(), "EOS": self.processor.eos_id()}.get(name)
            if actual is None:
                actual = self.processor.piece_to_id(f"<{name}>")
            if actual != value:
                raise ValueError("tokenizer control mismatch")

    def assemble(self, request: dict, task: dict) -> tuple[list[int], list[int]]:
        request = validate_request(request, task)
        ids = self.config["special_ids"]
        token_ids = [ids["CLS"], ids["TASK"]]
        segments = [0, 0]
        fields = ((request["task_id"], 0), (None, 1), (request["policy"], 1),
                  (None, 2), (request["state"], 2), (None, 3),
                  (request["question"], 3))
        controls = iter((ids["POLICY"], ids["STATE"], ids["QUESTION"]))
        for value, segment in fields:
            pieces = [next(controls)] if value is None else self.processor.encode(value, out_type=int,
                                                                                   enable_sampling=False)
            token_ids.extend(pieces)
            segments.extend([segment] * len(pieces))
        token_ids.append(ids["EOS"])
        segments.append(3)
        if len(token_ids) > task["max_sequence_length"]:
            raise ValueError("input_too_long")
        if 0 in token_ids:
            raise ValueError("PAD appeared in real input")
        return token_ids, segments

    def cache_key(self, request: dict, task: dict) -> str:
        validate_request(request, task)
        material = self.model_path.read_bytes() + task["assembly_version"].encode() + canonical(request)
        return hashlib.sha256(material).hexdigest()
