"""Deterministic split assignment and atomic frozen-dataset writer."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .contracts import sample as validate_sample, task as validate_task
from .journal import atomic_write, canonical, digest

SPLITS = ("train", "dev", "calibration_fit", "calibration_select", "test")


def split_for(family_id: str) -> str:
    raw = hashlib.sha256(("arbitrium-split-v1\n42\n" + family_id).encode()).digest()
    bucket = int.from_bytes(raw[:8], "big") % 10000
    if bucket < 6000:
        return "train"
    if bucket < 7500:
        return "dev"
    if bucket < 8500:
        return "calibration_fit"
    if bucket < 9000:
        return "calibration_select"
    return "test"


def _normalized_varying_input(record: dict) -> str:
    # Policy/question are fixed by TaskSpec and therefore excluded from dedup similarity.
    return " ".join(record["input"]["state"].casefold().split())


def _ngrams(value: str) -> set[tuple[str, ...]]:
    tokens = value.split()
    return {tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _file_entry(path: Path, records: int | None = None) -> dict:
    raw = path.read_bytes()
    result = {"sha256": digest(raw), "bytes": len(raw)}
    if records is not None:
        result["records"] = records
    return result


class DatasetError(ValueError):
    pass


class Freezer:
    def __init__(self, root: Path | str, dataset_id: str, task: dict, *, fixture_only=False):
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.task = validate_task(task)
        self.fixture_only = fixture_only
        if not dataset_id or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-" for c in dataset_id):
            raise DatasetError("invalid dataset ID")
        self.journal_path = self.root / ".freeze-journal" / f"{dataset_id}.json"

    def _created_at(self) -> str:
        if self.journal_path.exists():
            value = json.loads(self.journal_path.read_text())
            if set(value) != {"dataset_id", "created_at"} or value["dataset_id"] != self.dataset_id:
                raise DatasetError("invalid freeze journal")
            return value["created_at"]
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        atomic_write(self.journal_path, canonical({"dataset_id": self.dataset_id, "created_at": created}))
        return created

    def validate(self, records: list[dict], provenance: dict[str, dict]) -> None:
        ids: set[str] = set()
        normalized: dict[str, str] = {}
        grams: list[tuple[str, str, set]] = []
        families: dict[str, str] = {}
        for record in records:
            validate_sample(record, self.task)
            if record["sample_id"] in ids:
                raise DatasetError("duplicate sample ID")
            ids.add(record["sample_id"])
            expected = split_for(record["family_id"])
            if record["split"] != expected:
                raise DatasetError("split does not match frozen family hash")
            prior = families.setdefault(record["family_id"], record["split"])
            if prior != record["split"]:
                raise DatasetError("family crosses splits")
            value = _normalized_varying_input(record)
            if value in normalized:
                raise DatasetError("duplicate normalized input")
            normalized[value] = record["sample_id"]
            current = _ngrams(value)
            for other_id, other_family, other in grams:
                if other_family != record["family_id"] and _jaccard(current, other) >= 0.80:
                    raise DatasetError(f"near-duplicate families: {other_id} and {record['sample_id']}")
            grams.append((record["sample_id"], record["family_id"], current))
            audit = provenance.get(record["provenance_id"])
            if not audit or audit.get("sample_id") != record["sample_id"]:
                raise DatasetError("missing or mismatched provenance")
            if record["verification_kind"] == "audited_paraphrase" and audit.get("human_audit") != "accepted":
                raise DatasetError("paraphrase lacks accepted human audit")

    def freeze(self, records: list[dict], provenance: dict[str, dict], *,
               parent_dataset_id=None, curriculum_sha256="0" * 64,
               generator_git_sha="unknown", known_limitations=()) -> tuple[Path, str]:
        self.validate(records, provenance)
        destination = self.root / self.dataset_id
        if destination.exists():
            raise FileExistsError("dataset ID is immutable")
        created_at = self._created_at()
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.dataset_id}.", dir=self.root))
        try:
            task_raw = canonical(self.task)
            atomic_write(temporary / "task.json", task_raw)
            by_split = {name: [] for name in SPLITS}
            for record in sorted(records, key=lambda value: value["sample_id"]):
                by_split[record["split"]].append(record)
            for name, values in by_split.items():
                atomic_write(temporary / f"{name}.jsonl", b"".join(canonical(v) for v in values))
            families = [{"family_id": family, "split": split}
                        for family, split in sorted({r["family_id"]: r["split"] for r in records}.items())]
            atomic_write(temporary / "families.jsonl", b"".join(canonical(v) for v in families))
            referenced = sorted({r["provenance_id"] for r in records})
            provenance_values = [provenance[key] for key in referenced]
            atomic_write(temporary / "provenance-index.jsonl", b"".join(canonical(v) for v in provenance_values))

            files = {
                "task.json": _file_entry(temporary / "task.json"),
                "families.jsonl": _file_entry(temporary / "families.jsonl", len(families)),
                "provenance-index.jsonl": _file_entry(temporary / "provenance-index.jsonl", len(provenance_values)),
            }
            counts = {}
            for name, values in by_split.items():
                files[f"{name}.jsonl"] = _file_entry(temporary / f"{name}.jsonl", len(values))
                counts[name] = {
                    "records": len(values),
                    "answerable": Counter(str(v["target"]["answerable"]).lower() for v in values),
                    "labels": Counter(str(v["target"]["label"]) for v in values),
                    "families": len({v["family_id"] for v in values}),
                    "verification_kind": Counter(v["verification_kind"] for v in values),
                }
            manifest = {
                "schema_version": "arbitrium.dataset.v1",
                "dataset_id": self.dataset_id,
                "parent_dataset_id": parent_dataset_id,
                "task_id": self.task["task_id"],
                "task_sha256": digest(task_raw),
                "curriculum_sha256": curriculum_sha256,
                "created_at": created_at,
                "generator_git_sha": generator_git_sha,
                "split_algorithm": "arbitrium-split-v1",
                "split_seed": 42,
                "normalization_version": "state-casefold-space-v1",
                "dedup_version": "token-5gram-jaccard-v1",
                "files": files,
                "counts": counts,
                "provenance_index_sha256": files["provenance-index.jsonl"]["sha256"],
                "audits": {"paraphrases_require_human_acceptance": True},
                "license_summary": "See immutable provenance records.",
                "known_limitations": list(known_limitations),
                "fixture_only": self.fixture_only,
            }
            manifest_raw = canonical(manifest)
            atomic_write(temporary / "manifest.json", manifest_raw)
            os.replace(temporary, destination)
            parent_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return destination, digest(manifest_raw)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
