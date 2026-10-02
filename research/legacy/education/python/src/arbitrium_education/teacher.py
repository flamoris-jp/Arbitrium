from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .contracts import request as validate_request
from .oracle import validate_facts

CONCEPTS = ('negation','contrast','comparison','quantifiers','conditionals','temporal_order',
            'causal_evidence','uncertainty','contradiction','evidence_sufficiency')

@dataclass(frozen=True)
class CurriculumRequest:
    concept: str
    difficulty: int
    scenario_id: str
    facts: dict
    task: dict
    sample_count: int = 8

    def __post_init__(self):
        validate_facts(self.facts)
        if self.concept not in CONCEPTS or type(self.difficulty) is not int or not 1<=self.difficulty<=5:
            raise ValueError('invalid curriculum')
        if type(self.sample_count) is not int or not 1<=self.sample_count<=32:
            raise ValueError('invalid sample_count')

@dataclass(frozen=True)
class CandidateLesson:
    state: str
    proposed_target: str | None
    proposed_answerable: bool
    rationale: str = ''

@dataclass(frozen=True)
class ReviewInput:
    input: dict
    task: dict
    allowed_facts: dict

@dataclass(frozen=True)
class ReviewResult:
    decision: str
    target: str | None
    answerable: bool
    extracted_facts: dict
    issues: tuple[str, ...]
    notes: str

class TeacherProvider(Protocol):
    def generate(self, request: CurriculumRequest) -> list[CandidateLesson]: ...

class ReviewerProvider(Protocol):
    def review(self, request: ReviewInput) -> ReviewResult: ...
