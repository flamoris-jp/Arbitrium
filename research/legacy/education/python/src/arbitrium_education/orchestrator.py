"""Adjudication retains original proposals. Model agreement never verifies prose."""
from dataclasses import dataclass, asdict
from .contracts import request as validate_request
from .oracle import decide
from .teacher import ReviewInput

@dataclass(frozen=True)
class Adjudication:
    sample_id: str
    candidate: object
    review: object
    oracle: object
    disposition: str

class EducationOrchestrator:
    def __init__(self,teacher,reviewer,journal):
        self.teacher=teacher;self.reviewer=reviewer;self.journal=journal

    def generate_and_review(self,spec):
        expected=decide(spec.facts)
        result=[]
        for i, candidate in enumerate(self.teacher.generate(spec)):
            sample_id=f'{spec.scenario_id}.{i}'
            request={'schema_version':'arbitrium.request.v1','request_id':sample_id,
                     'task_id':spec.task['task_id'],'kind':spec.task['kind'],'language':'en',
                     'state':candidate.state,'policy':spec.task['canonical_policy'],
                     'question':spec.task['question'],'choices':[x['id'] for x in spec.task['labels']]}
            validate_request(request,spec.task)
            review=self.reviewer.review(ReviewInput(request,spec.task,spec.facts))
            agrees=(review.decision=='accept' and not review.issues and review.extracted_facts==spec.facts
                    and review.target==expected.label and review.answerable==expected.answerable
                    and candidate.proposed_target==expected.label and candidate.proposed_answerable==expected.answerable)
            row=Adjudication(sample_id,candidate,review,expected,
                             'semantic_model_reviewed_pending_audit' if agrees else 'quarantine')
            self.journal.append({'event':'adjudicated',**asdict(row)})
            result.append(row)
        return result
