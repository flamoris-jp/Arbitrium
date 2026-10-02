"""Finite recovery oracle from education-system.md; never used for text inference.

SPEC-001 resolved: TaskSpec and oracle both allow healthy or transient providers.
"""
from dataclasses import dataclass, asdict
from itertools import product

DOMAINS = {
    'repeat_safe': (False, True),
    'provider_state': ('healthy', 'transient', 'permanent'),
    'retry_budget': ('zero', 'positive'),
    'fallback_available': (False, True),
    'fallback_permitted': (False, True),
    'fatal_violation': (False, True),
}


@dataclass(frozen=True)
class Target:
    answerable: bool
    label: str | None


def validate_facts(facts):
    if type(facts) is not dict or set(facts) != set(DOMAINS):
        raise ValueError('exact structured fact set required')
    for key, domain in DOMAINS.items():
        v = facts[key]
        if v == 'unknown':
            continue
        if not any(type(v) is type(x) and v == x for x in domain):
            raise ValueError(f'invalid fact: {key}')
    return facts


def _complete(f):
    if f['fatal_violation']:
        return 'stop'
    if f['repeat_safe'] and f['provider_state'] in ('healthy', 'transient') and f['retry_budget'] == 'positive':
        return 'retry'
    if f['fallback_available'] and f['fallback_permitted']:
        return 'fallback'
    return 'stop'


def decide(facts, *, contradictory=False):
    validate_facts(facts)
    if type(contradictory) is not bool:
        raise ValueError('contradictory must be boolean')
    if contradictory:
        return Target(False, None)
    domains = [d if facts[k] == 'unknown' else (facts[k],) for k, d in DOMAINS.items()]
    outcomes = {_complete(dict(zip(DOMAINS, values))) for values in product(*domains)}
    return Target(len(outcomes) == 1, next(iter(outcomes)) if len(outcomes) == 1 else None)


def resolve_events(events):
    """Each event explicitly gives time and fact; later evidence supersedes.

    Simultaneous incompatible values remain contradictory. No language parsing.
    """
    latest = {}
    for event in events:
        if set(event) != {'time', 'fact', 'value'} or type(event['time']) is not int or event['fact'] not in DOMAINS:
            raise ValueError('invalid event')
        key, timestamp, value = event['fact'], event['time'], event['value']
        candidate = {k: 'unknown' for k in DOMAINS}
        candidate[key] = value
        validate_facts(candidate)
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, [value])
        elif timestamp == latest[key][0] and value not in latest[key][1]:
            latest[key][1].append(value)
    conflict = any(len(values) > 1 for _, values in latest.values())
    facts = {k: latest[k][1][0] if k in latest else 'unknown' for k in DOMAINS}
    return facts, conflict


def render(facts):
    validate_facts(facts)
    phrases = {
        'repeat_safe': {True: 'Repeating the operation is safe.', False: 'Repeating the operation is unsafe.', 'unknown': 'Repeat safety is unknown.'},
        'provider_state': {'healthy': 'The provider is healthy.', 'transient': 'The provider failure is transient.', 'permanent': 'The provider failure is permanent.', 'unknown': 'The provider state is unknown.'},
        'retry_budget': {'zero': 'No retry attempts remain.', 'positive': 'Retry attempts remain.', 'unknown': 'The retry budget is unknown.'},
        'fallback_available': {True: 'An alternative is available.', False: 'No alternative is available.', 'unknown': 'Alternative availability is unknown.'},
        'fallback_permitted': {True: 'The alternative is permitted.', False: 'The alternative is forbidden.', 'unknown': 'Alternative permission is unknown.'},
        'fatal_violation': {True: 'A fatal policy violation is confirmed.', False: 'There is no fatal policy violation.', 'unknown': 'Whether a fatal policy violation occurred is unknown.'},
    }
    return ' '.join(phrases[k][facts[k]] for k in DOMAINS)
