"""Shared v1 schemas plus semantic checks; no model or label inference."""
from __future__ import annotations

import json
import math
from importlib.resources import files
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


class ContractError(ValueError):
    def __init__(self, message: str, code: str = 'invalid_request'):
        super().__init__(message)
        self.code = code


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ContractError('duplicate JSON key')
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError('nonfinite number')
    if isinstance(value, str):
        try:
            value.encode('utf-8', errors='strict')
        except UnicodeError as e:
            raise ContractError('invalid UTF-8') from e
    if isinstance(value, dict):
        for k, v in value.items():
            _finite(k)
            _finite(v)
    elif isinstance(value, list):
        for v in value:
            _finite(v)


def loads(raw: str | bytes, limit: int = 65536):
    try:
        data = raw.encode('utf-8') if isinstance(raw, str) else raw
        if len(data) > limit:
            raise ContractError('JSON byte limit exceeded', 'input_too_long')
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ContractError('nonfinite number')))
        _finite(value)
        return value
    except (UnicodeError, json.JSONDecodeError) as e:
        raise ContractError('invalid UTF-8 JSON') from e


_SCHEMAS = {name: json.loads(files(__package__).joinpath('schemas', name+'.schema.json').read_text())
            for name in ('request', 'task', 'sample', 'result')}
_REGISTRY = Registry().with_resources((s['$id'], Resource.from_contents(s))
                                      for s in _SCHEMAS.values())


def shape(value, name):
    _finite(value)
    errors = list(Draft202012Validator(_SCHEMAS[name], registry=_REGISTRY).iter_errors(value))
    if errors:
        raise ContractError(f'{name} shape: {errors[0].message}')


def text(value: str, limit: int):
    if not isinstance(value, str):
        raise ContractError('text must be a string')
    _finite(value)
    if len(value.encode('utf-8')) > limit:
        raise ContractError('field byte limit exceeded', 'input_too_long')
    if not value.strip() or any((ord(c) < 32 and c not in '\t\r\n') or ord(c) == 127 for c in value):
        raise ContractError('empty text or prohibited control')
    return value.replace('\r\n', '\n').replace('\r', '\n')


def task(value):
    shape(value, 'task')
    text(value['question'], 2048)
    text(value['canonical_policy'], 8192)
    ids = [x['id'] for x in value['labels']]
    if len(set(ids)) != len(ids):
        raise ContractError('duplicate task labels')
    if value['kind'] == 'ordinal':
        anchors = [x['anchor'] for x in value['labels']]
        if any(a >= b for a, b in zip(anchors, anchors[1:])):
            raise ContractError('ordinal anchors must increase strictly')
    return value


def request(value, spec):
    task(spec)
    shape(value, 'request')
    if value['task_id'] != spec['task_id'] or value['kind'] != spec['kind']:
        raise ContractError('task/kind mismatch', 'unsupported_task')
    result = dict(value)
    for field, limit in [('state', 16384), ('policy', 8192), ('question', 2048)]:
        result[field] = text(value[field], limit)
    if result['question'] != text(spec['question'], 2048):
        raise ContractError('canonical question mismatch', 'question_mismatch')
    # The reviewed taxonomy has no policy_mismatch code; use invalid_request.
    if result['policy'] != text(spec['canonical_policy'], 8192):
        raise ContractError('canonical policy mismatch')
    if len(value['choices']) != len(spec['labels']) or set(value['choices']) != {x['id'] for x in spec['labels']}:
        raise ContractError('exact label set required', 'invalid_choices')
    return result


def sample(value, spec):
    shape(value, 'sample')
    request(value['input'], spec)
    if value['sample_id'] != value['input']['request_id'] or value['task_id'] != spec['task_id']:
        raise ContractError('sample identity mismatch')
    if value['target']['answerable'] and value['target']['label'] not in {x['id'] for x in spec['labels']}:
        raise ContractError('unknown target label')
    if value['supersedes'] == value['sample_id']:
        raise ContractError('self-supersession')
    return value


def result(value, spec, req):
    shape(value, 'result')
    if value['status'] == 'error':
        return value
    request(req, spec)
    if value['request_id'] != req['request_id'] or value['task_id'] != spec['task_id'] or value['kind'] != spec['kind']:
        raise ContractError('result identity mismatch')
    probs = value['probabilities']
    if [x['label'] for x in probs] != req['choices']:
        raise ContractError('result label order mismatch')
    if abs(sum(x['probability'] for x in probs)-1) > 1e-5:
        raise ContractError('probabilities must sum to one')
    by_id = {x['label']: x['probability'] for x in probs}
    if value['status'] == 'ok':
        winner = max(spec['labels'], key=lambda x: by_id[x['id']])['id']
        if value['verdict'] != winner or abs(value['confidence']-by_id[winner]) > 1e-5:
            raise ContractError('verdict/confidence mismatch')
        if spec['kind'] == 'ordinal':
            expected = sum(by_id[x['id']]*x['anchor'] for x in spec['labels'])
            if abs(value['score']-expected) > 1e-5:
                raise ContractError('ordinal expectation mismatch')
    return value
