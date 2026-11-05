"""Durable bounded-attempt accounting; no claim of exactly-once crash recovery.

Each simulator process fsyncs intent before backend.step and completion immediately
on return. An unmatched intent bounds an uncertain command by [0,1]. A benchmark
attempt is never resumed; checkpoint restoration is only a diagnostic/warm start.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def append_durable(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def reject_restart(config, restart):
    fresh = config.get('benchmark', {}).get('fresh_uninterrupted', False) or config.get('contract', {}).get('fresh_uninterrupted', False)
    if fresh and restart:
        raise ValueError('Benchmark requires a fresh uninterrupted attempt; checkpoint restart is not a new exact-total attempt')


def command_call(function, action, *, settling=False):
    directory = os.environ.get('RL_VLA_ATTEMPT_DIR')
    if not directory:
        return function(action)
    scope = os.environ.get('RL_VLA_ATTEMPT_PHASE', 'diagnostic') + ('_settling' if settling else '_controller')
    identifier = uuid.uuid4().hex
    path = Path(directory) / 'accounting' / f'commands_{os.getpid()}.jsonl'
    common = {'attempt_id': os.environ['RL_VLA_ATTEMPT_ID'], 'command_id': identifier, 'scope': scope, 'pid': os.getpid()}
    append_durable(path, {**common, 'event': 'intent', 'time_ns': time.time_ns(), 'action': list(map(float, action))})
    result = function(action)
    # Persist before observation processing / IPC reply; lost replies don't erase
    # a backend completion already observed and durably recorded by the child.
    append_durable(path, {**common, 'event': 'completion', 'time_ns': time.time_ns()})
    return result


def summarize(directory):
    intents, completed = {}, set()
    corrupt = []
    for path in sorted((Path(directory) / 'accounting').glob('commands_*.jsonl')):
        for index, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                corrupt.append({'path': str(path), 'line': index})
                continue
            if row['event'] == 'intent': intents[row['command_id']] = row['scope']
            elif row['event'] == 'completion': completed.add(row['command_id'])
    scopes = {}
    for identifier, scope in intents.items():
        r = scopes.setdefault(scope, {'intended': 0, 'completed': 0, 'uncertain': 0})
        r['intended'] += 1
        r['completed' if identifier in completed else 'uncertain'] += 1
    for r in scopes.values():
        r['physical_command_lower_bound'] = r['completed']
        r['physical_command_upper_bound'] = r['completed'] + r['uncertain']
    return {'scopes': scopes, 'corrupt_records': corrupt, 'bounds_valid': not corrupt,
            'semantics': 'backend.step return acknowledgements; unmatched intent is uncertain [0,1]; no simulator-exact continuation'}
