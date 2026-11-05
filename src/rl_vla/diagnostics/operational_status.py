"""Read-only queue health; a stopped parent cannot masquerade as progress."""
import hashlib
import json
from pathlib import Path

TASKS = ('object_0', 'object_1', 'object_2')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'state': fields[0], 'start_ticks': int(fields[19]),
                'command': [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]}
    except FileNotFoundError:
        return None


def launch(root, task, domain):
    if domain == 'causal':
        candidates = []
        for path in (root/'diagnostic_policy_dispatch_launch').glob(f'{task}_*.json'):
            value = read(path)
            if value.get('task') == task and value.get('pid') and value.get('command'):
                candidates.append((path.stat().st_mtime_ns, path, value))
        if not candidates:
            return None
        _, path, value = max(candidates)
        return {'path': str(path), 'pid': value['pid'], 'command': value['command']}
    path = root/'ood_per_lane_release_review_v1/launch_manifest_v1.json'
    value = read(path)
    row = value.get('new_queues', {}).get(task) if value else None
    return {'path': str(path), **row['parent']} if row else None


def snapshot(root, *, process_reader=process):
    root = Path(root).resolve()
    rows = []
    for domain, folder in (('causal', 'interventions'), ('ood', 'ood')):
        for task in TASKS:
            path = root/folder/task/'queue_status.json'
            marker = root/folder/task/'queue_complete.json'
            try:
                status, complete, owner = read(path) or {}, read(marker) or {}, launch(root, task, domain)
                current = process_reader(owner['pid']) if owner else None
                command = owner['command'] if owner else None
                if command and command[:2] == ['taskset', '-c']:
                    command = command[3:]
                alive = (current is not None and current['state'] != 'Z' and current['command'] == command
                         and (not owner.get('start_ticks') or owner['start_ticks'] == current['start_ticks'])) if owner else None
                done = complete.get('complete') is True and complete.get('passed') is not False
                if domain == 'causal':
                    done &= complete.get('passed') is True
                failed = status.get('state') == 'failed' or status.get('passed') is False or complete.get('passed') is False
                reason = status.get('error') or status.get('last_error') or ''
                if not done and owner and not alive:
                    failed = True
                    reason = reason or 'Launched parent exited or no longer matches its identity before a complete marker'
                state = 'failed' if failed else ('complete' if done else 'pending')
                rows.append({'domain': domain, 'task': task, 'state': state, 'queue_state': status.get('state'),
                             'reason': reason, 'parent': owner, 'parent_alive': alive,
                             'status_path': str(path), 'status_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                             'completion_path': str(marker), 'completion_marker_present': marker.exists(),
                             'active_lanes': status.get('active_lanes'), 'completed_cells': status.get('completed_cells')})
            except (OSError, ValueError, TypeError, KeyError) as error:
                rows.append({'domain': domain, 'task': task, 'state': 'failed', 'reason': f'Queue health could not be read: {error}',
                             'status_path': str(path), 'completion_path': str(marker)})
    failures = [row for row in rows if row['state'] == 'failed']
    return {'rows': rows, 'failures': failures, 'failed_queues': len(failures),
            'scope': 'Operational health only. Completed valid artifacts are preserved; missing/incomplete conditions are never inferred from episode counts. This collector does not release resources or change producer state.'}
