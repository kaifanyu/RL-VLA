"""Single-use persistent launch for the user-assigned GPU 0 RLDT pilot."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/workspace/upenn/worktrees/online-object0-ready-20261105')
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / 'third_party/openpi/.venv/bin/python'


def main():
    used = int(subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    if used > 1024:
        raise RuntimeError(f'Assigned GPU 0 is occupied: {used} MiB')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 48 * 2**30 or shutil.disk_usage(HERE).free < 100 * 2**30:
        raise RuntimeError('Need 48 GiB host RAM and 100 GiB free disk')
    cpus = set(range(8))
    if not cpus <= os.sched_getaffinity(0):
        raise RuntimeError('Assigned CPU set 0-7 is not available')
    # A finished or interrupted run must never be accidentally relaunched here.
    (HERE / 'launch.lock').mkdir()
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', RLDT_CPUSET='0,1,2,3,4,5,6,7',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               LP_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.55',
               JAX_COMPILATION_CACHE_DIR=str(HERE / 'jax-cache'))
    command = [str(PYTHON), 'scripts/launch_online_object0.py', '--method', 'rldt',
               '--seed', '0', '--gpu', '0', '--output-root', str(HERE / 'runs')]
    record = {'command': command, 'gpu': 0, 'cpus': sorted(cpus),
              'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'status': 'running', 'wrapper_pid': os.getpid()}
    def save():
        temporary = HERE / 'launch_status.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(HERE / 'launch_status.json')
    save()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        record.update(exit_code=result.returncode,
                      status='complete' if result.returncode == 0 else 'failed')
    except BaseException as exc:
        record.update(status='interrupted', error=repr(exc))
        raise
    finally:
        record['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
