"""M1 顺序入口：准备数据 → 评测底座 → 依次训练各阶段并评测里程碑 → 汇总。持进程锁，跳过已完成部分。"""
import argparse
import fcntl
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from common import ROOT, read_json
from m1_common import atomic_json, digest, eval_steps, load_config, roots, signature
from prepare_m1 import prepare

CODE = ['m1.py', 'm1_common.py', 'prepare_m1.py', 'train_m1.py', 'eval_m1.py', 'report_m1.py', 'common.py', 'train.py', 'eval_checkpoint.py', 'stage2_common.py']


def plan(c, smoke):
    """(kind, run, step) in execution order; a B-stage's step 0 is the A endpoint and is not re-evaluated."""
    jobs = [('eval', 'base', 0)]
    for r in c['runs']:
        jobs.append(('train', r['id'], None))
        jobs += [('eval', r['id'], s) for s in eval_steps(c, smoke)]
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/m1.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); config_path = ROOT / a.config; c = load_config(config_path)
    jobs = plan(c, a.smoke)
    if a.dry_run:
        for kind, run, step in jobs: print(kind, run, '' if step is None else f'step {step}')
        return
    R = roots(a.smoke)
    for k in ['results', 'artifacts', 'reports']: R[k].mkdir(parents=True, exist_ok=True)
    with (R['results'] / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another M1 pipeline is already running')
        prepare(c)
        if a.prepare_only: return
        identity = {'config': c, 'data_manifest_sha256': digest(R['data'] / 'manifest.json'), 'uv_lock_sha256': digest(ROOT / 'uv.lock'),
                    'code': {s: digest(ROOT / 'scripts' / s) for s in CODE}, 'smoke': a.smoke}
        path = R['results'] / 'experiment.json'
        if path.exists() and read_json(path)['signature'] != signature(identity):
            raise RuntimeError('Code/config/data changed since this M1 experiment; archive its outputs before rerunning')
        atomic_json(path, {'signature': signature(identity), 'identity': identity})
        (R['results'] / 'environment.txt').write_text('\n'.join(f"{d.metadata['Name']}=={d.version}" for d in sorted(distributions(), key=lambda d: d.metadata['Name'].lower())) + '\n')
        status = lambda phase, job=None: atomic_json(R['results'] / 'running-status.json', {'phase': phase, 'job': job, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'smoke': a.smoke})
        from report_m1 import report
        try:
            for kind, run, step in jobs:
                status('running', [kind, run, step])
                script = 'train_m1.py' if kind == 'train' else 'eval_m1.py'
                cmd = [sys.executable, str(ROOT / 'scripts' / script), '--config', str(config_path), '--run', run] + ([] if step is None else ['--step', str(step)]) + (['--smoke'] if a.smoke else [])
                log = R['results'] / run / 'console.log'; log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as stream:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in proc.stdout: print(line, end='', flush=True); stream.write(line); stream.flush()
                        code = proc.wait()
                    except BaseException:
                        proc.terminate(); proc.wait(); raise
                if code: raise RuntimeError(f'{kind} {run} {step} failed (exit {code}); see {log}')
                if kind == 'eval': report(c, a.smoke)
            status('complete')
        except BaseException as error:
            status('interrupted_or_failed', str(error)); raise
        finally:
            report(c, a.smoke)


if __name__ == '__main__': main()
