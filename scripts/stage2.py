"""Stage2 顺序入口，持有进程锁，跳过已完成 run。"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
from importlib.metadata import distributions
from pathlib import Path
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, read_json
from stage2_common import atomic_json, load_config, output_roots, signature, digest, prepared_manifest, check_space
from prepare_stage2 import prepare
from train_stage2 import run_specs
from stage2_eval import evaluate


def main():
    p=argparse.ArgumentParser(description='Stage2: prepare, references, sensitivity, recovery, report')
    p.add_argument('--config',default='configs/stage2.json');p.add_argument('--smoke',action='store_true')
    p.add_argument('--runs',nargs='+',help='Only these run IDs; rerun without this flag for the rest')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c=load_config(a.config);specs=run_specs(c)
    ids=[r['id'] for r in specs]
    if a.runs:
        unknown=set(a.runs)-set(ids)
        if unknown: p.error(f'Unknown runs: {sorted(unknown)}')
        specs=[r for r in specs if r['id'] in a.runs]
    if a.dry_run:
        print('1. Verify local models and prepare Stage2 data')
        print('2. Evaluate shared base and collapsed references (FP32 softmax)')
        for r in specs: print(f"   {r['id']}: {'2 smoke steps' if a.smoke else str(c[r['group']]['max_steps'])+' steps'}")
        print('3. Write Stage2 curves and status; no Stage1 files modified')
        return
    roots=output_roots(a.smoke)
    for path in roots.values(): path.mkdir(parents=True,exist_ok=True)
    with (roots['results']/'.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another Stage2 pipeline is already running')
        check_space(c,roots)
        prepare(c);manifest=prepared_manifest(c)
        for relative,expected in manifest['model_hashes'].items():
            if digest(ROOT/relative)!=expected: raise RuntimeError(f'Source model changed: {relative}')
        if a.prepare_only: return
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        code=['stage2.py','stage2_common.py','prepare_stage2.py','stage2_eval.py','train_stage2.py','train.py','common.py','eval_checkpoint.py']
        identity={'config':c,'data_manifest_sha256':digest(ROOT/'data/stage2/manifest.json'),
                  'code':{s:digest(ROOT/'scripts'/s) for s in code},'uv_lock_sha256':digest(ROOT/'uv.lock'),'smoke':a.smoke}
        path=roots['results']/'experiment.json'
        exp={'signature':signature(identity),'config_signature':signature(c),'identity':identity}
        if path.exists() and read_json(path)['signature']!=exp['signature']:
            raise RuntimeError('Code/config/inputs changed since this Stage2 experiment; archive its Stage2 output directories before rerunning')
        atomic_json(path,exp)
        (roots['results']/'environment.txt').write_text('\n'.join(f"{d.metadata['Name']}=={d.version}" for d in sorted(distributions(),key=lambda d:d.metadata['Name'].lower()))+'\n')
        from report_stage2 import report
        state_path=roots['results']/'running-status.json'
        def status(phase,run=None):
            atomic_json(state_path,{'phase':phase,'run':run,'updated_utc':datetime.now(timezone.utc).isoformat(),'smoke':a.smoke})
        try:
            for ref in ['base','collapsed']:
                status('reference_evaluation',ref)
                out=roots['results']/'references'/ref
                if (out/'complete.json').exists():
                    for rel,expected in read_json(out/'complete.json')['hashes'].items():
                        if digest(ROOT/rel)!=expected: raise RuntimeError(f'Reference output changed: {rel}')
                    continue
                check_space(c,roots)
                tok=AutoTokenizer.from_pretrained(ROOT/c['references']['base'],local_files_only=True)
                model=AutoModelForCausalLM.from_pretrained(ROOT/c['references'][ref],local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
                evaluate(model,tok,c,out,profile='full',batch_size=c['evaluation']['standalone_batch_size'],smoke=a.smoke)
                atomic_json(out/'complete.json',{'hashes':{str(f.relative_to(ROOT)):digest(f) for f in out.glob('*.json') if f.name!='complete.json'}})
                del model,tok;gc.collect();torch.cuda.empty_cache()
            report(c,a.smoke)
            for spec in specs:
                status('training',spec['id'])
                cmd=[sys.executable,str(ROOT/'scripts/train_stage2.py'),'--config',str(Path(a.config).resolve()),'--run',spec['id']]
                if a.smoke: cmd.append('--smoke')
                log=roots['results']/spec['id']/'console.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as stream:
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    try:
                        for line in process.stdout: print(line,end='',flush=True);stream.write(line);stream.flush()
                        code=process.wait()
                    except BaseException:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                        raise
                if code: raise RuntimeError(f"Run {spec['id']} failed (exit {code}); see {log}")
                report(c,a.smoke)
            status('complete')
        except BaseException as error:
            status('interrupted_or_failed',str(error));raise
        finally: report(c,a.smoke)


if __name__=='__main__': main()
