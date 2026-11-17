"""汇总 Stage2：快速集曲线与完整集终点分开展示。"""
import argparse
import csv
from pathlib import Path
from common import read_json
from stage2_common import load_config, output_roots
from train_stage2 import run_specs


def report(c,smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roots=output_roots(smoke);out=roots['reports'];out.mkdir(parents=True,exist_ok=True)
    quick_rows=[];full_rows=[];states=[]
    def record(run_id,step,metrics,status):
        return {'run':run_id,'step':step,'status':status, **{k:metrics[k]['accuracy'] for k in ['ag_test','gsm_test','mmlu_test']},
                'gsm_news_label_rate':metrics['gsm_test']['news_label_rate'],
                'mmlu_letter_probability_mass':metrics['mmlu_test']['letter_probability_mass']}
    for spec in run_specs(c):
        folder=roots['results']/spec['id']
        status='complete' if (folder/'complete.json').exists() else 'partial' if folder.exists() else 'pending'
        states.append((spec['id'],status))
        reference=roots['results']/'references'/spec['start']/'metrics.json'
        if reference.exists():
            m=read_json(reference);quick_rows.append(record(spec['id'],0,m['quick_view'],status))
        for path in sorted(folder.glob('step-*/metrics.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
            m=read_json(path);step=int(path.parent.name.split('-')[-1])
            quick_rows.append(record(spec['id'],step,m.get('quick_view',m),status))
            if m['profile']=='full': full_rows.append(record(spec['id'],step,m,status))
    for name,rows in [('quick_curve',quick_rows),('full_endpoints',full_rows)]:
        if rows:
            with (out/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if quick_rows:
        for group in ['sensitivity','recovery']:
            ids=[r['id'] for r in run_specs(c) if r['group']==group]
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            for ax,task in zip(axes,['ag_test','gsm_test','mmlu_test']):
                for run_id in ids:
                    rows=[r for r in quick_rows if r['run']==run_id]
                    if rows: ax.plot([r['step'] for r in rows],[100*r[task] for r in rows],marker='o',label=run_id)
                ax.set_title(task+' (fixed quick subset)');ax.set_xlabel('Optimizer steps');ax.set_ylabel('Accuracy (%)');ax.set_ylim(-2,102)
            axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/(group+'.png'),dpi=160);plt.close(fig)
    lines=['# Stage2 运行状态','','smoke 验证结果，不可作为正式结论。' if smoke else '快速集曲线和完整集终点分别汇总；partial 表示尚未完成。','','| Run | 状态 |','|---|---|']
    lines += [f'| {name} | {state} |' for name,state in states]
    lines += ['','输出：quick_curve.csv、full_endpoints.csv、sensitivity.png、recovery.png（有对应数据时生成）。','完整原始输出在同名 results 目录。未完成的 run 只作进度展示，不能当作最终结果。']
    (out/'STATUS.md').write_text('\n'.join(lines)+'\n')
    print('Stage2 report:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2.json');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();report(load_config(a.config),a.smoke)
