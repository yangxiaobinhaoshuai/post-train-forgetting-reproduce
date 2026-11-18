"""汇总 M1/M2：每条方法链的 R 矩阵、AA/遗忘/BWT、学习量、配对 bootstrap CI、曲线。只读取已完成的评测。"""
import argparse
import csv
from common import ROOT, read_json
from m1_common import chain, eval_steps, load_config, paired_bootstrap, roots

METRICS = ['ll_accuracy', 'gen_accuracy', 'll_macro_f1', 'format_valid_rate', 'gold_nll']


def load(R, run, step):
    folder = R['results'] / run / f'step-{step}'
    return read_json(folder / 'metrics.json') | {'_folder': folder} if (folder / 'eval_complete.json').exists() else None


def correct(m, task):
    return [int(r['pred_ll'] == r['label']) for r in read_json(m['_folder'] / f'{task}_test.json')]


def pp(x):
    return f'{100*x:.1f}'


def report(c, smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = roots(c, smoke); out = R['reports']; out.mkdir(parents=True, exist_ok=True)
    steps = eval_steps(c, smoke); n = steps[-1]; tasks = list(c['tasks']); by_id = {r['id']: r for r in c['runs']}
    leaves = [r['id'] for r in c['runs'] if not any(x['init'] == r['id'] for x in c['runs'])]
    s = c['evaluation']['bootstrap_samples']
    rows = []
    for leaf in leaves:
        pts = [('base', 0, 0)] + [(run, st, i * n + st) for i, run in enumerate(chain(c, leaf)) for st in steps]
        for run, st, g in pts:
            m = load(R, run, st)
            if m is None: continue
            for task in tasks:
                for split in c['evaluation']['splits']:
                    if f'{task}_{split}' in m:
                        rows.append({'chain': leaf, 'run': run, 'global_step': g, 'task': task, 'split': split, **{k: m[f'{task}_{split}'][k] for k in METRICS}})
    if rows:
        with (out / 'curves.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4))
        for ax, task in zip(axes, tasks):
            for leaf in leaves:
                xs = sorted([r for r in rows if r['chain'] == leaf and r['task'] == task and r['split'] == 'test'], key=lambda r: r['global_step'])
                line, = ax.plot([r['global_step'] for r in xs], [100 * r['ll_accuracy'] for r in xs], marker='o', ms=3, label=f'{leaf} label-LL')
                ax.plot([r['global_step'] for r in xs], [100 * r['gen_accuracy'] for r in xs], ls='--', color=line.get_color(), label=f'{leaf} generation')
            for i in range(1, len(chain(c, leaves[0]))): ax.axvline(i * n, color='grey', lw=0.8)
            ax.set_title(f'{task} test'); ax.set_xlabel('Optimizer steps (stages in sequence)'); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(-2, 102)
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / 'curves.png', dpi=160); plt.close(fig)

    lines = [f"# {c['name'].upper()} 汇总", '', ('smoke 验证结果，不可作为结论。' if smoke else '单 seed pilot；探索性，不作正式结论。'), '',
             '标签似然准确率为主指标；括号外/内为 test 标签似然 / 自由生成准确率（%）。配对 bootstrap 95% CI 只量化测试样本不确定性。']
    base = load(R, 'base', 0); finals = {}
    for leaf in leaves:
        runs = chain(c, leaf); order = [by_id[r]['task'] for r in runs]
        ends = [base] + [load(R, r, n) for r in runs]
        lines += ['', f'## {leaf}：' + ' → '.join(order), '', '| 阶段 | ' + ' | '.join(tasks) + ' |', '|---|' + '---:|' * len(tasks)]
        for name, m in zip(['初始'] + [f'{r} 后' for r in runs], ends):
            cells = ['—' if m is None else f"{pp(m[f'{t}_test']['ll_accuracy'])} ({pp(m[f'{t}_test']['gen_accuracy'])})" for t in tasks]
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        if any(m is None for m in ends): continue
        finals[leaf] = ends[-1]
        acc = lambda t, i: ends[i][f'{order[t]}_test']['ll_accuracy']  # R[i, task t]; stage index 0 = initial
        T = len(order)
        aa = sum(acc(t, T) for t in range(T)) / T
        lines += ['', f'- 最终平均 AA：{pp(aa)}']
        if T > 1:
            forget = [max(acc(t, i) for i in range(t + 1, T)) - acc(t, T) for t in range(T - 1)]
            bwt = [acc(t, T) - acc(t, t + 1) for t in range(T - 1)]
            lines.append(f'- 平均遗忘 F：{pp(sum(forget)/len(forget))} pp；BWT：{pp(sum(bwt)/len(bwt))} pp')
            for t in range(T - 1):
                b = paired_bootstrap(correct(ends[t + 1], order[t]), correct(ends[T], order[t]), s, c['seed'])
                lines.append(f"- {order[t]}：学完后 {pp(acc(t, t+1))} → 最终 {pp(acc(t, T))}，下降 {pp(b['mean'])} pp [{pp(b['ci95'][0])}, {pp(b['ci95'][1])}]；阶段末最大值后遗忘 {pp(forget[t])} pp")
        for t in range(T):
            val = lambda i: ends[i][f'{order[t]}_val']['ll_accuracy']
            lines.append(f"- 学习量 {order[t]}：test {pp(acc(t, t+1) - acc(t, t))} pp；val {pp(val(t))} → {pp(val(t+1))}（{'≥' if val(t+1) - val(t) >= 0.05 else '<'} +5 pp 判据）")
        for r in runs:
            e = R['results'] / r / 'exposure.json'
            if e.exists() and by_id[r]['replay_fraction']:
                x = read_json(e)
                lines.append(f"- {r} 曝光：新任务 {x['new_samples']} 条 / replay {x['replay_samples']} 条；目标 token 新 {x['new_target_tokens']} / replay {x['replay_target_tokens']}")
    if len(finals) == 2:
        (x, mx), (y, my) = finals.items()
        lines += ['', f'## {y} − {x}（最终 test，标签似然）', '']
        for t in tasks:
            b = paired_bootstrap(correct(my, t), correct(mx, t), s, c['seed'])
            lines.append(f"- {t}：{b['mean']*100:+.1f} pp [{b['ci95'][0]*100:+.1f}, {b['ci95'][1]*100:+.1f}]")
    lines += ['', '图：curves.png（实线标签似然，虚线自由生成，竖线为阶段边界）。逐样本输出见 results 同名目录。']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('Report:', out / 'SUMMARY.md', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/m1.json'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); report(load_config(ROOT / a.config), a.smoke)
