"""汇总 M1：R 矩阵、遗忘/BWT、学习量、配对 bootstrap CI、曲线。只读取已完成的评测。"""
import argparse
import csv
from common import ROOT, read_json
from m1_common import eval_steps, load_config, paired_bootstrap, roots

METRICS = ['ll_accuracy', 'gen_accuracy', 'll_macro_f1', 'format_valid_rate', 'gold_nll']


def load(R, run, step):
    folder = R['results'] / run / f'step-{step}'
    return folder if (folder / 'eval_complete.json').exists() else None


def correct(folder, task):
    return [int(r['pred_ll'] == r['label']) for r in read_json(folder / f'{task}_test.json')]


def report(c, smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = roots(smoke); out = R['reports']; out.mkdir(parents=True, exist_ok=True)
    steps = eval_steps(c, smoke); n = steps[-1]; tasks = list(c['tasks'])
    by_id = {r['id']: r for r in c['runs']}
    a_id = next(r['id'] for r in c['runs'] if r['init'] == 'base')
    b_ids = [r['id'] for r in c['runs'] if r['init'] == a_id]
    # Curves: global step; B runs start from the A endpoint.
    points = {}
    for run in [a_id] + b_ids:
        seq = [('base', 0, 0)] if run == a_id else [(a_id, n, n)]
        seq += [(run, s, s + (0 if run == a_id else n)) for s in steps]
        points[run] = [(g, load(R, r, s)) for r, s, g in seq]
    rows = []
    for run, pts in points.items():
        for g, folder in pts:
            if folder is None: continue
            m = read_json(folder / 'metrics.json')
            for task in tasks:
                for split in c['evaluation']['splits']:
                    rows.append({'run': run, 'global_step': g, 'task': task, 'split': split, **{k: m[f'{task}_{split}'][k] for k in METRICS}})
    if rows:
        with (out / 'curves.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4))
        for ax, task in zip(axes, tasks):
            for run in b_ids:
                xs = [r for r in rows if r['task'] == task and r['split'] == 'test' and (r['run'] == run or (r['run'] == a_id))]
                xs = sorted(xs, key=lambda r: r['global_step'])
                line, = ax.plot([r['global_step'] for r in xs], [100 * r['ll_accuracy'] for r in xs], marker='o', label=f'{run} label-LL')
                ax.plot([r['global_step'] for r in xs], [100 * r['gen_accuracy'] for r in xs], ls='--', color=line.get_color(), label=f'{run} generation')
            ax.axvline(n, color='grey', lw=0.8); ax.set_title(f'{task} test'); ax.set_xlabel('Optimizer steps (A then B)')
            ax.set_ylabel('Accuracy (%)'); ax.set_ylim(-2, 102)
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / 'curves.png', dpi=160); plt.close(fig)
    # R matrix on test, label-likelihood accuracy (primary) and generation accuracy.
    stages = [('初始', load(R, 'base', 0)), (f'{a_id} 后', load(R, a_id, n))] + [(f'{b} 后', load(R, b, n)) for b in b_ids]
    lines = ['# M1 汇总', '', ('smoke 验证结果，不可作为结论。' if smoke else '单 seed pilot；用于验证证据链，不作正式结论。'), '',
             '## R 矩阵（test，标签似然准确率 % / 自由生成准确率 %）', '', '| 阶段 | ' + ' | '.join(tasks) + ' |', '|---|' + '---:|' * len(tasks)]
    for name, folder in stages:
        if folder is None: lines.append(f'| {name} | ' + ' | '.join('—' for _ in tasks) + ' |'); continue
        m = read_json(folder / 'metrics.json')
        lines.append(f'| {name} | ' + ' | '.join(f"{100*m[f'{t}_test']['ll_accuracy']:.1f} / {100*m[f'{t}_test']['gen_accuracy']:.1f}" for t in tasks) + ' |')
    base, a_end = load(R, 'base', 0), load(R, a_id, n)
    a_task, b_task = by_id[a_id]['task'], by_id[b_ids[0]]['task']
    if base and a_end:
        mb, ma = read_json(base / 'metrics.json'), read_json(a_end / 'metrics.json')
        gain = ma[f'{a_task}_val']['ll_accuracy'] - mb[f'{a_task}_val']['ll_accuracy']
        lines += ['', '## 学习检查', '', f"- {a_task} val 标签似然准确率：{100*mb[f'{a_task}_val']['ll_accuracy']:.1f} → {100*ma[f'{a_task}_val']['ll_accuracy']:.1f}"
                  f"（{100*gain:+.1f} pp；pilot 工程判据 ≥ +5 pp：{'通过' if gain >= 0.05 else '未通过'}）"]
    ends = {b: load(R, b, n) for b in b_ids}
    if a_end and any(ends.values()):
        lines += ['', f'## {a_task} 遗忘（test，标签似然；配对 bootstrap 95% CI，只量化测试样本不确定性）', '',
                  '| 方法 | 遗忘 F = R[A,A]−R[B,A] (pp) | BWT (pp) | B 学习量 R[B,B]−R[A,B] (pp) |', '|---|---:|---:|---:|']
        ca = correct(a_end, a_task); s = c['evaluation']['bootstrap_samples']
        for b, folder in ends.items():
            if folder is None: continue
            boot = paired_bootstrap(ca, correct(folder, a_task), s, c['seed'])
            learn = read_json(folder / 'metrics.json')[f'{b_task}_test']['ll_accuracy'] - read_json(a_end / 'metrics.json')[f'{b_task}_test']['ll_accuracy']
            lines.append(f"| {b} | {100*boot['mean']:.1f} [{100*boot['ci95'][0]:.1f}, {100*boot['ci95'][1]:.1f}] | {-100*boot['mean']:.1f} | {100*learn:+.1f} |")
        if len(b_ids) == 2 and all(ends.values()):
            x, y = b_ids
            boot = paired_bootstrap(correct(ends[y], a_task), correct(ends[x], a_task), s, c['seed'])
            lines += ['', f"{y} − {x} 在 {a_task} test 上的差：{100*boot['mean']:+.1f} pp [{100*boot['ci95'][0]:+.1f}, {100*boot['ci95'][1]:+.1f}]"]
    for b in b_ids:
        e = R['results'] / b / 'exposure.json'
        if e.exists():
            x = read_json(e); lines.append(f"- {b} 曝光：新任务 {x['new_samples']} 条 / replay {x['replay_samples']} 条；目标 token 新 {x['new_target_tokens']} / replay {x['replay_target_tokens']}")
    lines += ['', '图：curves.png（实线标签似然，虚线自由生成）。逐样本输出见 results 同名目录。']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('M1 report:', out / 'SUMMARY.md', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/m1.json'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); report(load_config(ROOT / a.config), a.smoke)
