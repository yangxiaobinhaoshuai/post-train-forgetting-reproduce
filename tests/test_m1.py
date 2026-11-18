from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT
from m1_common import chain, encode, load_config, macro_f1, paired_bootstrap, replay_slots, stage_order, stratified_buffer, task_buffer, eval_steps

C = load_config(ROOT / 'configs/m1.json')
C2 = load_config(ROOT / 'configs/m2.json')


class M1Tests(unittest.TestCase):
    def test_replay_fraction_is_exact_and_spread(self):
        slots = replay_slots(4992, 0.1)
        self.assertEqual(sum(slots), 499)
        per_step = [sum(slots[i:i + 16]) for i in range(0, 4992, 16)]
        self.assertEqual(set(per_step), {1, 2})
        self.assertFalse(any(replay_slots(100, 0.0)))

    def test_replay_replaces_new_samples_and_shares_new_order(self):
        new = [{'source_id': i, 'label': i % 4} for i in range(5000)]
        buf = [{'source_id': 10000 + i, 'label': i % 4} for i in range(500)]
        seq = stage_order(new, [], 4992, 0.0, 42); rep = stage_order(new, buf, 4992, 0.1, 42)
        self.assertEqual(len(seq), len(rep))
        rep_new = [r['source_id'] for k, r in rep if k == 'new']
        self.assertEqual(rep_new, [r['source_id'] for _, r in seq][:len(rep_new)])
        self.assertTrue(all(r['source_id'] >= 10000 for k, r in rep if k == 'replay'))
        self.assertEqual(len({r['source_id'] for k, r in rep if k == 'replay'}), 499)

    def test_buffer_is_stratified(self):
        rows = [{'label': i % 4, 'source_id': i} for i in range(5000)]
        b = stratified_buffer(rows, 500, 42)
        self.assertEqual([sum(r['label'] == k for r in b) for k in range(4)], [125] * 4)

    def test_multitask_buffer_is_even_over_tasks_and_classes(self):
        rows = {'ag': [{'label': i % 4, 'source_id': i} for i in range(400)], 'dbpedia': [{'label': i % 14, 'source_id': i} for i in range(1400)]}
        b = task_buffer(rows, 500, 42)
        self.assertEqual(sum(r['task'] == 'ag' for r in b), 250)
        self.assertEqual(sorted({sum(r['task'] == 'dbpedia' and r['label'] == k for r in b) for k in range(14)}), [17, 18])

    def test_chain_follows_init(self):
        self.assertEqual(chain(C2, 'ABC_replay10'), ['A', 'AB_replay10', 'ABC_replay10'])

    def test_macro_f1_counts_invalid_as_error(self):
        self.assertEqual(macro_f1([0, 1], [0, 1], 2), 1.0)
        self.assertAlmostEqual(macro_f1([0, 1], [0, None], 2), 0.5)

    def test_paired_bootstrap(self):
        r = paired_bootstrap([1] * 10, [0] * 10, 200, 1)
        self.assertEqual(r['mean'], 1.0); self.assertEqual(r['ci95'], [1.0, 1.0])

    def test_eval_steps(self):
        self.assertEqual(eval_steps(C), [78, 156, 234, 312])

    def test_loss_mask_covers_only_label_and_eos(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ROOT / C['base'], local_files_only=True)
        labels = C['tasks']['dbpedia']['labels']
        row = {'title': 'X', 'text': 'word ' * 2000, 'label': 5}
        e, truncated = encode(tok, 'dbpedia', row, labels, 512)
        self.assertTrue(truncated); self.assertEqual(len(e['input_ids']), 512)
        target = [t for t in e['labels'] if t != -100]
        self.assertEqual(tok.decode(target), 'Mean of Transportation' + tok.eos_token)
        self.assertEqual(e['input_ids'][-len(target):], target)


if __name__ == '__main__': unittest.main()
