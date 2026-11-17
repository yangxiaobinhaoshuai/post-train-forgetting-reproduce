import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from stage2_common import load_config, math_encode, stratified_first, output_roots, check_space
from stage2_eval import inference_state


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dropout=torch.nn.Dropout();self.config=SimpleNamespace(use_cache=False)
        self.is_gradient_checkpointing=True
    def gradient_checkpointing_disable(self): self.is_gradient_checkpointing=False
    def gradient_checkpointing_enable(self): self.is_gradient_checkpointing=True


class Stage2Tests(unittest.TestCase):
    def test_inference_restores_rng_and_training_state_on_failure(self):
        model=DummyModel();model.train();tokenizer=SimpleNamespace(padding_side='right',truncation_side='right')
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with inference_state(model,tokenizer):
                self.assertFalse(model.training);self.assertFalse(model.is_gradient_checkpointing)
                random.random();np.random.rand();torch.rand(5)
                raise RuntimeError('probe')
        self.assertEqual((random.random(),float(np.random.rand()),float(torch.rand(()))),expected)
        self.assertTrue(model.training);self.assertTrue(model.dropout.training)
        self.assertTrue(model.is_gradient_checkpointing);self.assertFalse(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side,'right');self.assertEqual(tokenizer.truncation_side,'right')
        self.assertTrue(torch.is_grad_enabled())

    def test_stratified_sampling_is_fixed_and_rejects_short_groups(self):
        rows=[{'label':i%4,'id':i} for i in range(20)]
        result=stratified_first(rows,'label',2)
        self.assertEqual([r['id'] for r in result],[0,4,1,5,2,6,3,7])
        with self.assertRaises(ValueError):stratified_first(rows,'label',6)

    def test_response_is_never_truncated(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/base',local_files_only=True)
        row={'question':'What is two plus three?','answer':'Two plus three is five.\n#### 5'}
        encoded=math_encode(tok,row,1024)
        first=next(i for i,x in enumerate(encoded['labels']) if x!=-100)
        self.assertEqual(tok.decode(encoded['input_ids'][first:]),row['answer']+tok.eos_token)
        with self.assertRaises(ValueError): math_encode(tok,row,4)

    def test_prepared_recovery_sets_have_no_question_overlap(self):
        root=ROOT/'data/stage2'
        if not (root/'manifest.json').exists():self.skipTest('Run prepare_stage2.py first')
        read=lambda name:json.loads((root/(name+'.json')).read_text())
        norm=lambda row:' '.join(row['question'].lower().split())
        a,b,c=[read(f'gsm_train_{n}') for n in [32,128,512]]
        self.assertEqual(a,b[:32]);self.assertEqual(b,c[:128])
        train={norm(r) for r in c};dev={norm(r) for r in read('gsm_dev')}
        test={norm(r) for r in read('eval_full')['gsm_test']}
        self.assertEqual(len(train),512);self.assertFalse(train&dev);self.assertFalse(train&test);self.assertFalse(dev&test)

    def test_space_guard_rejects_over_budget(self):
        c=load_config(ROOT/'configs/stage2.json')
        with self.assertRaises(RuntimeError):check_space(c,output_roots(True),reserve_bytes=10**15)


if __name__=='__main__':unittest.main()
