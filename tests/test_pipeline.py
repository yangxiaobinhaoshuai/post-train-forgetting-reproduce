"""Checks that guard against false forgetting caused by broken masking/scoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transformers import AutoTokenizer
from common import ROOT, LABELS
from train import encode, Collator
from eval_checkpoint import number, request


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/base")

    def test_response_mask_and_chat_tail_survive_truncation(self):
        row = encode(
            self.tokenizer, {"text": "Long news story. " * 1000, "label": 2}, 512
        )
        start = next(i for i, x in enumerate(row["labels"]) if x != -100)
        self.assertEqual(
            self.tokenizer.decode(row["input_ids"][start:]),
            LABELS[2] + self.tokenizer.eos_token,
        )
        self.assertTrue(
            self.tokenizer.decode(row["input_ids"][:start]).endswith(
                "<|im_start|>assistant\n"
            )
        )
        self.assertEqual(len(row["input_ids"]), 512)
        self.assertTrue(all(x == -100 for x in row["labels"][:start]))

    def test_padding_does_not_contribute_to_loss(self):
        rows = [
            encode(self.tokenizer, {"text": t, "label": 0}, 512)
            for t in ["hello", "hello " * 100]
        ]
        batch = Collator(self.tokenizer.pad_token_id)(rows)
        end = len(rows[0]["input_ids"])
        self.assertTrue((batch["labels"][0, end:] == -100).all())
        self.assertTrue((batch["attention_mask"][0, end:] == 0).all())

    def test_score_request_and_numeric_parser(self):
        self.assertEqual(
            request("loglikelihood", ("prompt", "A"), 0).args, ("prompt", "A")
        )
        self.assertEqual(number("Calculation 10 + 2 = 12. #### 12"), "12")
        self.assertEqual(number("#### 1,234"), "1234")
        self.assertEqual(number("#### 12.0"), "12")
        self.assertIsNone(number("World"))


if __name__ == "__main__":
    unittest.main()
