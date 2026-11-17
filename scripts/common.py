import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
LABELS = ['World', 'Sports', 'Business', 'SciTech']

def config():
    return json.loads((ROOT / 'configs/stage1.json').read_text())

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def read_json(path):
    return json.loads(Path(path).read_text())

def ag_prompt(text):
    return 'Classify the news article. Reply with exactly one category: World, Sports, Business, SciTech.\n\nArticle: ' + text

def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
