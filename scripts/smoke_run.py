import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from dlm_attn_res.models.llada import LLaDAConfig, LLaDAModelLM


MODEL_DIR = Path(
    '/extra_disk_1/vasilievpavel/dlm-attn-res/huggingface/hub/'
    'models--GSAI-ML--LLaDA-8B-Base/snapshots/'
    '0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
)
assert MODEL_DIR.is_dir(), f'Local checkpoint is missing: {MODEL_DIR}'
os.environ['HF_HUB_OFFLINE'] = '1'
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
config = LLaDAConfig.from_pretrained(MODEL_DIR, local_files_only=True)
model = LLaDAModelLM.from_pretrained(
    MODEL_DIR, config=config, torch_dtype=torch.bfloat16, local_files_only=True
).cuda().eval()

prompt = 'What is going on?'
encoded = tokenizer(prompt, return_tensors='pt')
device = next(model.parameters()).device
input_ids = encoded['input_ids'].to(device)
attention_mask = encoded['attention_mask'].to(device)

def get_logits(model, input_ids, attention_mask, use_attention_residuals, attention_residual_scale=0.0):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_attention_residuals=use_attention_residuals,
        attention_residual_scale=attention_residual_scale,
    )
    return output.logits

logits1 = get_logits(model, input_ids, attention_mask, False)
logits_attn_res = get_logits(model, input_ids, attention_mask, True)

mean_abs_diff = (logits1 - logits_attn_res).abs().mean()
max_abs_diff = (logits1 - logits_attn_res).abs().max()
torch.testing.assert_close(logits1, logits_attn_res, rtol=0, atol=0)
print(f'baseline equivalence: mean_abs_diff={mean_abs_diff}, max_abs_diff={max_abs_diff}')
