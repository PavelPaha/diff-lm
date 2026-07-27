#!/usr/bin/env bash
set -euo pipefail

# LLaDA paper Table 1 setting: no classifier-free guidance.
export HF_DATASETS_TRUST_REMOTE_CODE=true
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
eval_python="${DLM_EVAL_PYTHON:-/extra_disk_1/vasilievpavel/dlm-attn-res/venv-lm-eval/bin/python}"

"${eval_python}" -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m dlm_attn_res.evaluation.lm_eval_adapter \
  --tasks hellaswag \
  --model llada_dist \
  --batch_size 8 \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=128" \
  --output_path "${project_root}/results/llada_8b_base_hellaswag_mc128_cfg0"
