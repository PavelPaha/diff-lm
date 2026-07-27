pip install transformers==4.57.1 accelerate==0.34.2 lm-eval==0.4.10 peft
pip install antlr4-python3-runtime==4.11 math_verify sympy
pip install datasets==3.6.0


export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true



#===============================
# ILLaDA-8B-Base
#===============================

# For MMLU, ARC-C, Hellaswag, we use 8 gpus
accelerate_args="--multi_gpu --num_processes 8 --main_process_port $(expr $RANDOM % 10000 + 10000)"

accelerate launch $accelerate_args eval_illada.py 
    --tasks arc_challenge,hellaswag \
    --num_fewshot 0 \
    --model illada_dist \
    --batch_size 32 \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Base',cfg=0.0,is_check_greedy=False,padd_eos=True,llh='confidence',add_bos_token=True 

accelerate launch $accelerate_args eval_illada.py \
    --tasks mmlu,cmmlu \
    --num_fewshot 5 \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,padd_eos=True,add_bos_token=True


# For BBH, GSM8K, Math, Humaneval and MBPP, we use 32 gpus
# You can modify the following lines to use your own cluster
accelerate_args="\
  --multi_gpu \
  --num_machines ${nnodes} \
  --num_processes ${total_processes} \
  --machine_rank ${ARNOLD_ID} \
  --main_process_ip ${master_addr} \
  --main_process_port ${master_port}"


accelerate launch $accelerate_args eval_illada.py \
  --tasks bbh \
  --model illada_dist \
  --model_args model_path='GSAI-ML/iLLaDA-8B-Base',gen_length=1024,steps=32,block_length=32,var=True


accelerate launch $accelerate_args eval_illada.py \
  --tasks gsm8k_cot \
  --model illada_dist \
  --num_fewshot 8 \
  --model_args model_path='GSAI-ML/iLLaDA-8B-Base',gen_length=1024,steps=32,block_length=32,var=True


accelerate launch $accelerate_args eval_illada.py \
  --tasks minerva_math \
  --model illada_dist \
  --model_args model_path='GSAI-ML/iLLaDA-8B-Base',gen_length=1024,steps=32,block_length=32,var=True


accelerate launch $accelerate_args eval_illada.py \
  --tasks humaneval \
  --model illada_dist \
  --confirm_run_unsafe_code \
  --model_args model_path='GSAI-ML/iLLaDA-8B-Base',gen_length=512,steps=512,block_length=512


accelerate launch $accelerate_args eval_illada.py \
  --tasks mbpp \
  --model illada_dist \
  --confirm_run_unsafe_code \
  --model_args model_path='GSAI-ML/iLLaDA-8B-Base',gen_length=1024,steps=32,block_length=32,var=True



#===============================
# ILLaDA-8B-Instruct
#===============================
# You can modify the following lines to use your own cluster
accelerate_args="\
  --multi_gpu \
  --num_machines ${nnodes} \
  --num_processes ${total_processes} \
  --machine_rank ${ARNOLD_ID} \
  --main_process_ip ${master_addr} \
  --main_process_port ${master_port}"


task='mmlu_redux_generative'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=4,block_length=4,steps=4 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn


task='minerva_math'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=4096,steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=3 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template 


export PYTHONPATH="./eval_instruct${PYTHONPATH:+:$PYTHONPATH}"
task='mmlu_generative'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=3,block_length=3,steps=3 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn


export PYTHONPATH="./eval_instruct${PYTHONPATH:+:$PYTHONPATH}"
task='mmlu_pro_reasoning'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=4096,steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=3 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template 


export PYTHONPATH="./eval_instruct${PYTHONPATH:+:$PYTHONPATH}"
task='gsm8k_cot_reasoning'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=2048,steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=3 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template 


export PYTHONPATH="./eval_instruct${PYTHONPATH:+:$PYTHONPATH}"
task='humaneval_reasoning'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=2048,steps=32,block_length=32,var=True,end_think_logit_boost=100,end_think_boost_power=1 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path /mnt/hdfs/nieshen/Eval_illada_ablation/${task}_${epochs}epoch \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template 


export PYTHONPATH="./eval_instruct${PYTHONPATH:+:$PYTHONPATH}"
task='mbpp_reasoning'
accelerate launch $accelerate_args eval_illada.py \
    --model illada_dist \
    --model_args model_path='GSAI-ML/iLLaDA-8B-Instruct',gen_length=2048,steps=16,block_length=16,var=True,end_think_logit_boost=100,end_think_boost_power=3 \
    --tasks ${task} \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template 