"""Minimal interactive chat using the official LLaDA sampling settings."""

import torch
from transformers import AutoTokenizer

from dlm_attn_res.models.llada import LLaDAConfig, LLaDAModelLM
from dlm_attn_res.sampling.llada import generate


def chat():
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    config = LLaDAConfig.from_pretrained(model_id)
    model = LLaDAModelLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    gen_length = steps = 128
    print(f"** Answer Length: {gen_length} | Sampling Steps: {steps} **")
    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")
        message = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        input_ids = torch.tensor(tokenizer(text)["input_ids"], device="cuda").unsqueeze(0)
        prompt = input_ids if conversation_num == 0 else torch.cat([prompt, input_ids[:, 1:]], dim=1)
        out = generate(
            model, prompt, steps=steps, gen_length=gen_length, block_length=32,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
        )
        print("Bot's reply:", tokenizer.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=True)[0])
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1


if __name__ == "__main__":
    chat()
