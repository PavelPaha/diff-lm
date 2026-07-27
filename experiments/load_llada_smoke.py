
import torch
from transformers import AutoTokenizer

from dlm_attn_res.models.llada import LLaDAConfig, LLaDAModelLM


def main():
    model_id = "GSAI-ML/LLaDA-8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = LLaDAConfig.from_pretrained(model_id)
    model = LLaDAModelLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    print(f"Loaded {model_id}; tokenizer size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
