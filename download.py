# This file runs during container build time to get model weights built into the container

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/CodeLlama-34B-GPTQ",
        use_cache="cache"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/CodeLlama-34B-GPTQ",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache="cache"
    )

if __name__ == "__main__":
    download_model()