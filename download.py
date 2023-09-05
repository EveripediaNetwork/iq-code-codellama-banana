# This file runs during container build time to get model weights built into the container

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained(
        "Phind/Phind-CodeLlama-34B-v2",
        use_cache="cache"
    )
    model =  AutoModelForCausalLM.from_pretrained(
        "Phind/Phind-CodeLlama-34B-v2",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None
    )

if __name__ == "__main__":
    download_model()