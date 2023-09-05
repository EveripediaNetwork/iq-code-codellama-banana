# This file runs during container build time to get model weights built into the container

import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/CodeLlama-34B-GPTQ",
        use_cache="cache"
    )
    model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/CodeLlama-34B-GPTQ",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None
    ).to("cuda")

if __name__ == "__main__":
    download_model()