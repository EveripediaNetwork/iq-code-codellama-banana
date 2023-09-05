# This file runs during container build time to get model weights built into the container

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

def download_model():
    model_path = "Phind/Phind-CodeLlama-34B-v2"
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

if __name__ == "__main__":
    download_model()