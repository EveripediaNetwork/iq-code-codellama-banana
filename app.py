from potassium import Potassium, Request, Response
import torch
from transformers import AutoTokenizer, AutoGPTQForCausalLM

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/CodeLlama-34B-GPTQ",
        use_cache="cache"
    )
    model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/CodeLlama-34B-GPTQ",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None
    )
    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_new_tokens = request.json.get("max_new_tokens")

    tokenizer = context.get("tokenizer")
    model = context.get("model")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=int(max_new_tokens))
    output = tokenizer.decode(outputs[0])

    return Response(
        json = {"outputs": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()