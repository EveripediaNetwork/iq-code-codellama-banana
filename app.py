from potassium import Potassium, Request, Response
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model_path = "Phind/Phind-CodeLlama-34B-v2"
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    temperature = request.json.get("temperature", 0)

    tokenizer = context.get("tokenizer")
    model = context.get("model")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    output = tokenizer.decode(outputs[0])

    return Response(
        json = {"outputs": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()