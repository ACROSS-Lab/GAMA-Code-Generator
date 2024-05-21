from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Note: Uncomment this if you want to use the base_model

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  
    quantization_config=bnb_config,  
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)


ft_model = PeftModel.from_pretrained(base_model, "mistral-allblocks/checkpoint-250")

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

@app.route("/")
def home():
    return render_template("/content/index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if request.method == "POST":
        prompt = request.form["prompt"]
        model_input = eval_tokenizer(prompt, return_tensors="pt").to("cuda")

        ft_model.eval()
        with torch.no_grad():
            generated_output = ft_model.generate(**model_input, max_new_tokens=700, repetition_penalty=1.15)[0]

        generated_text = eval_tokenizer.decode(generated_output, skip_special_tokens=True)

        return render_template("/content/index.html", prompt=prompt, generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True, port=8000)