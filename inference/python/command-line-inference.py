import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from tqdm import tqdm 

def load_model():
    # Load base model
    base_model_id = 'Phanh2532/GAMA-Code-generator-v1.0'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model from the Hugging Face Hub
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    # Load the tokenizer from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side='left',  # Set padding_side to 'left' for correct generation results
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def run_inference(model, tokenizer):

    # Read input lines from input.txt file
    with open("./inference/python/input.txt", "r") as input_file:
        for line in tqdm(input_file):
            eval_prompt = line.strip()  # Remove trailing newline and whitespace
            model_input_prompt = tokenizer(eval_prompt, return_tensors='pt').to('cuda')
        
            model.eval()
            with torch.no_grad():
                generated_text = tokenizer.decode(model.generate(**model_input_prompt,
                                                    max_new_tokens=512,
                                                    repetition_penalty=1.15)[0],
                                        skip_special_tokens=True)
                print(generated_text)
                # Append input prompt and generated output to the output list
                output_list.append({"question": eval_prompt, "answer": generated_text})

    # Write the output list to a JSON file
    with open("./output.json", "w") as output_json:
        json.dump(output_list, output_json, indent=4)

if __name__ == "__main__":
    model, tokenizer = load_model()
    run_inference(model, tokenizer)

