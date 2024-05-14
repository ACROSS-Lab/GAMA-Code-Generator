from peft import PeftModel
import gradio as gr
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter


# Load base model 
model_id = "Phanh2532/GAMA-Code-generator-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)

# Use peft to apply checkpoint on base model
#model = PeftModel.from_pretrained(base_model, "/home/phuong-anh/gama/trained-model/mistral1/checkpoint-200")
#model = model.eval()



class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    stop = StopOnTokens()

    history_transformer_format = history + [[message, ""]]
    messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]]) for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=0.4,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield f"```{partial_message}```" 
    
    
if __name__ == "__main__":      
    gr.ChatInterface(predict).launch(share=True)
