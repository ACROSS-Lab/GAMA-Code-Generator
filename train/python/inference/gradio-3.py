from peft import PeftModel
import gradio as gr
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_peft_model():
    # Load base model 
    model_id = "Phanh2532/GAMA-Code-generator-v1.0"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)

    # Use peft to apply checkpoint on base model
    peft_model = PeftModel.from_pretrained(base_model, checkpt)
    peft_model = peft_model.eval()

    return peft_model, tokenizer

def greet(name):

    return f"Hello {name}! I'm GAMA Chatbot. \n A code generator bot, who will help you get into GAML language of GAMA platform. \n For more details, you can visit https://gama-platform.org/wiki/Introduction"

'''
def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)

def generate_response(peft_model, tokenizer, history):
    # Concatenate chat history
    chat_history = ""
    for entry in history:
        chat_history += entry[0] + "\n"

    # Tokenize input
    model_input = tokenizer(chat_history, return_tensors="pt").to("cuda")

    # Generate response
    output = peft_model.generate(**model_input, max_new_tokens=700, repetition_penalty=1.15)

    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Check if the response contains code
    # Need improvement in futer
    if "{" in response:
        response = f"```\n{response}\n```"  # Enclose response in triple backticks for Markdown code block

    return response

def bot(peft_model, tokenizer, history):

    # Use peft model to generate response
    response = generate_response(peft_model, tokenizer ,history)
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history
    
    return history

def print_like_dislike(x):
    print(x.index, x.value, x.liked)

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history
'''
def gradio_interface():
    avatar_image_url = "https://i.imgur.com/DxhdL3t.jpg"  # Modify this path accordingly

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div style='display: flex; justify-content: center; align-items: center; font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;'>
                <div style='margin-right: 10px;'>
                    <img src='https://i.imgur.com/DuzoviE.png' alt='Your Image' width='30'>
                </div>
                <div style='text-align: center;'>
                    <h1>GAMA PLATFORM CHATBOT</h1>
                </div>

            </div>
            """
        )

        greet_input = gr.Textbox(placeholder="What is your name?")
        greet_output = gr.Textbox()
        greet_input.change(greet, greet_input, greet_output)

        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, avatar_image_url),
        )

        with gr.Row():
            txt = gr.Textbox(
                scale=30,
                show_label=False,
                placeholder="Enter text and press enter, or upload an text file",
                container=False,
            )
            '''
            experiment_prompt_input = gr.Textbox(
                label="Experiment Prompt",
                placeholder="Enter your experiment prompt here",
                container=False
            )
            '''
            btn = gr.UploadButton("📁", file_types=["text"])

        txt_msg = txt.submit([chatbot, txt], [chatbot, txt], queue=False).then(
            bot, chatbot, chatbot, api_name="bot_response"
        )
        txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        # chatbot.like(print_like_dislike, None, None)
	
    demo.queue()
    demo.launch(share=True)

if __name__ == "__main__":
    checkpt = input("Enter checkpoint path link: ")
    peft_model, tokenizer = load_peft_model()
    gradio_interface()
