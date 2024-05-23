import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb

import wandb

# Define token 
hf_token = "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"
wb_token = "d60af3a011349e60b02cd5a596da250258b482f7"
wandb.login(key=wb_token)

model_name = "Phanh2532/GAMA-Code-generator-v1.0"
new_model = "GAMA-Code-generator-v2.1"



def chatml_format(example):
    # Format system
    if len(example['system']) > 0:
        message = {"role": "user", "content": example['system'] + example['question']}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    else:
        message = {"role": "user", "content": example['question']}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)


    # Format instruction
    #message = {"role": "user", "content": example['question']}
    #prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load dataset
dataset = load_dataset("Phanh2532/RLHF_GAML_DATASET")['train']

# Save columns
original_columns = dataset.column_names

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)
# Print sample
print(dataset[1])




def train():
    # LoRA configuration
    peft_config = LoraConfig(
    	# Use DoRA for finetuning
    	use_dora = True, 	# Comment this if occur an error 
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=200,
        save_strategy="no",
        logging_steps=1,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True,
        report_to="wandb",
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        #peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()


    # Save artifacts
    dpo_trainer.model.save_pretrained("/home/phuong-anh/trained-model/rlhf/final-checkpoint")
    tokenizer.save_pretrained("/home/phuong-anh/trained-model/rlhf/final-checkpoint")

    # Flush memory
    del dpo_trainer, model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

train()
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# Reload model in FP16 (instead of NF4)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Merge base model with the adapter
#model = PeftModel.from_pretrained(base_model, )
#model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Push them to the HF Hub
model.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
