import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
from datetime import datetime
from trl import SFTTrainer

import argparse
import os


def load_base_model():
    # Load base model
    base_model_id = args.base_model_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=args.quantization,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model from the Hugging Face Hub
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    # Load the tokenizer from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side=args.padding_side,
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_data():
    # Load training and evaluation datasets
    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path

    # Preprocess the datasets
    train_dataset = formatting_func(train_data_path)
    eval_dataset = formatting_func(eval_data_path)

    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    return tokenized_train_dataset, tokenized_val_dataset


def formatting_func(input):
    # Format the input data
    text = f"### Question: {input['question']}\n ### Answer: {input['answer']}"
    return text


def generate_and_tokenize_prompt(prompt):
    # Generate and tokenize prompts
    max_length = args.max_token_length
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def config_model(model):
    # Configure LoRA model
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
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
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Get the LoRA model
    #model = get_peft_model(model, config)

    # Configure Fully Sharded Data Parallel (FSDP)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    # Initialize the Accelerator
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)

    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        # Create a list of device IDs
        device_ids = list(range(num_gpus))

        # Move the model to a device
        device = torch.device("cuda:0")  # Assuming you want to use GPU 0
        model.to(device)

        # Wrap the model with DataParallel
        model = nn.DataParallel(model, device_ids)

    return model, config


def train_model(tokenized_train_dataset, tokenized_val_dataset, model, config, tokenizer):
    # Initialize the train args
    args=transformers.TrainingArguments(
            output_dir=args.out_model_dir,
            #num_train_epochs=10,
            warmup_steps=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            optim="paged_adamw_8bit",
            logging_steps=args.logging_steps,
            logging_dir=args.logs_dir,
            save_strategy="steps",
            save_steps=args.save_steps,
            evaluation_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            do_eval=args.do_eval,
            #report_to="wandb",
            run_name=f"{args.out_model_dir}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        peft_config=config,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=args,
        max_seq_length=700,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        ),

    # Disable caching for training
    model.config.use_cache = False
    trainer.train()    


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # General configuration
    parser.add_argument('--base_model_id', type=str, help='Hugging Face id ot the base model', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data', default='./total.json')
    parser.add_argument('--eval_data_path', type=str, help='Path to the validation data', default='./eval.json')
    parser.add_argument('--max_token_length', type=int, help='Maximum token length allowed', default=1700)
    parser.add_argument('--logs_dir', type=str, help='Ouput log directory', default='./logs')
    parser.add_argument('--out_model_dir', type=str, help='Output directory for training checkpoints', default='./mistral')
    parser.add_argument('--padding_side', type=str, help='Side to pad the dataset sample', choices=['right','left'], default='left')
    parser.add_argument('--quantization', type=str, help='Quantization type', choices=['nf4','fp4'], default='nf4')

    # Training config
    parser.add_argument('--logging_steps', type=int, help='When to start reporting loss', default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, help='Number of GPUs used for training', default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=4)
    parser.add_argument('--save_steps', type=int, help='Number of epochs between two checkpoints saves', default=50)
    parser.add_argument('--eval_steps', type=int, help='Number of epochs between two evaluation steps', default=50)
    parser.add_argument('--learning_rate', type=int, help='Learning rate for training', default=1e-4)
    parser.add_argument('--do_eval', type=bool, help='Learning rate for training', default=True)
    parser.add_argument('--max_steps', type=int, help='Training steps', default=200)

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, help='Intrinsic rank for LoRA optimization', default=32)
    parser.add_argument('--lora_alpha', type=int, help='Alpha parameter for LoRA optimization', default=64)
    parser.add_argument('--lora_dropout', type=int, help='Dropout parameter for LoRA optimization', default=0.05)
    args = parser.parse_args()

    # Load the base model and tokenizer
    model, tokenizer = load_base_model()

    # Load and preprocess the data
    tokenized_train_dataset, tokenized_val_dataset = load_data()

    # Configure the model
    model, config = config_model(model)

    # Train the model
    train_model(tokenized_train_dataset, tokenized_val_dataset, model, config) 
