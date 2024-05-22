import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
from datetime import datetime
from datasets import load_dataset

import argparse
import os

def load_base_model(base_model_id, load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )

    # Load the model from the Hugging Face Hub
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load the tokenizer from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side='left',
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return base_model, tokenizer

def load_data(train_data_path, eval_data_path, tokenizer):
    # Load training and evaluation datasets
    train_dataset = load_dataset('json', data_files=train_data_path, split='train')
    print(train_dataset)
    eval_dataset = load_dataset('json', data_files=eval_data_path, split='train')
    print(eval_dataset)

    tokenized_train_dataset = train_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    tokenized_val_dataset = eval_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

    return tokenized_train_dataset, tokenized_val_dataset

def formatting_func(sample):
    bos_token = "<s>"
    instruct_message = "[INST] You are a chatbot who can learn and generate GAML code! You will help us generate GAML code snippet based on given question.[/INST]</s>"
    question = sample["question"].replace("\n\n### Question\n", "").strip()
    answer = sample["answer"].replace("\n### Response\n", "").strip()
    eos_token = "</s>"

    full_prompt = bos_token + instruct_message + "\n[INST]" + question + " [/INST]\n\n" + answer + eos_token

    return full_prompt

def generate_and_tokenize_prompt(prompt, tokenizer):
    # Generate and tokenize prompts
    max_length = 1700
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def config_model(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # Configure LoRA model
    config = LoraConfig(
        # Use DoRA for finetuning
        use_dora=True,  # Comment this if occur an error
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate",
            "w1",
            "w2",
            "w3",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Get the LoRA model
    model = get_peft_model(model, config)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)
    return model

def train_model(tokenized_train_dataset, tokenized_val_dataset, model, tokenizer, args):
    # Set device to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Specify GPU index for tokenized_train_dataset
    gpu_indices = [0, 1]  # Choose the GPU indices you want to use
    for id in tokenized_train_dataset.keys():
        tokenized_train_dataset[id] = tokenized_train_dataset[id].to(f"cuda:{gpu_indices[0]}")  # Move dataset to the first GPU

    for id2 in tokenized_val_dataset.keys():
        tokenized_val_dataset[id2] = tokenized_val_dataset[id2].to(f"cuda:{gpu_indices[1]}")  # Move dataset to the second GPU 

    # Set up DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model.is_parallelizable = True
        model.model_parallel = True

    # Use wandb to log down the checkpoint
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            warmup_steps=args.warmup_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_gpu_train_batch_size=args.per_gpu_train_batch_size,
            auto_find_batch_size=args.auto_find_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            optim=args.optim,
            logging_steps=args.logging_steps,
            logging_dir=args.logging_dir,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            do_eval=args.do_eval,
            overwrite_output_dir=args.overwrite_output_dir,
            report_to=args.report_to,
            run_name=f"{args.output_dir}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Disable caching for training
    model.config.use_cache = False
    trainer.train()    

    # Push to hub
    hub_model_id = "Phanh2532/GAMA-Code-generator-v2.0"
    trainer.model.push_to_hub(hub_model_id, use_temp_dir=False, token="")
    tokenizer.push_to_hub(hub_model_id, use_temp_dir=False, token="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model ID")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load model in 4-bit")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True, help="Use double quantization in 4-bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type in 4-bit")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="Compute dtype in 4-bit")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to the evaluation data JSON file")
    parser.add_argument("--wandb_project", type=str, default="", help="Weights and Biases project name")
    parser.add_argument("--output_dir", type=str, default="./trained-model/llama/", help="Directory to save the trained model")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=2, help="Batch size per GPU during training")
    parser.add_argument("--auto_find_batch_size", type=bool, default=True, help="Auto find batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--max_steps", type=int, default=5, help="Total number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--fp16", type=bool, default=True, help="Use 16-bit floating point precision")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=5, help="Save checkpoint every X steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=50, help="Run an evaluation every X steps")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run evaluation during training")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False, help="Overwrite the output directory")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report to (e.g., 'wandb')")

    args = parser.parse_args()

    # Load the base model and tokenizer
    base_model, tokenizer = load_base_model(
        args.base_model_id,
        args.load_in_4bit,
        args.bnb_4bit_use_double_quant,
        args.bnb_4bit_quant_type,
        args.bnb_4bit_compute_dtype
    )

    # Load and preprocess the data
    tokenized_train_dataset, tokenized_val_dataset = load_data(args.train_data_path, args.eval_data_path, tokenizer)

    # Configure the model
    model = config_model(base_model)

    # Train the model
    train_model(tokenized_train_dataset, tokenized_val_dataset, model, tokenizer, args)
