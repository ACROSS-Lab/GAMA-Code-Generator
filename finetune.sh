#!/bin/bash

# Define arguments
BASE_MODEL_ID="mistralai/Mistral-Instruct-7B-v0.2"
LOAD_IN_4BIT=true
BNB_4BIT_USE_DOUBLE_QUANT=true
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="bfloat16"
TRAIN_DATA_PATH="/home/phuong-anh/gama/data/train/json/total.json"
EVAL_DATA_PATH="/home/phuong-anh/gama/data/eval/eval.json"
WANDB_PROJECT="gamaft-total"
OUTPUT_DIR="./trained-model/llama/"
WARMUP_STEPS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_GPU_TRAIN_BATCH_SIZE=2
AUTO_FIND_BATCH_SIZE=true
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
MAX_STEPS=5
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=2e-5
FP16=true
OPTIM="paged_adamw_8bit"
LOGGING_STEPS=50
LOGGING_DIR='./logs'
SAVE_STRATEGY="steps"
SAVE_STEPS=5
EVALUATION_STRATEGY="steps"
EVAL_STEPS=50
DO_EVAL=true
OVERWRITE_OUTPUT_DIR=false
REPORT_TO="wandb"

# Run the Python script with the specified arguments
python ./train/train.py --base_model_id $BASE_MODEL_ID --load_in_4bit $LOAD_IN_4BIT --bnb_4bit_use_double_quant $BNB_4BIT_USE_DOUBLE_QUANT --bnb_4bit_quant_type $BNB_4BIT_QUANT_TYPE --bnb_4bit_compute_dtype $BNB_4BIT_COMPUTE_DTYPE --train_data_path $TRAIN_DATA_PATH --eval_data_path $EVAL_DATA_PATH --wandb_project $WANDB_PROJECT --output_dir $OUTPUT_DIR --warmup_steps $WARMUP_STEPS --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE --auto_find_batch_size $AUTO_FIND_BATCH_SIZE --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --gradient_checkpointing $GRADIENT_CHECKPOINTING --max_steps $MAX_STEPS --num_train_epochs $NUM_TRAIN_EPOCHS --learning_rate $LEARNING_RATE --fp16 $FP16 --optim $OPTIM --logging_steps $LOGGING_STEPS --logging_dir $LOGGING_DIR --save_strategy $SAVE_STRATEGY --save_steps $SAVE_STEPS --evaluation_strategy $EVALUATION_STRATEGY --eval_steps $EVAL_STEPS --do_eval $DO_EVAL --overwrite_output_dir $OVERWRITE_OUTPUT_DIR --report_to $REPORT_TO
