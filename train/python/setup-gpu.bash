#!/bin/bash

# Determine IP Address and Port
# export MASTER_ADDR=192.168.1.100
# export MASTER_PORT=12345

# Set Environment Variables for each GPU
for ((i=0; i<4; i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    export RANK=$i
    export WORLD_SIZE=4
    
    # Run your training script for each GPU
    python your_script.py &
done

# Wait for all processes to finish
wait

