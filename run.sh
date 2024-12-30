#!/bin/bash

# Create weights directory if it doesn't exist
mkdir -p weights

# Enable detailed debugging information
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Use torchrun with detected GPU count
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py