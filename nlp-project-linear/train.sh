#!/bin/bash

### Environment Variables
GPU_ID=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 main.py