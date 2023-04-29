#!/bin/bash

### Environment Variables
GPU_ID=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/bin/python3.8 main.py