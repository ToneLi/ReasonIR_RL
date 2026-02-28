#!/bin/bash

export DATASET_SOURCE='/home/mingchen/3_Query_rewrite_RL/3_Diver-main/data/BRIGHT'
export MODEL_PATH='AQ-MedAI/Diver-Retriever-4B'
export MODEL_NAME='diver-retriever'
export CACHE_DIR='/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/cache/cache_diver-retriever'

CUDA_VISIBLE_DEVICES=1 uvicorn search_host_with_diver:app --host 0.0.0.0 --port 8516
