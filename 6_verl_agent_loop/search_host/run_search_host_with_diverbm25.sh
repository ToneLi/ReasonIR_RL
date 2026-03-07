#!/bin/bash

export DATASET_SOURCE='/home/mingchen/3_Query_rewrite_RL/3_Diver-main/data/BRIGHT'
export MODEL_PATH='AQ-MedAI/Diver-Retriever-4B'
export MODEL_NAME='diver-retriever'
export CACHE_DIR='/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/cache/cache_diver-retriever'

CUDA_VISIBLE_DEVICES=1 python -m uvicorn search_host_with_diverbm25:app --host 0.0.0.0 --port 8510
