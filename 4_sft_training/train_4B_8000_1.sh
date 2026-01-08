#!/bin/bash
# Qwen/Qwen3-4B-Thinking-2507
# cd /home/mingchen/3_Query_rewrite_RL/3_Diver-main/0_model_train

CUDA_VISIBLE_DEVICES=2 python unsloth_sft_trainer.py \
  --train_file data/converted_multiturn_right_path_8000_train.jsonl \
  --eval_file data/converted_multiturn_right_path_8000_dev.jsonl \
  --wandb_project 4B_8000_1 \
  --model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
  --output_dir ./sft_output_4b_8000_1  \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --max_seq_length 8192 \
  --report_to wandb \
  --visualize_masking
