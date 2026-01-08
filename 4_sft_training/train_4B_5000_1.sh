#!/bin/bash
# Qwen/Qwen3-4B-Thinking-2507
CUDA_VISIBLE_DEVICES=0 python unsloth_sft_trainer.py \
  --train_file data/converted_multiturn_right_path_5000_train.jsonl \
  --eval_file data/converted_multiturn_right_path_5000_dev.jsonl \
  --model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
  --output_dir ./sft_output_4b_5000_1 \
  --num_train_epochs 2 \
  --wandb_project 4B_5000_1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --max_seq_length 8192 \
  --report_to wandb \
  --visualize_masking
