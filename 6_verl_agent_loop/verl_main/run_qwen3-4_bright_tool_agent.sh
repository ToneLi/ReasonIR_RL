#!/bin/bash
# BRIGHT tool-agent GRPO training on 2xGPU
# Usage: nohup bash run_qwen3-4_bright_tool_agent.sh > run_bright.log 2>&1 &
set -x

# Activate the verl conda environment (Python 3.12, required for | union syntax)
# source /home/mingchen/anaconda3/etc/profile.d/conda.sh
# conda activate verl

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# ---- Isolate this Ray cluster from other jobs on the same machine ----
# Use a unique port so ray stop/init never touches the other job's cluster
export RAY_PORT=6390
export RAY_DASHBOARD_PORT=8275

export RAY_worker_register_timeout_seconds=600
export BRIGHT_TASKS='["biology","earth_science","economics","psychology","robotics","stackoverflow","sustainable_living","leetcode","pony","aops","theoremqa_theorems","theoremqa_questions"]'
export BRIGHT_CACHE_DIR="/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/cache/cache_diver-retriever"
export BRIGHT_BASE_DOC_DIR="/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/0_evaluation/bright/Diver/Retriever/data/BRIGHT/document"
export RAY_max_pending_calls_per_actor=10
export BRIGHT_MAX_DOC_TOKENS=300
export BRIGHT_RETRIEVAL_URL="http://172.16.34.29:8515/batch_retrieve"

CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='bright_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/experiment_data_gamma/mingchen/models/final \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.rollout_data_dir="$PROJECT_DIR/rollout_data_batch" \
    trainer.project_name='bright_tool-agent' \
    trainer.experiment_name='qwen3-4b-think_bright_our_sft_32' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.total_training_steps=320 \
    trainer.default_local_dir=verl_checkpoints \
    trainer.test_freq=50 \
    data.train_files="$PROJECT_DIR/data_progress/bright/part_48_49_50_rounds_train.parquet" \
    data.val_files="$PROJECT_DIR/data_progress/bright/part_48_49_50_rounds_dev.parquet" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/bright_tool_config.yaml" \
    trainer.val_before_train=True \
    trainer.total_epochs=7 $@
