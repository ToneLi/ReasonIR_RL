#!/bin/bash

# 并行推理脚本 - 每个query生成16条并行推理路径
# "biology" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions"
#"biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops"
# tasks=( "theoremqa_theorems" "theoremqa_questions")
tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow"  "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions" )

KEEP_PASSAGE_NUM=3
NUM_HITS=100

LOGFILE="run_time_log_parallel.txt"
echo "==== Parallel Run started at $(date) ====" >> $LOGFILE

for DATASET in "${tasks[@]}"; do
    echo "Running task: $DATASET"
    echo "---- $DATASET ----" >> $LOGFILE

    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0,1 python3 zero_shot_parallel_main.py \
        --dataset_source ../data/BRIGHT \
        --task ${DATASET} \
        --cache_dir cache/cache_${model_name} \
        --keep_passage_num ${KEEP_PASSAGE_NUM} \
        --num_hits ${NUM_HITS} \
        --output_dir ./output_parallel_part0/${DATASET} \
        --overwrite_output_dir \
        --max_tokens 8192 
    

    END=$(date +%s)
    DIFF=$((END - START))

    echo "Time used: ${DIFF} seconds" >> $LOGFILE
    echo "Time used: $(printf "%d minutes %d seconds" $((DIFF/60)) $((DIFF%60)))" >> $LOGFILE
    echo "" >> $LOGFILE
done

echo "==== Parallel Run ended at $(date) ====" >> $LOGFILE
