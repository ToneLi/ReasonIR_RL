#!/bin/bash

tasks = ( "dbpedia-entity" "fiqa" "nfcorpus" "scidocs" "scifact" "trec-covid" "webis-touche2020")
KEEP_PASSAGE_NUM=3
NUM_HITS=100
NUM_PATHS=1
MAX_ROUNDS=5

#  (1) need to change

 
DATASET_SOURCE="../data/Beir"
#EXAMPLES_PATH="../data_making/split_datasets/part_${PART}"
EXAMPLES_PATH="../data/Beir/excamples"
OUTPUT_FILE="Beir_30_diver"  # 30b_diver_beir, 
#  (2) need to change,   search server IP   
BATCH_SERVER_URL="http://172.16.34.22:8505/batch_retrieve"
#  (3) need to change   search server IP
TRUNCATE_URL="http://172.16.34.22:8505/truncate"
#  (4) need to change
SUMMARIZATION_BATCH_URL="http://localhost:8502/summrization"




LOGFILE="run_time_log_parallel.txt"
echo "==== Parallel Run started at $(date) ====" >> $LOGFILE

for DATASET in "${tasks[@]}"; do
    echo "Running task: $DATASET"
    echo "---- $DATASET ----" >> $LOGFILE

    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0,1 python3 zero_shot_parallel_main.py \
        --dataset_source ${DATASET_SOURCE} \
        --examples_path ${EXAMPLES_PATH} \
        --task ${DATASET} \
        --cache_dir cache/cache_${model_name} \
        --keep_passage_num ${KEEP_PASSAGE_NUM} \
        --num_hits ${NUM_HITS} \
        --Batch_SERVER_URL ${BATCH_SERVER_URL} \
        --Truncate_URL ${TRUNCATE_URL} \
        --summarization_batch_URL ${SUMMARIZATION_BATCH_URL} \
        --NUM_PATHS ${NUM_PATHS} \
        --MAX_ROUNDS ${MAX_ROUNDS} \
        --output_dir ./${OUTPUT_FILE}/${DATASET} \
        --overwrite_output_dir \
        --max_tokens 8192 
    

    END=$(date +%s)
    DIFF=$((END - START))

    echo "Time used: ${DIFF} seconds" >> $LOGFILE
    echo "Time used: $(printf "%d minutes %d seconds" $((DIFF/60)) $((DIFF%60)))" >> $LOGFILE
    echo "" >> $LOGFILE
done

echo "==== Parallel Run ended at $(date) ====" >> $LOGFILE
