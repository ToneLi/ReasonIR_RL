#!/bin/bash

# Batch generate score statistics for each part
cd /home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/0_evaluation/bright/Diver/Retriever

BASE_PATH="/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/0_evaluation/bright/Diver/Retriever/output"

echo "=========================================="
echo "Start generating score statistics for each part"
echo "=========================================="

# Process part_2
echo "Processing part_2..."
python get_score_for_each_query.py --part_num part_2 --base_path "$BASE_PATH" --output_file results_sorted_part_2_path_score.json

# Process part_3
echo "Processing part_3..."
python get_score_for_each_query.py --part_num part_3 --base_path "$BASE_PATH" --output_file results_sorted_part_3_path_score.json

# Process part_4
echo "Processing part_4..."
python get_score_for_each_query.py --part_num part_4 --base_path "$BASE_PATH" --output_file results_sorted_part_4_path_score.json

# Process part_5
echo "Processing part_5..."
python get_score_for_each_query.py --part_num part_5 --base_path "$BASE_PATH" --output_file results_sorted_part_5_path_score.json

echo "=========================================="
echo "All parts processed successfully!"
echo "=========================================="
echo "Generated files:"
ls -lh results_sorted_part_*.json
