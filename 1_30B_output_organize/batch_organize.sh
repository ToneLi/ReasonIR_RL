#!/bin/bash

# Batch processing script for organizing 30B outputs
# Usage: bash batch_organize.sh

cd "$(dirname "$0")"

echo "================================"
echo "30B Output Organization Batch"
echo "================================"

# Define input/output pairs
declare -a roots=(
    "/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/0_reasoning_step_generation_part12/output_parallel_part_12"
    "/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/0_reasoning_step_generation_part11/output_parallel_part_11"
    "/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/0_reasoning_step_generation_part10/output_parallel_part_10"
    # "/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/output_parallel_part1"
)

declare -a outputs=(
    # "30B_LLM_dynamic_c.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part12.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part11.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part10.jsonl"
)

# Process each pair
for i in "${!roots[@]}"; do
    root_dir="${roots[$i]}"
    output_file="${outputs[$i]}"
    
    echo ""
    echo "[$((i+1))/${#roots[@]}] Processing part..."
    echo "  Input:  $root_dir"
    echo "  Output: $output_file"
    
    if [ ! -d "$root_dir" ]; then
        echo "  ✗ ERROR: Directory not found: $root_dir"
        continue
    fi
    
    python organize_output.py --root_dir "$root_dir" --output_file "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ SUCCESS"
    else
        echo "  ✗ FAILED"
    fi
done

echo ""
echo "================================"
echo "Batch processing complete!"
echo "================================"
