#!/bin/bash

# Batch filter script for processing multiple parts
# Requires: filters.py to be in the current directory
# Processes parts 2, 3, 4, 5 and generates filtered positive trajectory files


echo "=========================================="
echo "Batch Filter - Processing Multiple Parts"
echo "=========================================="

for part in 2 3 4 5; do
    echo ""
    echo "Processing part_$part..."
    
    score_file="results_sorted_part_${part}_path_score.json"
    input_file="../1_30B_output_organize/30B_LLM_dynamic_8_rounds_output_part${part}.jsonl"
    output_file="30B_LLM_part${part}_pos.jsonl"
    
    # Check if score file exists
    if [ ! -f "$score_file" ]; then
        echo "  ERROR: Score file not found: $score_file"
        continue
    fi
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "  ERROR: Input file not found: $input_file"
        continue
    fi
    
    echo "  Running: python filters.py -p part_$part -s \"$score_file\" -i \"$input_file\" -o \"$output_file\""
    python filters.py -p "part_$part" -s "$score_file" -i "$input_file" -o "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Part $part processed successfully"
    else
        echo "  ✗ Part $part processing failed"
    fi
done

echo ""
echo "=========================================="
echo "Batch filtering completed!"
echo "Output files:"
echo "  - 30B_LLM_part2_pos.jsonl"
echo "  - 30B_LLM_part3_pos.jsonl"
echo "  - 30B_LLM_part4_pos.jsonl"
echo "  - 30B_LLM_part5_pos.jsonl"
echo "=========================================="
