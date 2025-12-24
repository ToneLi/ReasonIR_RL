#!/bin/bash

 

SCRIPT="$retriever_script.sh"
PART=4

# GPU 分配：Round -> GPU
declare -a GPUS=(1 1 2 3 4 5 6 7)

echo "Starting batch run for 8 rounds..."
echo "=================================="

# 同时启动所有 round
for ROUND in 0 1 2 3 4 5 6 7; do
    GPU=${GPUS[$ROUND]}
    
    echo "start Round $ROUND (GPU=$GPU)"
    
    # 后台运行脚本
    bash "$SCRIPT" --round $ROUND --part $PART --gpu $GPU &
done

echo ""
echo "=================================="
echo "All tasks have been started, waiting for completion..."
echo "=================================="

# Wait for all background processes to complete
wait

echo ""
echo "=================================="
echo "All 8 rounds have been completed!"
echo "=================================="
