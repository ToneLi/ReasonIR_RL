#! /bin/bash -i

# 通用 retriever 脚本 - 接收命令行参数
# 使用方式：bash retriever_script.sh --round 0 --part 1 --gpu 5

# 默认值
ROUND=0
PART=1
GPU=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --round)
            ROUND="$2"
            shift 2
            ;;
        --part)
            PART="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


models=(diver-retriever)
REASONING="30B_LLM_part${PART}_round${ROUND}"
BS=-1

path="../0_reasoning_step_generation/data/BRIGHT/"
examples_path="../0_reasoning_step_generation/data_making/split_datasets/part_${PART}/"
generated_files="1_30B_output_organize/30B_LLM_dynamic_8_rounds_oupput_part${PART}.jsonl"    

output_dir="output/part_${PART}_${MODEL}_reasoning_round${ROUND}"
dataset_source="../0_reasoning_step_generation/data/BRIGHT/"
cache_dir="../0_reasoning_step_generation/cache/cache_diver-retriever"
tasks=(biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems)

echo "=========================================="
echo "Running ROUND=$ROUND, PART=$PART, GPU=$GPU"
echo "=========================================="

for MODEL in ${models[@]}; do
    for TASK in ${tasks[@]}; do
        echo "Running task: $TASK (Round=$ROUND)"
        CUDA_VISIBLE_DEVICES=$GPU python main_run.py --round_number $ROUND --path $path --examples_path $examples_path --generated_files $generated_files --task $TASK --model $MODEL --dataset_source ${dataset_source} --output_dir ${output_dir}  --cache_dir ${cache_dir} --reasoning $REASONING --encode_batch_size $BS 
    done
done

echo "Round $ROUND completed!"
