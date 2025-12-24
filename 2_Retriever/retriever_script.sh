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

path="/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/data/BRIGHT/"
examples_path="/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/data_making/split_datasets/part_${PART}/"
generated_files="/mnt/data_218/home1/Cool_Chen/0_reasonIR_bright/0_evaluation_upload/30B_LLM_dynamic_8_rounds_oupput_part${PART}.jsonl"    

output_dir="output/part_${PART}_${MODEL}_reasoning_round${ROUND}"
dataset_source="xlangai/BRIGHT"

tasks=(biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems)

echo "=========================================="
echo "Running ROUND=$ROUND, PART=$PART, GPU=$GPU"
echo "=========================================="

for MODEL in ${models[@]}; do
    for TASK in ${tasks[@]}; do
        echo "Running task: $TASK (Round=$ROUND)"
        CUDA_VISIBLE_DEVICES=$GPU python main_run.py --round_number $ROUND --path $path --examples_path $examples_path --generated_files $generated_files --task $TASK --model $MODEL --dataset_source ${dataset_source} --output_dir ${output_dir}  --cache_dir ${dataset_source}/cache/cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS 
    done
done

echo "Round $ROUND completed!"
