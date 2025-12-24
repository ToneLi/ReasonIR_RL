## configration

```
cuda 12.9
flash attention (2.7.0),   

cuda 12.4
flash attention (2.6.0),   
```



# ReasonIR_RL
## Step 1
###
Aim: Generate reasoning paths for each question in data_making/split_datasets. Each file (e.g., Part 1) in this directory contains around 2,000 questions across 11 tasks (e.g., AoPS). Therefore, each part corresponds to approximately 2,000 questions, and we need reasoning paths for at least 8,000 questions in total. I am running Part 1, so you can run Parts 2, 3, 4, 5, 6, and 7.
Please only change the following parameters in run_parallel.sh:
```
EXAMPLES_PATH="data_making/split_datasets/part_X"
--output_dir ./output_parallel_partX/${DATASET}
```
where X ∈ {2, 3, 4, 5, 6, 7, 8, ...}.
### environment configuration
```
conda env create -f environment.yml
```

###  Model preparation
Please download these two models in your model files:

[Diver-Retriever-4B](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B),  please change the model path in search_host_with_BM25.py. line 333
```
model_path = "AQ-MedAI/Diver-Retriever-4B"
```

[Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), please change the model path in summrization_host.py.
```
generation_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
```

###  Data preparation

1) download the data, data_making from [Google_data](https://drive.google.com/drive/folders/1BGbbE_qOVAzvhiXy1i3TBGwQZyG9Rnvu?usp=sharing), and place them under 0_reasoning_step_generation.
2) Download the [cache.zip](https://vadata.s3.us-east-2.amazonaws.com/cache.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMiJHMEUCIQCXuK4CwxeRwYupKhdNm2Lyjx59qptABcpaFDolE1K1LQIgc2V07VdtVYilSyed1JOpjapHYEwElenEQsn5mqq2FbEquQMIcxAAGgwyNTM0OTA3ODU5ODEiDGOD64BnYWgH6LxPZiqWAzAeAdiJQ3g%2FNW1ZKbP%2B7HWl5QOqmOlJZ%2FE3JUJAt0Jrb7tr66sqvx3ZbtrUOQU8eKnOZm5xlteS793K%2FSkSmz%2FWME4A08xxgbXtlaPlBe29a%2FHouTwF2MT1aM5UKujsmOOdmXWeugAwFZSDhwB%2ByM5e%2Bnpu11dkQPBMZx%2BI%2F%2FEUPDDTxdFZg%2FI0ij%2FU4TuN3cA7rcI3%2FHc5MkvJg%2B4IXgymF8VisxUinxATwxJllboYJ6gHuy0sLToirJxHgrWMPFSrpNbCI44FNvtRJZjjn9ZDXnXol3VyVKBiEqj0VyhJMY1jPGAQMGf1EV50pAwhnShI4bjpuJcYirkbFuuzLGQ2sQWTXRtdKLCChdiCZN3sFvptyHQ2%2BqAb8X4qWIGP5mtnRVd0KlKSoY3B%2FNboSibg26CCx5bKxdoMW2f0N1jx7il%2BUMUDNEaJ8gBupHUwQ9B%2F5ANxRSx7jDPtsP1RfvplSssybvFCn2fJMlurWpeg8zZceqcguabR86yTgC1aeMtGKoFdgwewnV2sgpSlWWM8jZugnNow7aiGygY63gK%2BLo6G3wGTpkGe2%2FG48lkeySQEFVkLBQEa6OyLujQZvVMr3DxQWUf6qTXw06m7pgSDqVRS3kI2Ygzhxi4S9DJXHUctCFh4vhYJzWdUrFns8Zx4mLgqHQSd4scLTp7tZJNfc5%2FmkdEyBR2lHlPlYkMu95DD01TWA0RXCFPU4XduOjtomVPKJoJORERsVHqnLXvshZmMOImoj6G7iHDzR2TvjZR6cx80VKg5AVPOZxelVBoJCaq0PuPZDJA7U%2B1SMQRgcMpVYeUHHWc5t1Eon5XffTi18WWWdTS8lqHvjLHegRsLJmEEC7ZReCdeWHFOma96q0nRIGVYlWX3NvkBGCwNSdMZkkrQVV%2F3dwuHf2YoEiU8UDaXGspdkb9Qkhfs5zbwA6nhtDbYe9JXJfxooBYwpoOGSXZf%2FeOPIcTRzyngSgKASTrZjaFQe1HeADoMMGkagCjA1sYgCgM%2BEU6l6w%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIATWBJ2N2663W7QSK5%2F20251217%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20251217T021618Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=54930e8f99d914ce4dc492b397021bc2658cd0bb95b9c30d21632f82270e44d1) and unzip it and  and place them under 0_reasoning_step_generation.

###  start the summrization_host  host
```
CUDA_VISIBLE_DEVICES=2,3 uvicorn summrization_host:app --host 0.0.0.0 --port 8502

2 A100, 78GB, 
```
###  start Search search_host_with_BM25
```
CUDA_VISIBLE_DEVICES=5 uvicorn search_host_with_BM25:app --host 0.0.0.0 --port 8505
1 A100 around 8000MB
```
###  run code:
```
bash run_parallel.sh
```

## Step 2
Go to 1_30B_output_organize and run bash_organize.sh.
The purpose of this step is to extract the expanded queries for each original query. Each query has 8 trajectories, and each trajectory provides its own expanded query.

This step is used to compute the NDCG score for each trajectory. Specifically, NDCG is calculated between the original query and the expanded query generated in each trajectory, allowing us to evaluate the retrieval quality of different trajectories.

The output will be the dict for each part, 
```
declare -a roots=(
    "./0_reasoning_step_generation/output/diver_output_2"
    "./0_reasoning_step_generation/output/diver_output_3"
    "./0_reasoning_step_generation/output/diver_output_4"
    "./0_reasoning_step_generation/output/diver_output_5"
)

declare -a outputs=(
    # "30B_LLM_dynamic_c.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part2.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part3.jsonl"
    "30B_LLM_dynamic_8_rounds_output_part4.jsonl"
      "30B_LLM_dynamic_8_rounds_output_part5.jsonl"
)

```
####

Unzip doc_id.zip and place the extracted files in
0_reasoning_step_generation/cache/cache_diver-retriever.
Make sure that doc_emb and doc_id are located in the same directory.
