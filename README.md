# ReasonIR_RL

### environment configuration
```
conda env create -f environment.yml
```

###  Model preparation
Please download these two models in your model files:
[Diver-Retriever-4B](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B)

[Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

###  Data preparation

download the data, data_making from [Google_data](https://drive.google.com/drive/folders/1BGbbE_qOVAzvhiXy1i3TBGwQZyG9Rnvu?usp=sharing), and put them under the 0_reasoning_step_generation

###  start the summrization_host  host
```
CUDA_VISIBLE_DEVICES=2,3 uvicorn summrization_host:app --host 0.0.0.0 --port 8502

2 A100, 78GB, 
```
###  start Search search_host_with_BM25
```
CUDA_VISIBLE_DEVICES=5 uvicorn search_host_with_BM25:app --host 0.0.0.0 --port 8505
80GB
```
###  run code:
```
bash run_parallel.sh
```



