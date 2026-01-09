import argparse
import os
from transformers import AutoTokenizer
from vllm_server.vllm_completion import VLLMCompletion
import asyncio
from fastapi import FastAPI
app = FastAPI()

generation_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
#generation_model="/data/home_beta/mingchen/3_DeepRetrieval/cold_start/output_dir/qwen3b_merged"
#"/home/mingchen/3_Query_rewrite_RL/3_Diver-main/0_model_train/sft_output_4b_Thinking_pos1/final"
# search_api = VectorSearchInterface(args, doc_ids, documents)

# generation model
no_thinking=False
openai_api = VLLMCompletion(model_name=generation_model, control_thinking=no_thinking)

@app.post("/summrization")
def summrization(req: dict):

    user_prompt_list = req["user_prompt_list"]
    gen_fn = openai_api.completion_chat_batch
    # messages = [{"role": "user", "content": user_prompt}]

    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in user_prompt_list
    ]

    response_list =  gen_fn(messages_batch, max_tokens=8192, temperature=1, top_p=1,top_k=-1, n=1) # 32768



    return {"response_list":response_list}
