from vllm import LLM, SamplingParams
import os
from time import sleep
from transformers import AutoTokenizer
import torch
# from promts_llm_think import get_prompt
import requests
import pytrec_eval



SERVER_URL = "http://localhost:8505/retrieve"
Truncate_URL = "http://localhost:8505/truncate"
summarization_URL = "http://localhost:8502/summrization"



class VLLMCompletion(object):
    def __init__(
            self,
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            eos_token="<|endoftext|>",
            api_key=None,  # kept for interface compatibility
            control_thinking=False
    ):
        # self.build_prompt=get_prompt
        self.model_name = model_name
        self.eos_token = eos_token
        self.control_thinking = control_thinking
        self.max_tokens = 32768
        self.temperature = 1
        self.top_p = 1
        self.n = 1
        self.stop_tokens = ["</summary>", "</satisfy>"]

        # Initialize vLLM engine
        MY_GPU_COUNT = torch.cuda.device_count()
        print(f"MY_GPU_COUNT: {MY_GPU_COUNT}")
        
        # Initialize vLLM using the new LLM class
        self.engine = LLM(
            model=model_name,
            gpu_memory_utilization=0.95,
            max_model_len= 16384, # 162144   16384  32768
            tensor_parallel_size=MY_GPU_COUNT,
            dtype="bfloat16",
            # tensor_parallel_size=1,
            # disable_custom_all_reduce=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def truncate_text(self, doc_text, max_tokens=16384):
       # encode doc
        doc_ids = self.tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_ids) > max_tokens:
            doc_ids = doc_ids[:max_tokens]

        return self.tokenizer.decode(doc_ids, skip_special_tokens=True)

    def _generate(self, prompt_list, sampling_params):
        # prompt = self.truncate_text(prompt, max_tokens=16000)

        truncated_prompts = [
            self.truncate_text(p, max_tokens=16000)
            for p in prompt_list
        ]

        outputs = self.engine.generate(truncated_prompts, sampling_params)

        # results = []
        # for output in outputs:
        #     for generated_output in output.outputs:
        #         results.append(generated_output.text)
        # return results
        #
        batch_results = []

        for output in outputs:
            single_prompt_results = []
            for generated_output in output.outputs:  # n
                single_prompt_results.append(generated_output.text)
            batch_results.append(single_prompt_results)

        return batch_results



    def completion_chat(self, messages, top_passages=None, max_tokens=8192, temperature=1, top_p=1, top_k=-1,n=1):
        print(f"args: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, n={n}")
        def parse_messages(messages):
            if self.control_thinking:
                print('Adding no-thinking messages.')
                base_message = self.tokenizer.decode(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                return base_message + 'Okay, I think I have finished thinking.' + "\n</think>\n"
            else:
                return self.tokenizer.decode(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            top_k=top_k,
            stop=["</summary>","</satisfy>"]
        )

        prompt = parse_messages(messages)
        # print(prompt)
        print("----parse_messages   is finished----")
        responses = self._generate(prompt, sampling_params)

        return responses

 
    def completion_chat_batch(self, messages_batch, top_passages=None, max_tokens=32768, temperature=1,top_k=-1, top_p=1, n=1):
        print(f"args: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, n={n}")
        def parse_messages(messages):
            if self.control_thinking:
                print('Adding no-thinking messages.')
                base_message = self.tokenizer.decode(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                return base_message + 'Okay, I think I have finished thinking.' + "\n</think>\n"
            else:
                return self.tokenizer.decode(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))


        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            stop=["</summary>","</satisfy>"]
        )
        promt_list=[]
        for messages in messages_batch:
            prompt = parse_messages(messages)
            promt_list.append(prompt)

        # prompt = parse_messages(messages)
        # print(f"INPUT TO the model: {prompt}")
        get_result = False
        while not get_result:
            try:
                responses = self._generate(promt_list, sampling_params)
                get_result = True
            except:
                sleep(1)
                print('error in calling thinking LLM')

        return responses