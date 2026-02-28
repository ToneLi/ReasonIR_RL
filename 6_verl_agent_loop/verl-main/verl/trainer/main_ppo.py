# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket
from collections import defaultdict
import hydra
import ray
from omegaconf import OmegaConf
import re
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_extern_object
from tqdm import tqdm

import torch
import math

def extract_steps(trajectory: str, max_steps: int = 5):
    """Extract per-step info blocks from a trajectory string.

    Each step is expected to be either:
    1) <reason>...</reason> + <summary>...</summary> + <information>...</information>
    2) <reason>...</reason> + <satisfy> yes </satisfy>
    Returns a list of dicts with keys: reason, summary, information, satisfy.
    """

    # Accept three shapes per step:
    # 1) <reason>...</reason><summary>...</summary><information>...</information>
    # 2) <reason>...</reason><summary>...<information>...</information>   (missing </summary>)
    # 3) <reason>...</reason><satisfy>...</satisfy>
    # Allow literal "\n" sequences between tags (the trajectories are JSON-escaped).
    pattern = re.compile(
        r"<reason>(?P<reason>.*?)</reason>(?:\\n|\s)*"  # whitespace or literal \n between tags
        r"(?:"
        r"<summary>(?P<summary1>.*?)</summary>\s*<information>(?P<info1>.*?)</information>"  # well-formed
        r"|<summary>(?P<summary2>.*?)<information>(?P<info2>.*?)</information>"               # missing </summary>
        r"|<satisfy>(?P<satisfy>.*?)</satisfy>"                                              # satisfy branch
        r")",
        flags=re.DOTALL,
    )

    steps = []
    for match in pattern.finditer(trajectory):
        groups = match.groupdict()
        reason = groups.get("reason", "")
        summary = groups.get("summary1") or groups.get("summary2")
        information = groups.get("info1") or groups.get("info2")
        satisfy = groups.get("satisfy")
        steps.append(
            {
                "reason": reason.strip(),
                "summary": summary.strip() if summary else None,
                "information": information.strip() if information else None,
                "satisfy": satisfy.strip() if satisfy else None,
            }
        )
        if len(steps) >= max_steps:
            break
    return steps

def parse_running_context(running_context: str):
    """Parse query, input docs, and trajectory from the running context block."""

    input_section = ""
    output_section = ""

    input_match = re.search(
        r"INPUT BEGINS(.*?)(?:MODEL OUTPUT BEGINS|$)",
        running_context,
        flags=re.DOTALL,
    )
    if input_match:
        input_section = input_match.group(1)

    # output_match = re.search(
    #     r"=== MODEL OUTPUT BEGINS ======(.*)",
    #     running_context,
    #     flags=re.DOTALL,
    # )
    # if output_match:
    #     output_section = output_match.group(1)

    output_section=running_context.split("MODEL OUTPUT BEGINS")[1]
    query = ""
    query_match = re.search(r"<query>(.*?)</query>", input_section, flags=re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()

    info_matches = re.findall(r"<information>(.*?)</information>", input_section, flags=re.DOTALL)
    input_docs = "\n\n".join(s.strip() for s in info_matches) if info_matches else ""

    trajectory = output_section.strip()

    return query, input_docs, trajectory

def extract_expand_query(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]
    # fw1.write("query"+solution_str+"\n")
    # with open("output_check_steps.json", "a") as fw: 
    #     fw.write(solution_str+"\n")
    #     fw.write("--------------------------"+"\n")
    #     fw.flush()
    query, input_docs, trajectory = parse_running_context(solution_str)
    # fw1.write("query"+input_docs+"\n")
    # fw1.flush()
    # with open("output_check_steps_trajectory.json", "a") as fw: 
    #     fw.write(str(trajectory)+"\n")
    #     fw.write("--------------------------"+"\n")
    #     fw.flush()
    steps = extract_steps(trajectory)
    if len(steps)>=2:
        if steps[-1]["satisfy"] ==None:
            # print(steps[-1]["summary"])
            # fw1.write("summary"+steps[-1]["summary"]+"\n")
            # fw1.flush()
            if steps[-1]["summary"]!=None:
                if  len(steps[-1]["summary"])>20:
                    new_query=query+" "+ steps[-1]["summary"]
                else:
                    new_query=query+" "+ input_docs
                # print("-----",steps[-1]["summary"])
                # print("============summation--is nonne  -1 ")
                # with open("output_check_summay_1.json", "a") as fw: 
                #     fw.write(steps[-1]["summary"]+"\n")
                #     fw.write("--------------------------"+"\n")
                #     fw.flush()
            else:
                new_query=query+" "+ input_docs
        else:
            # print(steps[-2]["summary"])
            # fw1.write("summary"+steps[-2]["summary"]+"\n")
            # fw1.flush()
 
            if steps[-2]["summary"]!=None:  # remove ...
                # print("-----",steps[-2]["summary"])
                if  len(steps[-2]["summary"])>20:
                    new_query=query+" "+ steps[-2]["summary"]
                else:
                    new_query=query+" "+ input_docs
                # print("============summation--is nonne  -2 ")
                # with open("output_check_summay_2.json", "a") as fw: 
                #     fw.write(steps[-2]["summary"]+"\n")
                #     fw.write("--------------------------"+"\n")
                #     fw.flush()
            else:
                new_query=query+" "+ input_docs
    
    else:
        new_query=query+" "+ input_docs
        

    return new_query


def dcg_at_k(relevances, k=10):
    """relevances: list of relevance scores in ranked order"""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        dcg += (2**rel - 1) / math.log2(i + 2)   # i starts from 0, so rank = i+1
    return dcg


def number_duplicate_qids(q_ids):
    counter = defaultdict(int)
    new_qids = []

    for qid in q_ids:
        idx = counter[qid]
        new_qids.append(f"{qid}_{idx}")
        counter[qid] += 1

    return new_qids



def _batch_search(queries, search_host_url,question_ids=None, tasks=None):
        
        """Batchified search for queries."""
        final_scores = {}
        i=0
        task2datas = defaultdict(lambda: {
            "q_ids": [],
            "questions": []
        })

      
        for qid, query, task in zip(question_ids, queries, tasks):
            task2datas[task]["q_ids"].append(qid)
            task2datas[task]["questions"].append(query)

        for task, id_query in tqdm(task2datas.items()):
            ids=number_duplicate_qids(id_query["q_ids"])
            path_excluded_ids = {qid: ["N/A"] for qid in ids}
            payload = {
                "task": task,
                "q_id_list":  ids,
                "q_text_list": id_query["questions"],
                "excluded_ids": path_excluded_ids,
                "num_hits": 100,
            }
            data=requests.post(search_host_url, json=payload).json()
            id_doc_scores = data["scores"]
            for inst_id, (qid, docs_score) in enumerate(id_doc_scores.items()):
                i=i+1
                final_scores[(task, qid, inst_id)] = docs_score
        return final_scores

def ndcg_at_k(retrieved_scores: dict, ground_truth_ids: list, k=10):
    """
    retrieved_scores: dict {doc_id: score} (higher is better)
    ground_truth_ids: list of relevant doc_ids
    k: cutoff (default 10)

    Uses binary relevance: rel=1 if doc_id in ground_truth_ids else 0
    """
    # Sort retrieved docs by score descending
    ranked_docs = sorted(retrieved_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs[:k]]

    # Compute relevance list (binary)
    relevances = [1 if doc_id in ground_truth_ids else 0 for doc_id in ranked_doc_ids]

    # DCG
    dcg = dcg_at_k(relevances, k)

    # IDCG (ideal ranking: all relevant docs first)
    ideal_relevances = [1] * min(len(ground_truth_ids), k)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg
 




@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if need_reference_policy(config) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "veomni":
            raise NotImplementedError("VeOmni does not support legacy worker implementation")

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            from verl.workers.megatron_workers import CriticWorker

        elif config.critic.strategy == "veomni":
            if use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward_model.nnodes = config.trainer.nnodes
            config.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            # we do not use reward model workers, so we only register reward model in resource pool
            # without continue to register reward model worker in role mapping
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker group.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0,search_host_url="12.00.84.32")   #config.retriever.url
        # # config.retriever.url

        # # Note that we always use function-based RM for validation
        # val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1,search_host_url="12.00.84.32")
        

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            # reward_manager=reward_fn,
            # val_reward_manager=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config)

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
