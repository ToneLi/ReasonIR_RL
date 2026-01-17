import argparse
import os
from transformers import AutoTokenizer
from vllm_server.vllm_completion import VLLMCompletion
import openai, json
from utils.common import save_json, save_json_dict_format
from datasets import load_dataset
import re
import torch.distributed as dist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import requests
import pytrec_eval
import aiohttp
import asyncio
from promts_llm_think import get_prompt



print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


def parse_agent_output(text):
    """Parse LLM output to determine if need to continue retrieval (summary) or satisfied (satisfy)"""
    text_lower = text.lower()
    
    # Find the last <reason> block
    idx_reason_start = text_lower.rfind("<reason>")
    idx_reason_end = text_lower.rfind("</reason>")
    
    reason_block = ""
    if idx_reason_start != -1 and idx_reason_end != -1:
        reason_block = text_lower[idx_reason_start : idx_reason_end + len("</reason>")]
    
    # Rule 1: Check if <satisfy> is in reason block, and no <summary> after it
    if "satisfy" in reason_block and "<summary>" not in text_lower[idx_reason_end + len("</reason>"):]:
        return "satisfy", "done"
    
    # Rule 2: Find the last summary and satisfy
    idx_summary = text_lower.rfind("<summary>")
    idx_satisfy = text_lower.rfind("<satisfy>")
    
    # Only summary
    if idx_summary != -1 and (idx_satisfy == -1 or idx_summary > idx_satisfy):
        content = text[idx_summary + len("<summary>"):].strip()
        # Extract until </summary> or end of text
        end_idx = content.lower().find("</summary>")
        if end_idx != -1:
            content = content[:end_idx].strip()
        return "summary", content
    
    # Only satisfy
    if idx_satisfy != -1 and (idx_summary == -1 or idx_satisfy > idx_summary):
        content = text[idx_satisfy + len("<satisfy>"):].strip()
        return "satisfy", content
    
    # Unrecognized
    return "unknown", text


def search_iterator(args, qid_query_list, excluded_ids, original_qid=None):
    """
    Batch retrieval function
    
    Args:
        args: Arguments
        qid_query_list: [(qid, query_text), ...] - qid can be path_id or original qid
        excluded_ids: Original excluded_ids dict, keys are original query ids
        original_qid: If qid in qid_query_list is path_id, need to provide original qid
    """
    n = len(qid_query_list)
    batch_size = 64
    
    final_scores = {}
    for i in range(0, n, batch_size):
        batch = qid_query_list[i:i + batch_size]
        
        q_ids = [qid for qid, _ in batch]
        q_texts = [qtext for _, qtext in batch]
        
        # Create excluded_ids mapping for path_id
        # If qid is path_id (e.g., "query_0_path_1"), extract original qid
        path_excluded_ids = {}
        for qid in q_ids:
            if original_qid is not None:
                # Multi-path mode: all paths use the same original qid's excluded_ids
                path_excluded_ids[qid] = excluded_ids.get(original_qid, [])
            else:
                # Initialization mode: qid is the original qid
                path_excluded_ids[qid] = excluded_ids.get(qid, [])
        
        payload = {
            "task": args.task,
            "q_id_list": q_ids,
            "q_text_list": q_texts,
            "excluded_ids": path_excluded_ids,
            "num_hits": args.num_hits,
        }
        
        try:
            resp = requests.post(args.Batch_SERVER_URL, json=payload) # , timeout=300
            resp.raise_for_status()  # Check HTTP errors
            
            # Try to parse JSON
            try:
                data = resp.json()
            except requests.exceptions.JSONDecodeError as e:
                print(f"\n!!! JSON Parse Error !!!")
                print(f"Status Code: {resp.status_code}")
                print(f"Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
                print(f"Response Length: {len(resp.text)} characters")
                print(f"First 1000 characters of response:\n{resp.text[:1000]}")
                print(f"Last 500 characters of response:\n{resp.text[-500:]}")
                raise
            
            id_doc_scores = data["scores"]
            
            for qid, docs_score in id_doc_scores.items():
                final_scores[qid] = docs_score
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response Status Code: {e.response.status_code}")
                print(f"Response Content: {e.response.text[:500]}")
            raise
    
    return final_scores


class ParallelReasoningPath:
    """State of a single reasoning path"""
    def __init__(self, path_id, qid, original_query, initial_docs):
        self.path_id = path_id
        self.qid = qid
        self.original_query = original_query
        self.running_context = ""
        self.status = "active"  # active, satisfied, unknown
        self.round_num = 0
        self.last_summary = None
        
        # States for document filtering
        self.seen_passages = set()
        self.last_top_k_passages = set()
        self.last_last_top_k_passages = set()
        
        # Initialize context
        method = "think_prompt"
        initialized_document = ",".join(list(initial_docs))
        self.running_context = get_prompt(method, original_query, initialized_document)
    
    def update_context(self, llm_output):
        """Update context and parse status"""
        self.running_context += "\n" + llm_output + "\n"
        status, content = parse_agent_output(llm_output)
        
        if status == "satisfy":
            self.status = "satisfied"
            self.running_context += " <satisfy> yes </satisfy>"
            return None  # No retrieval needed
        elif status == "summary":
            self.status = "active"
            self.last_summary = content
            self.round_num += 1
            return content  # Return query that needs retrieval
        else:
            self.status = "unknown"
            return None
    
    def add_retrieved_docs(self, docs, keep_passage_num):
        """Add retrieved documents, using filtering logic to avoid duplicates"""
        # Move previous round's top passages to two rounds ago
        self.last_last_top_k_passages = self.last_top_k_passages
        # Store current round's top passages
        self.last_top_k_passages = set(docs[:keep_passage_num])
        
        # Fast filtering with combined blacklist
        filtered_passages = []
        
        for passage in docs:
            # Skip if already in permanent blacklist
            if passage in self.seen_passages:
                continue
            
            # If in last_last round, add to permanent blacklist and skip
            if passage in self.last_last_top_k_passages:
                self.seen_passages.add(passage)
                continue
            
            # New passage - add it
            filtered_passages.append(passage)
            if len(filtered_passages) >= keep_passage_num:
                break

        #retrieved_docs = docs[:keep_passage_num]
        retrieved_docs = filtered_passages[:keep_passage_num]
        info_block = "<information>\n" + ", ".join(retrieved_docs) + "\n</information>"
        self.running_context += info_block


def run_parallel_reasoning_agent(args, qid, original_query, initial_docs_set, did2content, excluded_ids):
    """
    Run 16 parallel reasoning paths for a single query
    
    Args:
        args: Arguments
        qid: query id
        original_query: Original query
        initial_docs_set: Initial retrieved top-k documents set
        did2content: Document id to content mapping
        excluded_ids: Excluded document ids
    
    Returns:
        Final context list of all paths
    """
    # Step 1: Initialize 16 paths
    paths = []
    for i in range(args.NUM_PATHS):
        path = ParallelReasoningPath(
            path_id=f"{qid}_path_{i}",
            qid=qid,
            original_query=original_query,
            initial_docs=initial_docs_set
        )
        # Initialize last_top_k_passages with initial documents
        path.last_top_k_passages = initial_docs_set
        paths.append(path)
    
    print(f"\n[Query {qid}] Initialize {args.NUM_PATHS} parallel paths")
    
    # Step 2: Iterative processing
    for round_num in range(args.MAX_ROUNDS):
        print(f"\n[Query {qid}] ===== Round {round_num + 1}/{args.MAX_ROUNDS} =====")
        
        # 2.1 Collect all active paths
        active_paths = [p for p in paths if p.status == "active"]
        
        if len(active_paths) == 0:
            print(f"[Query {qid}] All paths completed, early termination")
            break
        
        print(f"[Query {qid}] Number of active paths: {len(active_paths)}/{args.NUM_PATHS}")
        
        # 2.2 Batch generation: send all active paths' contexts to LLM
        batch_contexts = [p.running_context for p in active_paths]
        
        # Batch call to LLM
        try:
            payload = {"user_prompt_list": batch_contexts}
            resp = requests.post(args.summarization_batch_URL, json=payload) # , timeout=600
            resp.raise_for_status()
            data = resp.json()
            llm_outputs = data["response_list"]  # Should return a list
            
            print(f"[Query {qid}] LLM batch generation completed, received {len(llm_outputs)} outputs")
        except requests.exceptions.JSONDecodeError as e:
            print(f"LLM batch generation JSON parse error: {e}")
            print(f"Response status code: {resp.status_code}")
            print(f"Response content: {resp.text[:500]}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"LLM batch generation request error: {e}")
            raise
        
        # 2.3 Analyze output: update each path's status
        paths_need_retrieval = []
        retrieval_queries = []
        
        for path, llm_output in zip(active_paths, llm_outputs):
            # print("llm_output",llm_output)
            summary_query = path.update_context(llm_output[0])
            
            if path.status == "satisfied":
                print(f"  Path {path.path_id} satisfied (round {path.round_num})")
            elif path.status == "active" and summary_query:
                paths_need_retrieval.append(path)
                # Use original qid instead of path_id, because excluded_ids uses original qid as key
                retrieval_queries.append((path.path_id, summary_query))
                print(f"  Path {path.path_id} needs retrieval")
            elif path.status == "unknown":
                print(f"  Path {path.path_id} status unknown, marked as completed")
        
        # 2.4 Batch retrieval: execute batch retrieval for paths that need to continue
        if len(paths_need_retrieval) > 0:
            print(f"[Query {qid}] Start batch retrieval for {len(paths_need_retrieval)} paths")
            
            # Batch retrieval
            try:
                # Pass original qid, let search_iterator create correct excluded_ids mapping for each path_id
                final_scores = search_iterator(args, retrieval_queries, excluded_ids, original_qid=qid)
                print(f"[Query {qid}] Batch retrieval completed, received {len(final_scores)} results")
            except Exception as e:
                print(f"[Query {qid}] Batch retrieval failed: {e}")
                # If batch retrieval fails, mark all paths as unknown
                for path in paths_need_retrieval:
                    path.status = "unknown"
                continue
            
            # 2.5 Assign retrieval results to respective paths
            for path in paths_need_retrieval:
                path_qid = path.path_id
                if path_qid in final_scores:
                    did_scores = final_scores[path_qid]
                    # Get all retrieved documents (not just top-k)
                    all_retrieved_docs = [did2content[did] for did, score in did_scores.items()]
                    # Add documents using filtering logic
                    path.add_retrieved_docs(all_retrieved_docs, args.keep_passage_num)
                    print(f"  Path {path.path_id} added filtered documents")
        else:
            print(f"[Query {qid}] No paths need retrieval")
    
    # Step 3: Return final results of all paths
    print(f"\n[Query {qid}] Completed all rounds")
    print(f"  Satisfied paths: {sum(1 for p in paths if p.status == 'satisfied')}/{args.NUM_PATHS}")
    print(f"  Active paths: {sum(1 for p in paths if p.status == 'active')}/{args.NUM_PATHS}")
    print(f"  Unknown paths: {sum(1 for p in paths if p.status == 'unknown')}/{args.NUM_PATHS}")
    
    return paths


def main():
 
    parser = argparse.ArgumentParser(description='DIVER-QExpand Parallel Version.')
    parser.add_argument('--dataset_source', type=str, default='data/BRIGHT')
    parser.add_argument('--examples_path', type=str, default='data_making/split_datasets/part_0')
    parser.add_argument('--Batch_SERVER_URL', type=str, default='http://172.16.34.22:8506/batch_retrieve')
    parser.add_argument('--Truncate_URL', type=str, default='http://172.16.34.22:8505/truncate')
    parser.add_argument('--summarization_batch_URL', type=str, default='http://localhost:8502/summrization')
    parser.add_argument('--NUM_PATHS', type=int, default=8)
    parser.add_argument('--MAX_ROUNDS', type=int, default=5)
    
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology', 'earth_science', 'economics', 'pony', 'psychology', 'robotics',
                                 'stackoverflow', 'sustainable_living', 'aops', 'leetcode', 'theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--num_hits', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--keep_passage_num', type=int, default=10)
    parser.add_argument('--max_tokens', type=int, default=32768)
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # data loading
    # examples_path="/home/mingchen/3_Query_rewrite_RL/3_Diver-main2/zero_test_parallel/data_making/split_datasets/part_0"
    
    #examples = load_dataset("parquet", data_files=os.path.join(
       # args.examples_path, f"{args.task}-00000-of-00001.parquet"))["train"]
    examples = load_dataset("parquet", data_files=os.path.join(
        args.examples_path, f"{args.task}_examples.parquet"))["train"]
    org_qid_query_list = [(data['id'], data['query']) for data in examples]
    print(f"Number of queries: {len(org_qid_query_list)}")
    
    # Prepare excluded_ids (parse from string format to list)
    excluded_ids = {}
    for e in examples:
        # Handle both string 'N/A' and list format
        exc_ids = e['excluded_ids']
        if isinstance(exc_ids, str):
            # If it's a string like 'N/A', convert to list
            exc_ids = [exc_ids] if exc_ids != 'N/A' else ['N/A']
        excluded_ids[e['id']] = exc_ids
    
    # Load documents
    docs_path = os.path.join(args.dataset_source, 'documents', f'{args.task}-00000-of-00001.parquet')
    doc_pairs = load_dataset("parquet", data_files=docs_path, cache_dir=args.cache_dir)["train"]
    
    did2content = {}
    for dp in doc_pairs:
        did2content[dp['id']] = dp['content']
    
    # ========== Initialization Phase ==========
    print("\n========== Initialization Phase: Batch retrieve initial results ==========")
    final_all_scores = search_iterator(args, org_qid_query_list, excluded_ids)
    
    last_top_k_passages = {}
    for qid, did_scores in tqdm(final_all_scores.items(), desc="Prepare initial documents"):
        top_docs = [did2content[did] for did, score in 
                   list(did_scores.items())[:args.keep_passage_num]]
        last_top_k_passages[qid] = set(top_docs)
    
    print("Initialization phase completed")
    
    # ========== Parallel Reasoning Phase ==========
    print("\n========== Start Parallel Reasoning ==========")
    
    output_file = open(os.path.join(args.output_dir, "parallel_output_paths.jsonl"), "w")
    
    for idx, (qid, original_query) in enumerate(tqdm(org_qid_query_list, desc="Processing queries")):
        print(f"\n\n{'='*80}")
        print(f"Processing query {idx+1}/{len(org_qid_query_list)}: {qid}")
        print(f"{'='*80}")
        
        # Get initial documents
        initial_docs = last_top_k_passages[qid]
        
        # Run 16 parallel reasoning paths
        paths = run_parallel_reasoning_agent(
            args, qid, original_query, initial_docs, 
            did2content, excluded_ids
        )
        
        # Save each path as a separate line in JSONL
        for path in paths:
            result = {
                "task": args.task,
                "qid": qid,
                # "original_query": original_query,
                "path_id": path.path_id,
                "status": path.status,
                "rounds": path.round_num,
                "running_context": path.running_context
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        output_file.flush()
    
    output_file.close()
    print("\nAll queries processed!")


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
