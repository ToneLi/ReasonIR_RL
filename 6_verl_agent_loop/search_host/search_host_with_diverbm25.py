import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer
#from vllm_server.vllm_completion import VLLMCompletion
import openai, json
#from utils.common import save_json, save_json_dict_format
from datasets import load_dataset
import re
from promts_llm_think import get_prompt
import torch.distributed as dist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from pyserini import analysis
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
from collections import defaultdict
import asyncio
analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

from fastapi import FastAPI

app = FastAPI()

import pytrec_eval


def extract_key_sentences(response):
    pattern = r'"([^"]*)"'
    sentences = re.findall(pattern, response)
    joint_sentence = " ".join(sentences)
    return joint_sentence


def extract_answer(response):
    return response.split("</think>\n")[-1]


def extract_expansions(response_list):
    return [extract_answer(response) for response in response_list]


def get_scores(query_ids, doc_ids, scores, excluded_ids, return_full_scores=False, num_hits=1000):
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {}
        assert len(excluded_ids[query_id]) == 0 or (
                isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did, s in zip(doc_ids, doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did != "N/A":
                cur_scores.pop(did)
        if return_full_scores:
            cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:num_hits]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores


def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    # oracle reranker evaluation
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), key=lambda x: results[query_id][x], reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if doc_id in top_100_ids[query_id] and query_id in qrels and doc_id in qrels[
                query_id]:  # a doc is both top 100 and also in ground truth
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id]  # extract the score from ground truth
            else:
                oracle_results[query_id][doc_id] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    oracle_scores = evaluator.evaluate(oracle_results)
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg}

    return output


class Qwen3EmbeddingModel:
    def __init__(self, model_path, device="auto"):
        # if device == "auto":
        self.model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2",
                                               torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.task = "Given a web search query, retrieve relevant passages that answer the query"

    def encode_query(self, query: str, max_length=8192, bs=2):
        return self.encode(self.get_detailed_instruct(self.task, query), max_length)[0]

    def encode_query_batch(self, query_list: list, max_length=8192, bs=2):
        instruct_list = [
            self.get_detailed_instruct(self.task, q)
            for q in query_list
        ]

        return self.encode(instruct_list, max_length)

    def encode_doc(self, doc, max_length=16384):
        return self.encode(doc, max_length)[0]

    def encode_docs(self, docs, max_length=16384, bs=1):
        embeddings = []
        for i in trange(0, len(docs), bs):
            embeddings.append(self.encode(docs[i:i + bs], max_length))
        return np.vstack(embeddings)

    def encode(self, text: list, max_length: int = 16384):
        # Tokenize the input texts
        text = [text] if isinstance(text, str) else text
        batch_dict = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().detach().numpy()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'


class VectorSearchInterface(object):
    def __init__(self, model_path, model_name, model, cache_dir, task, doc_ids: list, documents: list):
        self.model_path = model_path
        self.model_name = model_name
        self.model = model
        self.cache_dir = cache_dir
        self.task = task
        self.doc_ids = doc_ids
        self.documents = documents
        self.docs_emb = self.get_docs_emb(documents)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def truncate_text(self, doc_text, max_tokens=100):
        doc_ids = self.model.tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_ids) > max_tokens:
            doc_ids = doc_ids[:max_tokens]
        return self.model.tokenizer.decode(doc_ids, skip_special_tokens=True)

    def get_docs_emb(self, documents):
        # cache docs emb
        cache_path = os.path.join(self.cache_dir, 'doc_emb', self.model_name, self.task, f"long_False")
        os.makedirs(cache_path, exist_ok=True)
        doc_cache_file = os.path.join(cache_path, '0.npy')

        print('Encoding documents to cache:', cache_path)
        if os.path.exists(doc_cache_file):
            docs_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            with torch.inference_mode():
                docs_emb = self.model.encode_docs(documents)
            torch.cuda.empty_cache()
            np.save(doc_cache_file, docs_emb)
            # print("Shape of doc emb", docs_emb.shape)

        return docs_emb

    torch.no_grad()

    def do_retrieval(self, qid, query_text, excluded_ids, num_hits=1000):
        '''
        return: dict of dict {qid: {doc_id: score}, }
        '''
        # print("===com in =======================")
        query_texts_list = [query_text] if isinstance(query_text, str) else query_text
        qid_list = [qid] if isinstance(qid, str) else qid
        # print("==query_texts =======================",query_texts)
        # print("==qid=======================", qid)
        # print("=======self.model",self.model)

        with torch.inference_mode():
            # for q in query_texts:
            # print("=---------subq",q)
            query_emb = self.model.encode_query_batch(query_texts_list)

        query_emb = np.array(query_emb)
        print("Shape of query emb", query_emb.shape)
        torch.cuda.empty_cache()

        scores = cosine_similarity(query_emb, self.docs_emb).tolist()

        qid_doc_scores = get_scores(query_ids=qid, doc_ids=self.doc_ids, scores=scores, excluded_ids=excluded_ids,
                                    num_hits=num_hits)
        # print("=========qid_doc_scores",qid_doc_scores)
        return qid_doc_scores

    def do_retrieval_batch(self, qid_list, query_text_list, excluded_ids, num_hits=1000):
        '''
        return: dict of dict {qid: {doc_id: score}, }
        '''
        # print("===com in =======================")
        query_texts = query_text_list  # [query_text] if isinstance(query_text, str) else query_text
        qid = qid_list  # [qid] if isinstance(qid, str) else qid
        # print("==query_texts =======================",query_texts)
        # print("==qid=======================", qid)
        # print("=======self.model",self.model)
        query_emb = []
        with torch.inference_mode():
            for q in query_texts:
                # print("=---------subq",q)
                query_emb.append(self.model.encode_query(q))

        query_emb = np.array(query_emb)
        print("Shape of query emb", query_emb.shape)
        torch.cuda.empty_cache()

        scores = cosine_similarity(query_emb, self.docs_emb).tolist()

        qid_doc_scores = get_scores(query_ids=qid, doc_ids=self.doc_ids, scores=scores, excluded_ids=excluded_ids,
                                    num_hits=num_hits)
        # print("=========qid_doc_scores",qid_doc_scores)
        return qid_doc_scores


def progressive_query_rewrite(
        openai_api, cur_query, top_passages, iter_round,
        accumulated_query_expansions=[],
        max_demo_len=None,
        expansion_method="",
        # accumulate=False,
        topic_id=None, search_api=None,
        *arg, **kwargs):
    if max_demo_len:  # 512
        top_passages = [search_api.truncate_text(psg, max_demo_len) for psg in top_passages]

    top_passages_str = "\n".join([f"[{idx + 1}]. {psg}" for idx, psg in enumerate(top_passages)])

    if iter_round > 1:
        user_prompt = get_prompt(expansion_method, cur_query, top_passages_str,
                                 accumulated_query_expansions[topic_id][-1])
    else:
        user_prompt = get_prompt("thinkqe_revise_0", cur_query, top_passages_str)

    messages = [{"role": "user", "content": user_prompt}]

    # print("Input message:" + user_prompt)
    gen_fn = openai_api.completion_chat
    response_list = gen_fn(messages, *arg, **kwargs)
    query_expansions = extract_expansions(response_list)

    accumulated_query_expansions[topic_id].extend(query_expansions)
    query = cur_query + "\n\n" + "\n".join(query_expansions)

    return query, response_list, accumulated_query_expansions

_workspace_root = Path(__file__).resolve().parents[4]

dataset_source = os.getenv(
    "BRIGHT_DATASET_SOURCE",
    str(_workspace_root / "3_Query_rewrite_RL/3_Diver-main/data/BRIGHT")
)
  
model_path = os.getenv("DIVER_MODEL_PATH", "AQ-MedAI/Diver-Retriever-4B")
model_name = os.getenv("DIVER_MODEL_NAME", "diver-retriever")
cache_dir = os.getenv(
    "DIVER_CACHE_DIR",
    str(_workspace_root / "3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/cache/cache_diver-retriever")
)

print("dataset_source:", dataset_source)
print("model_path:", model_path)
print("model_name:", model_name)
print("cache_dir:", cache_dir)

model = Qwen3EmbeddingModel(model_path)

TASK_SEARCH_API = {}
TASKS = [
    "biology",
"earth_science",
"economics",
"psychology",
"robotics",
"stackoverflow",
"sustainable_living",
"leetcode",
"pony",
"aops",
"theoremqa_questions",
"theoremqa_theorems"

]

Task_corpus_index={}
for task in TASKS:
    excample_doc={}
    docs_path = os.path.join(dataset_source, 'documents', f'{task}-00000-of-00001.parquet')
    doc_pairs = load_dataset("parquet", data_files=docs_path, cache_dir=cache_dir)["train"]

    doc_ids = []
    documents = []
    did2content = {}
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
        did2content[dp['id']] = dp['content']
    # excample_doc["queries"]=queries
    # excample_doc["query_ids"] = query_ids
    # excample_doc["excluded_ids"] = excluded_ids
    # excample_doc["documents"] = documents
    # excample_doc["doc_ids"] = doc_ids
    # Task_data[task]=excample_doc

    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    bm25_model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    D={}
    D["dictionary"]=dictionary
    D["bm25_index"] = bm25_index
    D["doc_ids"] = doc_ids
    D["bm25_model"] = bm25_model
    Task_corpus_index[task]=D

    TASK_SEARCH_API[task] = VectorSearchInterface(model_path, model_name, model, cache_dir, task, doc_ids, documents)


def retrieval_bm25(bm25_model,queries, query_ids, dictionary,bm25_index, doc_ids,excluded_ids, **kwargs):
    store_all_score = kwargs.get('store_all_scores', False)

    all_scores = {}
    repeat_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = bm25_model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        repeat_scores[str(query_id)] = defaultdict(list)
        for did, s in zip(doc_ids, similarities):
            # all_scores[str(query_id)][did] = s
            # repeat_scores[str(query_id)][did].append(s)
            if did not in all_scores[str(query_id)] or s > all_scores[str(query_id)][did]:
                all_scores[str(query_id)][did] = s  # refine docs, save the best score for each query-doc pair

        for did in set(excluded_ids[str(query_id)]):
            if did != "N/A":
                all_scores[str(query_id)].pop(did)
        if store_all_score:
            cur_scores = sorted(all_scores[str(query_id)].items(), key=lambda x: x[1], reverse=True)
        else:
            cur_scores = sorted(all_scores[str(query_id)].items(), key=lambda x: x[1], reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]

    print("Dedup Scores shape", len(repeat_scores))

    return all_scores

def merge_bm25_dense_score(bm25_scores, dense_scores,wgt_dense=0.5):
    wgt_bm25 = 1 - wgt_dense



    sorted_merge_scores = {}

    for query_id in dense_scores:
        dense_dict = dense_scores[query_id]
        bm25_dict = bm25_scores.get(query_id, {})

        # Normalize scores
        dense_values = list(dense_dict.values())
        dense_max = max(dense_values) if dense_values else 0
        dense_min = min(dense_values) if dense_values else 0
        bm25_values = list(bm25_dict.values())
        bm25_max = max(bm25_values) if bm25_values else 0
        bm25_min = min(bm25_values) if bm25_values else 0

        all_docs = set(dense_dict.keys()) | set(bm25_dict.keys())

        merged_doc_score = {}
        for did in all_docs:
            xrr_norm = (dense_dict.get(did, 0) - dense_min) / (dense_max - dense_min) if (
                                                                                                     dense_max - dense_min) != 0 else 0
            bm25_norm = (bm25_dict.get(did, 0) - bm25_min) / (bm25_max - bm25_min) if (bm25_max - bm25_min) != 0 else 0

            merged_doc_score[did] = xrr_norm * wgt_dense + bm25_norm * wgt_bm25

        sorted_docs = sorted(merged_doc_score.items(), key=lambda x: x[1], reverse=True)
        sorted_merge_scores[query_id] = dict(sorted_docs) # [:100]

    return sorted_merge_scores

@app.post("/truncate")
def truncate(req: dict):
    search_api = TASK_SEARCH_API[req["task"]]
    psg = req["psg"]
    max_demo_len = req["max_demo_len"]
    return {"text": search_api.truncate_text(psg, max_demo_len)}




@app.post("/batch_retrieve")
def batch_retrieve(req: dict):
    # print("===========req",req)
    # print("--------------req[task]",req["task"])
    # print("=======TASK_SEARCH_API",TASK_SEARCH_API)
    search_api = TASK_SEARCH_API[req["task"]]
    # print("===========search_api", search_api)
    q_id_list = req["q_id_list"]
    q_text_list = req["q_text_list"]
    excluded = req["excluded_ids"]
    num_hits = req["num_hits"]

    dense_scores = search_api.do_retrieval_batch(q_id_list, q_text_list, excluded, num_hits)
  
    bm25_scores=retrieval_bm25(Task_corpus_index[req["task"]]["bm25_model"],
                               q_text_list,q_id_list,
                               Task_corpus_index[req["task"]]["dictionary"],
                               Task_corpus_index[req["task"]]["bm25_index"],
                               Task_corpus_index[req["task"]]["doc_ids"],excluded)


    # print("======bm25_scores",bm25_scores) #'induce_sleep/Adenosinetriphosphat_377_0.txt': 0.6855451906713346,
    task=req["task"]
    wgt_dense =0.5
    if task in ["leetcode", "pony", "aops", "theoremqa_theorems", "theoremqa_questions"]:
        wgt_dense =0.7
    # print(wgt_dense)
    merged_score=merge_bm25_dense_score(bm25_scores, dense_scores, wgt_dense=wgt_dense)
    # print(merged_score)
    return {"scores": merged_score}

