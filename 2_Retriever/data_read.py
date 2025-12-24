import os
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics
from datasets import load_dataset, Dataset

task="biology"

dataset_source = 'xlangai/BRIGHT'
reasoning_source = 'xlangai/BRIGHT'

path = "/data/home_beta/mingchen/3_Query_rewrite_RL/0_evaluation/bright/Diver/Retriever/data/BRIGHT/"
examples = load_dataset(
    "parquet",
    data_files=path + f"excample/{task}-00000-of-00001.parquet"
)["train"]
doc_pairs = load_dataset(
    "parquet",
    data_files=path + f"document/{task}-00000-of-00001.parquet"
)["train"]

doc_ids = []
documents = []
for dp in doc_pairs:
    doc_ids.append(dp['id'])
    documents.append(dp['content'])
    print(dp['id'])
    print(dp['content'])


    break


queries = []
query_ids = []
excluded_ids = {}
key = 'gold_ids'
for qid, e in enumerate(examples):

    query_ids.append(e['id'])
    excluded_ids[e['id']] = e['excluded_ids']
    overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))

    print(e['id'])
    print(e['excluded_ids'])
    print("--------")
    print(e[key])
    """ 
    0
    ['N/A']
    """
    break

key = 'gold_ids'
ground_truth = {}
for e in tqdm(examples):
    ground_truth[e['id']] = {}
    
    # for gid in e[key]:
    #     ground_truth[e['id']][gid] = 1
    # for did in e['excluded_ids']:
    #     # assert not did in scores[e['id']]
    #     assert not did in ground_truth[e['id']]