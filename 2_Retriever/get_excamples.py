import argparse
import os

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



task="biology"

dataset_source="/data/home_beta/mingchen/3_Query_rewrite_RL/0_evaluation/bright/Diver/Retriever/data/BRIGHT/"

# load dataset
examples = load_dataset("parquet", data_files=os.path.join(dataset_source,
                                                           f"excample/{task}-00000-of-00001.parquet"))["train"]
org_qid_query_list = [(data['id'], data['query']) for data in examples]
print(org_qid_query_list[0])
excluded_ids = {}
for qid, e in enumerate(examples):
    excluded_ids[e['id']] = e['excluded_ids']
    overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
    assert len(overlap) == 0

ground_truth = {}
for e in tqdm(examples):
    ground_truth[e['id']] = {}
    for gid in e['gold_ids']:
        ground_truth[e['id']][gid] = 1
    for did in e['excluded_ids']:
        assert not did in ground_truth[e['id']]
# load documents
doc_id_ground=ground_truth["0"]
docs_path = os.path.join(dataset_source, 'document', f'{task}-00000-of-00001.parquet')
doc_pairs = load_dataset("parquet", data_files=docs_path)["train"]

doc_ids = []
documents = []
did2content = {}
for dp in doc_pairs:
    doc_ids.append(dp['id'])
    documents.append(dp['content'])
    did2content[dp['id']] = dp['content']
# print("=---------",doc_id_ground)
for doc_id in doc_id_ground.keys():

    print(doc_id)
    print(did2content[doc_id])
    print("================")