import os
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics
from datasets import load_dataset, Dataset





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round_number', type=int, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--generated_files', type=str, default=None)
    parser.add_argument('--examples_path', type=str, default=None)
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology', 'earth_science', 'economics', 'pony', 'psychology', 'robotics',
                                 'stackoverflow', 'sustainable_living', 'aops', 'leetcode', 'theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25', 'cohere', 'e5', 'google', 'grit', 'inst-l', 'inst-xl',
                                 'openai', 'qwen', 'qwen2', 'sbert', 'sf', 'voyage', 'bge',
                                 'bge_ce', 'nomic', 'm2', 'contriever', 'reasonir', 'rader', 'diver-retriever'])
    parser.add_argument('--model_id', type=str, default=None,
                        help='(Optional) Pass a different model ID for cache and output path naming.')
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--dataset_source', type=str, default='../data/BRIGHT')
    parser.add_argument('--document_expansion', default=None, type=str, choices=[None, 'gold', 'full', 'rechunk'],
                        help="Set to None to use original documents provided by BRIGHT; Set to `oracle` to use documents with oracle ones expanded'; Set to `full` to use all expanded documents.")
    parser.add_argument('--global_summary', default=None, choices=[None, 'concat'])
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
   


    parser.add_argument('--reasoning_id', type=str, default=None)
    parser.add_argument('--reasoning_length_limit', type=int, default=None)
    parser.add_argument('--separate_reasoning', action='store_true',
                        help='Append reasoning after the original query, separate by <REASON>.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--no_log', action='store_true', help="Disable logging to Google Sheets.")
    parser.add_argument('--sweep_output_dir', type=str, default=None)
    parser.add_argument('--skip_doc_emb', action='store_true', help="Skip document embedding.")
    parser.add_argument('--store_all_scores', action='store_true',
                        help="The default is to store the top 1000 scores. This option will store all scores.")
    args = parser.parse_args()
    if args.model_id is None:
        args.output_dir = os.path.join(args.output_dir, f"{args.task}_{args.model}_long_{args.long_context}")
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.task}_{args.model_id}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.reasoning is not None:
        score_file_path = os.path.join(args.output_dir, f'{args.reasoning}_score.json')
    else:
        score_file_path = os.path.join(args.output_dir, f'score.json')

    assert args.document_expansion is None or args.global_summary is None, "Cannot use expansion and summary together!"
    if args.global_summary:
        assert not args.long_context, "Global summary is supposed to enhance short-context retrieval!"

    # dataset_source = args.dataset_source
    document_postfix = ''

    dataset_source = 'xlangai/BRIGHT'
    reasoning_source = 'xlangai/BRIGHT'

    def load_generated_query():

        with open(args.generated_files) as fr:
            dic_ = {}
            for line in fr.readlines():
                line = json.loads(line.strip())
                generated_text = line["new_query"]
                task = line["task"]
                path_id=line["path_id"]  #  16_path_1
                round_number=path_id.split("path_")[1]
                #print("task",task)
                #print(line["qid"])
                # print(task + "|" + str(line["qid"]))
                if round_number==str(args.round_number):
                    dic_[task + "|" + str(line["qid"])] = generated_text
            return dic_

   
    Q_dict = load_generated_query()

    examples = load_dataset(
        "parquet",
        data_files=args.examples_path + f"{args.task}_examples.parquet"
    )["train"]
    doc_pairs = load_dataset(
        "parquet",
        data_files=args.path + f"documents/{args.task}-00000-of-00001.parquet"
    )["train"]

    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])

    if not os.path.isfile(score_file_path):
        print("The scores file does not exist, start retrieving...")
        if args.model in ['rader', 'reasonir']:
            with open(os.path.join(args.config_dir, args.model.split('_ckpt')[0].split('_bilevel')[0],
                                   f"{args.task}.json")) as f:
                config = json.load(f)
        else:
            config = {}
            config['instructions'] = None  # default instructions

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
     
        for qid, e in enumerate(examples):
            task_id = args.task + "|" + str(e['id'])
            if task_id not in Q_dict:
                generated_query= e['query']  # biology_8869
            else:
                generated_query = Q_dict[task_id]
            # print(generated_query)
            queries.append(generated_query)  # e["query"]
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))

            assert len(overlap) == 0
        assert len(queries) == len(query_ids), f"{len(queries)}, {len(query_ids)}"
        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir, 'doc_ids', f"{args.task}_{args.long_context}.json")):
            try:
                with open(os.path.join(args.cache_dir, 'doc_ids', f"{args.task}_{args.long_context}.json")) as f:
                    cached_doc_ids = json.load(f)
                for id1, id2 in zip(cached_doc_ids, doc_ids):
                    assert id2 in cached_doc_ids
            except:
                print("Document IDs mismatche with the cached version!")
        else:
            with open(os.path.join(args.cache_dir, 'doc_ids', f"{args.task}_{args.long_context}.json"), 'w') as f:
                json.dump(doc_ids, f, indent=2)
        assert len(doc_ids) == len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")

        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length > 0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length > 0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size > 0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})
        if args.skip_doc_emb:
            kwargs.update({'skip_doc_emb': args.skip_doc_emb})
        if args.store_all_scores:
            kwargs.update({'store_all_scores': args.store_all_scores})
        kwargs.update({'document_postfix': document_postfix})
        kwargs.update({'model_name': args.model})
        kwargs.update({'model_name': args.model})

        model_id = args.model_id if args.model_id is not None else args.model
        scores = RETRIEVAL_FUNCS[args.model](queries=queries, query_ids=query_ids, documents=documents,
                                             excluded_ids=excluded_ids,
                                             instructions=config['instructions_long'] if args.long_context else config[
                                                 'instructions'],
                                             doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir,
                                             long_context=args.long_context,
                                             model_id=model_id, checkpoint=args.checkpoint, **kwargs)
        with open(score_file_path, 'w') as f:
            json.dump(scores, f, indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path, 'exists')
    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]
    # print("mmmmmmmmmmmmmmmmmmmm",ground_truth)
    # print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results_qid_score.json'), 'w') as f:
        json.dump(results, f, indent=2)
        f.flush()

    # # print(results)
    # # track successful completion of the run
    # if args.sweep_output_dir:
    #     with open(os.path.join(args.sweep_output_dir, 'done'), 'w') as f:
    #         f.write('done')


    """
   nohup bash retriever_script_1_0.sh > retriever_1_0.log 2>&1 &
nohup bash retriever_script_1_1.sh > retriever_1_1.log 2>&1 &
nohup bash retriever_script_1_2.sh > retriever_1_2.log 2>&1 &
nohup bash retriever_script_1_3.sh > retriever_1_3.log 2>&1 &
nohup bash retriever_script_1_4.sh > retriever_1_4.log 2>&1 &
nohup bash retriever_script_1_5.sh > retriever_1_5.log 2>&1 &
nohup bash retriever_script_1_6.sh > retriever_1_6.log 2>&1 &
nohup bash retriever_script_1_7.sh > retriever_1_7.log 2>&1 &

    """