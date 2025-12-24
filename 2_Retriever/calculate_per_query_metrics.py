import os
import argparse
import json
from tqdm import tqdm
from retrievers import calculate_retrieval_metrics
from datasets import load_dataset


def calculate_per_query_metrics(scores, qrels):
    """
    Calculate retrieval metrics for each query individually
    
    Args:
        scores: dict of {query_id: {doc_id: score}}
        qrels: dict of {query_id: {doc_id: relevance}}
    
    Returns:
        dict of {query_id: metrics}
    """
    per_query_results = {}
    
    for query_id in tqdm(scores.keys(), desc="Calculating metrics per query"):
        # Create single query score and qrel dicts
        single_query_score = {query_id: scores[query_id]}
        single_query_qrel = {query_id: qrels[query_id]}
        
        # Calculate metrics for this single query
        metrics = calculate_retrieval_metrics(results=single_query_score, qrels=single_query_qrel)
        
        per_query_results[query_id] = metrics
    
    return per_query_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reasoning', type=str, required=True, help='e.g., 30B_LLM_round1')
    parser.add_argument('--model', type=str, default='diver-retriever')
    parser.add_argument('--long_context', type=str, default='False')
    args = parser.parse_args()
    
    # All available tasks
    all_tasks = ['biology', 'earth_science', 'economics', 'pony', 'psychology', 'robotics',
                 'stackoverflow', 'sustainable_living', 'aops', 'leetcode', 'theoremqa_theorems',
                 'theoremqa_questions']
    
    # Base directory
    base_output_dir = f"/data/home_beta/mingchen/3_Query_rewrite_RL/0_evaluation/bright/Diver/Retriever/output/diver-retriever_{args.reasoning}_reasoning"
    
    print(f"Processing reasoning: {args.reasoning}")
    print(f"Base directory: {base_output_dir}\n")
    
    # Process each task
    for task in all_tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task}")
        print(f"{'='*60}")
        
        # Construct paths for this task
        task_output_dir = os.path.join(base_output_dir, f"{task}_{args.model}_long_{args.long_context}")
        score_file_path = os.path.join(task_output_dir, f'{args.reasoning}_score.json')
        print("score_file_path",score_file_path)
        output_file_path = os.path.join(task_output_dir, f'{args.reasoning}_per_query_results.json')
        # Check if score file exists
        if not os.path.isfile(score_file_path):
            print(f"⚠️  Score file does not exist, skipping: {score_file_path}")
            continue
        
        # Load scores
        print(f"Loading scores from {score_file_path}")
        with open(score_file_path) as f:
            scores = json.load(f)
        
        # Load dataset
        path = "/data/home_beta/mingchen/3_Query_rewrite_RL/0_evaluation/bright/Diver/Retriever/data/BRIGHT/"
        try:
            examples = load_dataset(
                "parquet",
                data_files=path + f"excample/{task}-00000-of-00001.parquet"
            )["train"]
        except Exception as e:
            print(f"⚠️  Failed to load dataset for task {task}: {e}")
            continue
        
        # Build ground truth
        if args.long_context == 'True':
            key = 'gold_ids_long'
        else:
            key = 'gold_ids'
        
        ground_truth = {}
        for e in tqdm(examples, desc="Building ground truth"):
            ground_truth[e['id']] = {}
            for gid in e[key]:
                ground_truth[e['id']][gid] = 1
        
        # Calculate per-query metrics
        print("Calculating metrics for each query...")
        per_query_results = calculate_per_query_metrics(scores, ground_truth)
        
        # Save results
        print(f"Saving per-query results to {output_file_path}")
        with open(output_file_path, 'w') as f:
            json.dump(per_query_results, f, indent=2)
        
        # Calculate and print average metrics
        print("\n" + "-"*50)
        print(f"Average Metrics for {task}:")
        print("-"*50)
        
        # Calculate averages
        all_metrics = {}
        for query_id, metrics in per_query_results.items():
            for metric_name, metric_value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        average_metrics = {}
        for metric_name, values in all_metrics.items():
            average_metrics[metric_name] = sum(values) / len(values)
        
        for metric_name, avg_value in sorted(average_metrics.items()):
            print(f"{metric_name}: {avg_value:.4f}")
        
        # Save average metrics as well
        avg_output_path = output_file_path.replace('per_query_results.json', 'average_results.json')
        with open(avg_output_path, 'w') as f:
            json.dump(average_metrics, f, indent=2)
        
        print(f"\n✓ Average metrics saved to {avg_output_path}")
        print(f"✓ Per-query metrics saved to {output_file_path}")
        print(f"✓ Total queries processed: {len(per_query_results)}")
    
    print(f"\n{'='*60}")
    print("All tasks processed!")
    print(f"{'='*60}")
