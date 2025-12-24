import os
import json
import csv
from tqdm import tqdm

def analyze_rounds_ndcg():
    """
    Analyze NDCG@10 across 16 rounds for all tasks and queries
    """
    
    # Base directory
    base_dir = "/data/home_beta/mingchen/3_Query_rewrite_RL/0_evaluation/bright/Diver/Retriever/output"
    
    # All tasks
    all_tasks = ['aops', 'biology', 'earth_science', 'economics', 'leetcode', 'pony', 
                 'psychology', 'robotics', 'stackoverflow', 'sustainable_living', 
                 'theoremqa_theorems', 'theoremqa_questions']
    
    # Number of rounds
    num_rounds = 16
    
    # Output CSV file
    output_csv = os.path.join(base_dir, "ndcg10_analysis_all_rounds.csv")
    
    # Prepare CSV headers
    headers = ['task', 'query_id']
    for r in range(1, num_rounds + 1):
        headers.append(f'NDCG@10_round{r}')
    headers.extend(['avg_NDCG@10', 'max_NDCG@10', 'min_NDCG@10'])
    
    # Collect all data
    all_data = []
    
    print(f"Analyzing NDCG@10 for {len(all_tasks)} tasks across {num_rounds} rounds...\n")
    
    for task in all_tasks:
        print(f"Processing task: {task}")
        
        # Store data for each query in this task
        task_query_data = {}
        
        # Read data from all 16 rounds
        for round_num in range(1, num_rounds + 1):
            reasoning_folder = f"diver-retriever_30B_LLM_round{round_num}_reasoning"
            task_folder = f"{task}_diver-retriever_long_False"
            json_file = f"30B_LLM_round{round_num}_per_query_results.json"
            
            file_path = os.path.join(base_dir, reasoning_folder, task_folder, json_file)
            
            if not os.path.isfile(file_path):
                print(f"  ⚠️  File not found: {file_path}")
                continue
            
            # Load the JSON file
            with open(file_path, 'r') as f:
                round_data = json.load(f)
            
            # Extract NDCG@10 for each query
            for query_id, metrics in round_data.items():
                if query_id not in task_query_data:
                    task_query_data[query_id] = {
                        'task': task,
                        'query_id': query_id,
                        'rounds': {}
                    }
                
                # Store NDCG@10 for this round
                ndcg10 = metrics.get('NDCG@10', None)
                task_query_data[query_id]['rounds'][round_num] = ndcg10
        
        # Calculate statistics for each query
        for query_id, data in task_query_data.items():
            row = [data['task'], data['query_id']]
            
            # Add NDCG@10 for each round
            ndcg_values = []
            for r in range(1, num_rounds + 1):
                ndcg = data['rounds'].get(r, None)
                row.append(ndcg if ndcg is not None else 'N/A')
                if ndcg is not None:
                    ndcg_values.append(ndcg)
            
            # Calculate statistics
            if ndcg_values:
                avg_ndcg = sum(ndcg_values) / len(ndcg_values)
                max_ndcg = max(ndcg_values)
                min_ndcg = min(ndcg_values)
            else:
                avg_ndcg = 'N/A'
                max_ndcg = 'N/A'
                min_ndcg = 'N/A'
            
            row.extend([avg_ndcg, max_ndcg, min_ndcg])
            all_data.append(row)
        
        print(f"  ✓ Processed {len(task_query_data)} queries for task {task}")
    
    # Write to CSV
    print(f"\nWriting results to {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_data)
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"✓ Total queries processed: {len(all_data)}")
    print(f"✓ Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary by task:")
    task_counts = {}
    for row in all_data:
        task = row[0]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task in sorted(task_counts.keys()):
        print(f"  {task}: {task_counts[task]} queries")


if __name__ == '__main__':
    analyze_rounds_ndcg()
