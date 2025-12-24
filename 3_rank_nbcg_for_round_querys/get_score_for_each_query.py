#!/usr/bin/env python3
import json
import os
import sys
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Get score for each query by part')
    parser.add_argument('--part_num', '-p', type=str, required=True, help='Part number (e.g., part_2, part_12)')
    parser.add_argument('--base_path', '-b', type=str, 
                       default="/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/0_evaluation/bright/Diver/Retriever/output",
                       help='Base output path')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file name')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    part_num = args.part_num
    output_file = args.output_file
    
    # 聚合数据
    data = defaultdict(lambda: defaultdict(dict))

    for part_folder in sorted(os.listdir(base_path)):
        part_path = os.path.join(base_path, part_folder)
        if not os.path.isdir(part_path):
            continue
        
        round_num = int(part_folder.split("round")[-1])
        
        for dataset in os.listdir(part_path):
            results_file = os.path.join(part_path, dataset, "results_qid_score.json")
            if part_num in results_file:
                if os.path.exists(results_file):
                    with open(results_file) as f:
                        results = json.load(f)
                        for qid, metrics in results.items():
                            data[qid][round_num] = metrics

    output = {}
    for qid, rounds in data.items():
        sorted_rounds = []
        for round_num, metrics in rounds.items():
            ndcg_vals = [metrics.get(f"ndcg_cut_{k}", 0) for k in [1, 5, 10, 25, 50, 100]]
            avg = sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0
            sorted_rounds.append((round_num, avg))
        
        sorted_rounds.sort(key=lambda x: x[1], reverse=True)
        output[qid] = sorted_rounds
     
    print(f"Total queries: {len(set(output.keys()))}")

    with open(output_file, 'w') as fw:
        for qid, scores in output.items():
            json.dump({qid: scores}, fw)
            fw.write("\n")
    
    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    main()