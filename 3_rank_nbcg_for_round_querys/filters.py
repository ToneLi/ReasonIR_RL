import json
import argparse
import sys


def get_scores(score_file):
    """Extract good rounds from score file"""
    dic_ = {}
    with open(score_file) as fr:
        for line in fr.readlines():
            line = json.loads(line.strip())
            id_ = list(line.keys())[0]
            scores_rouds = list(line.values())[0]
            scores = []
            rounds = []
            for i in range(len(scores_rouds)):
                scores.append(scores_rouds[i][1])
                rounds.append(scores_rouds[i][0])
            
            good_round = []
            for j in range(len(scores)):
                if scores[j] > 0.5:
                    good_round.append((rounds[j], scores[j]))
            if len(good_round) != 0:
                dic_[id_] = good_round
    return dic_


def main():
    parser = argparse.ArgumentParser(description='Filter and add path scores to trajectory data')
    parser.add_argument('--part_num', '-p', type=str, required=True, help='Part number (e.g., part_2, part_10)')
    parser.add_argument('--score_file', '-s', type=str, required=True, help='Path to score file (results_sorted_part_X_path_score.json)')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='Input trajectory file (30B_LLM_dynamic_8_rounds_oupput_part_X.jsonl)')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file name (e.g., 30B_LLM_part_2_pos.jsonl)')
    
    args = parser.parse_args()
    
    part_num = args.part_num
    score_file = args.score_file
    input_file = args.input_file
    output_file = args.output_file
    
    print(f"Processing {part_num}...")
    print(f"  Score file: {score_file}")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    
    # Get scores
    qid_round_number = get_scores(score_file)
    
    # Process trajectory data
    count = 0
    with open(output_file, 'w') as fw:
        with open(input_file) as fr:
            for line in fr.readlines():
                line = json.loads(line.strip())
                qid = line["qid"]
                task = line["task"]
                path_id = line["path_id"].split("_")[-1]
                status = line["status"]
                rounds = line["rounds"]
                new_query = line["new_query"]
                
                if qid in qid_round_number:
                    round_nub = qid_round_number[qid]
                    
                    if len(round_nub) > 4:
                        round_nub = round_nub[:4]
                    
                    rouds = {}
                    for tup in round_nub:
                        rouds[str(tup[0])] = tup[1]
                    
                    if path_id in rouds:
                        line["path_score"] = rouds[path_id]
                        fw.write(json.dumps(line) + "\n")
                        fw.flush()
                        count += 1
    
    print(f"Done! Processed {count} records and saved to {output_file}")


if __name__ == "__main__":
    main()

            
