import json
import re
import os
from typing import List, Dict
from tqdm import tqdm


"""
Multi-turn conversation format:
[
{"role": "user", "content": "<query>...</query> <information>...</information>"},
{"role": "assistant", "content": "<reason>...</reason><summary>...</summary>"},
{"role": "tool", "content": "<information>...</information>"},
{"role": "assistant", "content": "<reason>...</reason><satisfy>yes</satisfy>"}
]
"""

def trajectory_to_multiturn(query: str, input_docs: str, trajectory_text: str) -> List[Dict]:
    """
    Convert trajectory to multi-turn format.
    
    After <reason>...</reason>:
    Case 1: <summary>...<information> + skip placeholder <information> + extract real <information> + loop <reason>
    Case 2: <satisfy>...</satisfy> + end trajectory
    """
    multi_turns = []
    
    # User turn
    user_content = f"<query>{query.strip()}</query>\n\n<information>{input_docs.strip()}</information>"
    multi_turns.append({"role": "user", "content": user_content})
    
    pos = 0
    
    while pos < len(trajectory_text):
        # Find next <reason>...</reason>
        reason_match = re.search(r'<reason>(.*?)</reason>', trajectory_text[pos:], re.DOTALL)
        
        if not reason_match:
            break
        
        reason_content = reason_match.group(1).strip()
        reason_end = pos + reason_match.end()
        rest_text = trajectory_text[reason_end:]
        
        # Check if there's <summary> (Case 1) or <satisfy> (Case 2)
        summary_match = re.search(r'<summary>(.*?)<information>', rest_text, re.DOTALL)
        
        if summary_match:
            # Case 1: has summary -> information pattern
            summary_content = summary_match.group(1).strip()
            
            # Add assistant message with reason + summary
            multi_turns.append({
                "role": "assistant",
                "content": f"<reason>{reason_content}</reason>\n\n<summary>{summary_content}</summary>"
            })
            
            # Now find all <information>...</information> blocks
            summary_end = reason_end + summary_match.end()-len("<information>")
            info_search_text = trajectory_text[summary_end:]
            
            # Find first (placeholder) and second (real) information blocks
            all_info_matches = list(re.finditer(r'<information>(.*?)</information>', info_search_text, re.DOTALL))
            # print("mmmmmmmm",all_info_matches)
            if len(all_info_matches) >= 2:
                # Skip first (placeholder), use second (real content)
                first_info = all_info_matches[0].group(1).strip()
                second_info = all_info_matches[1].group(1).strip()
                # print(first_info)
                if "next set of retrieved documents will appear here" in first_info:
                    # Confirmed placeholder, use second one
                    multi_turns.append({
                        "role": "tool",
                        "content": f"<information>{second_info}</information>"
                    })
                    # Continue from after second information block
                    pos = summary_end + all_info_matches[1].end()
                else:
                    # First one is real content
                    multi_turns.append({
                        "role": "tool",
                        "content": f"<information>{first_info}</information>"
                    })
                    pos = summary_end + all_info_matches[0].end()
            elif len(all_info_matches) == 1:
                # Only one information block
                info_content = all_info_matches[0].group(1).strip()
                if "retrieved documents will appear here" not in info_content:
                    multi_turns.append({
                        "role": "tool",
                        "content": f"<information>{info_content}</information>"
                    })
                pos = summary_end + all_info_matches[0].end()
            else:
                pos = summary_end
        
        else:
            satisfy_match = re.search(r'<satisfy>(.*?)</satisfy>', rest_text, re.DOTALL)
            if satisfy_match:
                satisfy_content = satisfy_match.group(1).strip()
                multi_turns.append({
                    "role": "assistant",
                    "content": f"<reason>{reason_content}</reason>\n\n<satisfy>{satisfy_content}</satisfy>"
                })
                break  # End of trajectory
            else:
                break
    
    return multi_turns


# ============================================
# Main Program
# ============================================

# ============================================
# Main Program
# ============================================

input_files = [
    "30B_LLM_part12_pos.jsonl",
    "30B_LLM_part11_pos.jsonl",
    "30B_LLM_part10_pos.jsonl"
]

# Open train and dev files
train_fw = open("converted_multiturn_train.jsonl", 'w', encoding='utf-8')
dev_fw = open("converted_multiturn_dev.jsonl", 'w', encoding='utf-8')

total_lines = 0
train_lines = 0
dev_lines = 0

for input_file in input_files:
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue
    
    print(f"\nProcessing {input_file}...")
    
    with open(input_file) as fr:
        for line_num, line in enumerate(tqdm(fr)):
            line = json.loads(line.strip())
            query = line["query"]
            input_docs = line["input_docs"]
            trajectory = line["trajectory"]
            
            multi_turns = trajectory_to_multiturn(query, input_docs, trajectory)
            dic = {"messages": multi_turns}
            output_json = json.dumps(dic, ensure_ascii=False) + '\n'
            
            # 9:1 split - 90% train, 10% dev
            if total_lines % 10 == 0:
                dev_fw.write(output_json)
                dev_fw.flush()
                dev_lines += 1
            else:
                train_fw.write(output_json)
                train_fw.flush()
                train_lines += 1
            
            total_lines += 1

train_fw.close()
dev_fw.close()

print(f"\n{'='*80}")
print(f"Conversion completed!")
print(f"Total lines: {total_lines}")
print(f"Train lines: {train_lines}")
print(f"Dev lines: {dev_lines}")
print(f"Output files:")
print(f"  - converted_multiturn_train.jsonl ({train_lines} lines)")
print(f"  - converted_multiturn_dev.jsonl ({dev_lines} lines)")
print(f"{'='*80}")
