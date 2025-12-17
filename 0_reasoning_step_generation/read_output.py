import json

l=[]
# Read first line of parallel output
with open('30B_LLM_zero_dynamic_stop_16_rounds_merged_output.jsonl', 'r') as f:
    for line in f.readlines():
      data = json.loads(line.strip())
      task=data["task"]
      if task=="theoremqa_questions":
        l.append(data["qid"])
      
      
print(len(set(l)))


    #print(len(data['original_query']))
     
    # break


