import  json


def get_scores():
    dic_={}
    path="/home/mingchen/3_Query_rewrite_RL/3_Diver-main/zero_test_parallel/0_evaluation/bright/Diver/Retriever/results_sorted_part_10_path_score.json"
    with open(path) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())
            id_=list(line.keys())[0]
            # print()

            scores_rouds=list(line.values())[0]
            scores=[]
            rounds=[]
            # max_round=[]
            for i in range(len(scores_rouds)):
                scores.append(scores_rouds[i][1])
                rounds.append(scores_rouds[i][0])
            
            good_round=[]
            for j in range(len(scores)):
                if scores[j]>0.5:
                    good_round.append((rounds[j],scores[j]))
            if len(good_round)!=0:
               
                # dic_["qid"]=id
                dic_[id_]=good_round
    return dic_

            
qid_round_number=get_scores()
# print(qid_round_number)
fw=open("30B_LLM_part10_pos.jsonl","w")
with open("30B_LLM_dynamic_8_rounds_oupput_part10.jsonl") as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        qid=line["qid"]
        task=line["task"]
        path_id=line["path_id"].split("_")[-1]
        status=line["status"]
        rounds=line["rounds"]
        new_query=line["new_query"]
        if qid in qid_round_number:
            round_nub=qid_round_number[qid]
            # print(round_nub)

            if  len(round_nub)>4:
                round_nub=round_nub[:4]
            rouds={}
            for tup in round_nub:
                rouds[str(tup[0])]=tup[1]
            
            if  path_id in rouds:
                line["path_score"]=rouds[path_id]
                print(line["new_query"])
                fw.write(json.dumps(line)+"\n")
                fw.flush()
                break

            