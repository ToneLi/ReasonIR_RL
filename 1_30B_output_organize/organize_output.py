import os
import json
import re
import sys
import argparse

def extract_last_summary(text,running_context):
    """?????? <summary> ... </summary> ???"""
    matches = re.findall(r"<summary>(.*?)<information>", text, flags=re.DOTALL)
    if matches:
        source_query_matches=re.findall(r"<query>(.*?)</query>", running_context, flags=re.DOTALL)[-1]
        return "summary", source_query_matches+","+matches[-1].strip()#len(matches)
    else:
        source_query_matches=re.findall(r"<query>(.*?)</query>", running_context, flags=re.DOTALL)[-1]
        matches = re.findall(r"<information>(.*?)</information>", running_context, flags=re.DOTALL)
        return "source", source_query_matches+","+matches[-1].strip()


def main():
    parser = argparse.ArgumentParser(description='Organize 30B output to JSONL')
    parser.add_argument('--root_dir', '-r', type=str, required=True, help='Root directory path')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    output_file = args.output_file
    
    all_=0
    hit=0
    do_not_need_retrival=0
    all_dd=[]
    fw=open(output_file,"w")
    for folder in os.listdir(root_dir):
        subdir = os.path.join(root_dir, folder)
        if not os.path.isdir(subdir):
            continue

        json_path = os.path.join(subdir, "parallel_output_paths.jsonl")
        if not os.path.exists(json_path):
            continue

        # ?? JSON ??
        # fw1=open("ddd.txt","w")
        # print("json_path",json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                all_=all_+1
                line_content = line.strip()
                task_match = re.search(r'"task":\s*"([^"]*)"', line_content)
                qid_match = re.search(r'"qid":\s*"([^"]*)"', line_content)
                path_id_match = re.search(r'"path_id":\s*"([^"]*)"', line_content)
                status_match = re.search(r'"status":\s*"([^"]*)"', line_content)
                rounds_match = re.search(r'"rounds":\s*(\d+)', line_content)

                
                running_context=line_content.split("running_context")[1][:-1]
                # print(running_context)
                # break
            
              
                raw_output = running_context.split("MODEL OUTPUT BEGINS")[1]
                question_inform_input=running_context.split("MODEL OUTPUT BEGINS")[0].split("INPUT BEGINS")[1]
                question_match=re.search(r"<query>(.*?)</query>",question_inform_input,re.DOTALL)
                query=question_match.group(1)

                infor_match=re.search(r"<information>(.*?)</information>",question_inform_input,re.DOTALL)
                input_docs=infor_match.group(1)



                tag,new_query = extract_last_summary(raw_output,running_context)
              
                if tag=="summary":
                        hit =hit+1
                else:
                    do_not_need_retrival=do_not_need_retrival+1


                dic_={}
                dic_["task"]=  task_match.group(1)
                all_dd.append(task_match.group(1))
                dic_["qid"] =qid_match.group(1)
                dic_["query"]=query
                dic_["input_docs"]=input_docs
                dic_["path_id"] =path_id_match.group(1)
                
                dic_["status"] =status_match.group(1)
                dic_["rounds"] =rounds_match.group(1)
                dic_["trajectory"] =raw_output
                dic_["new_query"] =new_query.replace("Original query (included as required)","")

                fw.write(json.dumps(dic_) + "\n")
            # break

    fw.close()
    print(set(all_dd))
    print(f"Done! Saved JSONL to {output_file}")
    print(all_)
    print(hit)
    print(hit/all_)
    print(do_not_need_retrival/all_)


if __name__ == "__main__":
    main()