
import json
from openai import OpenAI
import re
import random
from collections import defaultdict
from config import PROMPT_FORMAT

if __name__ == '__main__':

    '''Step 2: Answer secondary tasks '''
    # with open('./datasets/sep/train_dataset.json', 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    #
    # batch_requests = []
    # for it, item in enumerate(data_list):
    #     secondary_task = item["info"]["probe"]
    #     orig_data      = item["info"]["data_prompt"]
    #     secondary_task_type = item["info"]["info"]["secondary_task_type"]
    #     if secondary_task_type == "alpaca": # alpaca are QA tasks, does not depend on data
    #         instruction = secondary_task
    #         data = ""
    #     else:
    #         instruction = secondary_task
    #         data = orig_data
    #
    #     batch_requests.append(
    #         {"custom_id": f"request-{it}",
    #          "method": "POST",
    #          "url": "/v1/chat/completions",
    #          "body": {
    #             "model": "gpt-4o-mini-2024-07-18",
    #             "messages": [
    #                 {"role": "developer", "content": instruction},
    #                 {"role": "user", "content": data}
    #             ],
    #             "max_completion_tokens": 100,
    #              "temperature": 0
    #             }
    #         }
    #     )
    #
    # with open('./datasets/sep/sep_dpo_submit.jsonl', 'w') as outfile:
    #     for entry in batch_requests:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    #
    # # Submit batch file
    # client = OpenAI()
    # batch_input_file = client.files.create(
    #     file=open('./datasets/sep/sep_dpo_submit.jsonl', "rb"),
    #     purpose="batch"
    # )
    #
    # # Execute batch request
    # batch_input_file_id = batch_input_file.id
    # client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={
    #         "description": "nightly eval job"
    #     }
    # )
    #
    # exit()

    '''Step 3: DPO, one preferred answer, one rejected answer'''
    with open('./datasets/sep/sep_dpo_submit.jsonl', 'r', encoding='utf-8') as f_in:
        inputs = [json.loads(line) for line in f_in if line.strip()]

    # Load outputs and index by custom_id
    with open('./datasets/sep/sep_dpo_retrieved.jsonl', 'r', encoding='utf-8') as f_out:
        outputs = [json.loads(line) for line in f_out if line.strip()]

    injected_output_fromgpt = []
    resp_by_id = { o["custom_id"]: o for o in outputs }

    for item in inputs:
        cid = item.get("custom_id")
        matched = resp_by_id.get(cid)
        response = matched["response"] if matched else None
        if response:
            secondary_task  = item["body"]["messages"][0]["content"]
            data = item["body"]["messages"][1]["content"]
            response = response['body']['choices'][0]["message"]['content']
            injected_output_fromgpt.append({
                "secondary_task": secondary_task,
                "input": data,
                "output": response
            })
    inject_resp_by_key = {
        (o["secondary_task"], o["input"]): o
        for o in injected_output_fromgpt
    }


    with open('./datasets/sep/train_dataset.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    with open('./datasets/sep/sep_data_full_withanswer.json', 'r', encoding='utf-8') as f_in:
        clean_data_list = json.load(f_in)  # clean

    clean_resp_by_key = {
        (o["instruction"], o["input"]): o
        for o in clean_data_list
    }

    preference_data = []
    for it, item in enumerate(data_list):
        instruction = item["system_prompt"]
        data = item["data_prompt_clean"]

        key = (instruction, data)
        if key in clean_resp_by_key:
            matched_clean = clean_resp_by_key[key]
            clean_response = matched_clean["output"]
        else:
            continue

        secondary_task = item["info"]["probe"]
        orig_data      = item["info"]["data_prompt"]
        secondary_task_type = item["info"]["info"]["secondary_task_type"]
        if secondary_task_type == "alpaca":
            instruction = secondary_task
            data = ""
        else:
            instruction = secondary_task
            data = orig_data

        key = (instruction, data)
        if key in inject_resp_by_key:
            matched_inj = inject_resp_by_key[key]
            injected_response = matched_inj["output"]
        else:
            continue

        d_item = {"instruction": item["system_prompt"],
                  "input": item["data_prompt_instructed"]}
        frontend_delimiters = "SpclSpclSpcl" # secalign shares delimiters with StruQ

        preference_data.append({
            'prompt':   PROMPT_FORMAT[frontend_delimiters]["prompt_input"].format_map(d_item),
            'chosen':   clean_response + "</s>",
            'rejected': injected_response + "</s>"
        })

    with open('./datasets/sep/sep_data_dpo.json', 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)