
import json
import random
from config import PROMPT_FORMAT
from data_generation.sft_data_loader import jload, jdump, load_jsonl


def parse_gpt_batch_outputs(inputs: list, outputs: list) -> dict:
    """Return {(secondary_task, input): output} from GPT batch results."""
    resp_by_id = {o["custom_id"]: o for o in outputs}
    result = {}
    for item in inputs:
        cid     = item.get("custom_id")
        matched = resp_by_id.get(cid)
        if not matched:
            continue
        response = matched.get("response")
        if not response:
            continue
        secondary_task = item["body"]["messages"][0]["content"]
        data           = item["body"]["messages"][1]["content"]
        output         = response["body"]["choices"][0]["message"]["content"]
        result[(secondary_task, data)] = output
    return result


def parse_secalign_prompt(prompt: str) -> tuple[str, str]:
    """Extract (instruction, input) from a SecAlign/StruQ formatted prompt string."""
    instruction = prompt.split("\n\n[MARK] [INPT][COLN]\n")[0].split("[MARK] [INST][COLN]\n")[1]
    input_      = prompt.split("\n\n[MARK] [INPT][COLN]\n")[1].split("\n\n[MARK] [RESP][COLN]")[0]
    return instruction, input_


def build_sft(data_list: list, clean_resp_by_key: dict) -> list:
    """
    For each SEP item, emit two SFT examples:
      1. instruction + injected input  → clean response
      2. instruction + clean input     → clean response
    """
    sft_data = []
    for item in data_list:
        instruction = item["system_prompt"]
        clean_input = item["data_prompt_clean"]
        key = (instruction, clean_input)

        if key not in clean_resp_by_key:
            continue

        clean_response = clean_resp_by_key[key]["output"]
        sft_data.append({"instruction": instruction,
                         "input": item["data_prompt_instructed"],
                         "output": clean_response})
        sft_data.append({"instruction": instruction,
                         "input": clean_input,
                         "output": clean_response})

    return sft_data

def build_sft_clean(data_list: list, clean_resp_by_key: dict) -> list:
    """
    For each SEP item, emit ONE clean SFT example:
      instruction + clean input → clean response
    (No injected examples — unlike build_sft, which also emits the
    injected-input -> clean-response robustness example.)
    """
    sft_data = []
    for item in data_list:
        instruction = item["system_prompt"]
        clean_input = item["data_prompt_clean"]
        key = (instruction, clean_input)

        if key not in clean_resp_by_key:
            continue

        clean_response = clean_resp_by_key[key]["output"]
        sft_data.append({"instruction": instruction,
                         "input": clean_input,
                         "output": clean_response})

    return sft_data

def build_dpo(raw_dpo: list) -> list:
    """Convert SecAlign-format DPO pairs to (instruction, input, chosen, rejected)."""
    result = []
    for item in raw_dpo:
        instruction, input_ = parse_secalign_prompt(item["prompt"])
        result.append({
            "instruction": instruction,
            "input":       input_,
            "chosen":      item["chosen"].removesuffix("</s>"),
            "rejected":    item["rejected"].removesuffix("</s>"),
        })
    return result


def main():

    # ------------------------------------------------------------------
    # Step 2: Submit GPT batch for secondary task responses
    # Uncomment to submit; outputs → datasets/sep/sep_dpo_submit.jsonl
    # ------------------------------------------------------------------
    # from openai import OpenAI
    # data_list = jload("./datasets/sep/train_dataset.json")
    # batch_requests = []
    # for it, item in enumerate(data_list):
    #     secondary_task      = item["info"]["probe"]
    #     orig_data           = item["info"]["data_prompt"]
    #     secondary_task_type = item["info"]["info"]["secondary_task_type"]
    #     data = "" if secondary_task_type == "alpaca" else orig_data
    #     batch_requests.append({
    #         "custom_id": f"request-{it}",
    #         "method": "POST",
    #         "url": "/v1/chat/completions",
    #         "body": {
    #             "model": "gpt-4o-mini-2024-07-18",
    #             "messages": [
    #                 {"role": "developer", "content": secondary_task},
    #                 {"role": "user",      "content": data}
    #             ],
    #             "max_completion_tokens": 100,
    #             "temperature": 0
    #         }
    #     })
    # with open('./datasets/sep/sep_dpo_submit.jsonl', 'w') as f:
    #     for entry in batch_requests:
    #         json.dump(entry, f); f.write('\n')
    # client = OpenAI()
    # batch_input_file = client.files.create(
    #     file=open('./datasets/sep/sep_dpo_submit.jsonl', "rb"), purpose="batch"
    # )
    # client.batches.create(
    #     input_file_id=batch_input_file.id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    # )
    # return  # wait for batch to complete before running Step 3

    # ------------------------------------------------------------------
    # Step 3: Retrieve GPT batch outputs and assemble DPO pairs
    # Uncomment after sep_dpo_retrieved.jsonl is ready
    # ------------------------------------------------------------------
    # inputs  = load_jsonl("./datasets/sep/sep_dpo_submit.jsonl")
    # outputs = load_jsonl("./datasets/sep/sep_dpo_retrieved.jsonl")
    # inject_resp_by_key = parse_gpt_batch_outputs(inputs, outputs)
    #
    # data_list         = jload("./datasets/sep/train_dataset.json")
    # clean_data_list   = jload("./datasets/sep/sep_data_full_withanswer.json")
    # clean_resp_by_key = {(o["instruction"], o["input"]): o for o in clean_data_list}
    #
    # frontend_delimiters = "SpclSpclSpcl"  # SecAlign shares delimiters with StruQ
    # preference_data = []
    # for item in data_list:
    #     instruction = item["system_prompt"]
    #     clean_input = item["data_prompt_clean"]
    #     key = (instruction, clean_input)
    #     if key not in clean_resp_by_key: continue
    #     clean_response = clean_resp_by_key[key]["output"]
    #
    #     secondary_task      = item["info"]["probe"]
    #     orig_data           = item["info"]["data_prompt"]
    #     secondary_task_type = item["info"]["info"]["secondary_task_type"]
    #     inj_data = "" if secondary_task_type == "alpaca" else orig_data
    #     if (secondary_task, inj_data) not in inject_resp_by_key: continue
    #     injected_response = inject_resp_by_key[(secondary_task, inj_data)]
    #
    #     d_item = {"instruction": instruction, "input": item["data_prompt_instructed"]}
    #     preference_data.append({
    #         "prompt":   PROMPT_FORMAT[frontend_delimiters]["prompt_input"].format_map(d_item),
    #         "chosen":   clean_response + "</s>",
    #         "rejected": injected_response + "</s>",
    #     })
    # jdump(preference_data, "./datasets/sep/sep_data_dpo.json")
    # return

    # ------------------------------------------------------------------
    # Step 4: Build final SFT and DPO training sets
    # ------------------------------------------------------------------
    data_list         = jload("./datasets/sep/train_dataset.json")
    clean_data_list   = jload("./datasets/sep/sep_data_full_withanswer.json")
    clean_resp_by_key = {(o["instruction"], o["input"]): o for o in clean_data_list}

    # sft_data = build_sft(data_list, clean_resp_by_key)
    sft_data = build_sft_clean(data_list, clean_resp_by_key)
    random.shuffle(sft_data)
    jdump(sft_data, "./datasets/sep/sep_data_origdata_sft.json")
    print(f"SFT samples saved: {len(sft_data)}")
    #
    # raw_dpo  = jload("./datasets/sep/sep_data_dpo.json")
    # dpo_data = build_dpo(raw_dpo)
    # jdump(dpo_data, "./datasets/sep/sep_data_origdata_dpo.json")
    # print(f"DPO samples saved: {len(dpo_data)}")

    # raw_dpo = jload("./datasets/sep/sep_data_dpo.json")
    # dpo_data = build_dpo(raw_dpo)
    # jdump(dpo_data, "./datasets/sep/sep_data_air_dpo.json")
    # print(f"DPO samples saved: {len(dpo_data)}")


if __name__ == "__main__":
    main()