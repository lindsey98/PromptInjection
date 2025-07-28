import random

from modeling.gpt import GPTAgent
from data_generation.struq import jload, jdump, format_with_other_delimiters
from config import DELIMITERS
from copy import deepcopy
import numpy as np
import os
from tqdm import tqdm
# export OPENAI_API_KEY="your_api_key_here"


if __name__ == "__main__":
    # File paths
    DATA_PATH = "datasets/alpaca_data_cleaned.json"
    SAVE_PATH = "datasets/alpaca_data_cleaned_gpt.json"

    # Load dataset
    list_data_dict = jload(DATA_PATH)

    # Check if previous progress exists
    if os.path.exists(SAVE_PATH):
        print("Resuming from previously saved progress...")
        saved_data_dict = jload(SAVE_PATH)
        if len(saved_data_dict) == len(list_data_dict):
            list_data_dict = saved_data_dict  # Resume from saved state

    # Initialize GPT agent
    agent = GPTAgent()

    # Load reference responses
    ref_inst_resp = {sample['instruction']: sample['output'] for sample in jload('./datasets/alpaca_data.json')}

    # Track attacks
    attacks_conducted = []
    for i in range(len(list_data_dict)):
        if "gpt_responses" in list_data_dict[i]:  # Skip already processed samples
            attacks_conducted.append(False)
            continue

        if list_data_dict[i].get("input", "") == "":
            attacks_conducted.append(False)
            continue

        # Randomly choose another instruction from the dataset
        injected_sample = deepcopy(np.random.choice(list_data_dict))
        injected_prompt = injected_sample['instruction'].capitalize() + " " + injected_sample['input']

        data_dicts_item = deepcopy(list_data_dict[i])
        if data_dicts_item['input'][-1] != '.':
            data_dicts_item['input'] += '.'

        p = random.random()
        if p < 0.5:
            attacks_conducted.append(False)
            continue
        elif p < 0.75:  # Naive attack
            data_dicts_item['input'] += " " + injected_prompt
            list_data_dict[i] = data_dicts_item
            attacks_conducted.append(True)
        else:  # Completion attack
            data_dicts_item['input'] += "\n\n" + DELIMITERS['SpclSpclSpcl'][2] + "\n" + \
                                        ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + \
                                        "\n\n" + DELIMITERS['SpclSpclSpcl'][0] + "\n" + injected_prompt
            if injected_sample['input']:
                data_dicts_item['input'] += "\n\n" + DELIMITERS['SpclSpclSpcl'][1] + "\n" + injected_sample['input']
            data_dicts_item['input'] = format_with_other_delimiters(data_dicts_item['input'], test=False)
            list_data_dict[i] = data_dicts_item
            attacks_conducted.append(True)

    # Ensure the lengths match
    assert len(attacks_conducted) == len(list_data_dict)

    # Process GPT responses with retries
    SAVE_INTERVAL = 200  # Save progress every 10 samples

    for i in tqdm(range(len(list_data_dict))):
        if "gpt_responses" in list_data_dict[i]:  # Skip already processed
            continue

        if attacks_conducted[i] is True:
            gpt_responses = agent.task_completion(
                system_instruction=list_data_dict[i]["instruction"],
                user_input=list_data_dict[i]["input"]
            )
            list_data_dict[i]["gpt_responses"] = gpt_responses
            print(gpt_responses)
        # Save progress periodically
        if i % SAVE_INTERVAL == 0:
            jdump(list_data_dict, SAVE_PATH)
            print(f"Progress saved at sample {i}/{len(list_data_dict)}")

    # Final save
    jdump(list_data_dict, SAVE_PATH)
    print("Processing complete. Final dataset saved.")