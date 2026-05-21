"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import shortuuid
import torch
from tqdm import tqdm
from fastchat.llm_judge.common import load_questions, temperature_config
from data_generation.sft_data_loader import jload, _tokenize_fn, smart_tokenizer_and_embedding_resize
from config import DELIMITERS, DEFAULT_SYSTEM_PROMPT
from testing.test_gcg import CustomConversation
from testing.test import load_full_model
import fastchat

def run_eval(
    model,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
conv_template
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)
    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                conv_template
            )
        )


@torch.inference_mode()
def get_model_answers(
    model,
    model_id,
    questions,
    answer_file,
    max_new_token,
    conv_template
):

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        i = 0
        torch.manual_seed(i)
        conv = fastchat.conversation.get_conv_template(conv_template)
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)  # Using inst_delm
            conv.append_message(conv.roles[1], " " + conv.sep)  # Using resp_delm
            prompt = conv.get_prompt()
            tokenized_inpt = _tokenize_fn([prompt],
                                          tokenizer,
                                          frontend_delimiters=frontend_delimiters)
            input_ids = tokenized_inpt['input_ids'][0].unsqueeze(0).to(model.device)

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # some models may error out when generating long outputs
            try:
                output_ids = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_token,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument("--model-path", type=str, required=True, help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    parser.add_argument("--base_model_path", type=str, default=None,
                   help="Explicit base model path; required when adapter path "
                        "does not encode the base path via the usual suffix convention.")
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument("--bench-name", type=str, default="mtbench", help="The name of the benchmark question set.")
    parser.add_argument("--question-begin", type=int, help="A debug option. The begin index of questions.",)
    parser.add_argument("--question-end", type=int, help="A debug option. The end index of questions.")
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--max-new-token", type=int, default=512, help="The maximum number of new generated tokens.",)

    parser.add_argument('--customized_model_class', type=str, default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_path

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_path,
                                                                              customized_model_class=args.customized_model_class,
                                                                              base_model_path=args.base_model_path)

    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path

    # Unpack delimiters safely for both 3-role and 4-role models.
    # For 3-role: [system/inst, user/data, response]
    # For 4-role: [system, user/inst, data/tool, response]
    # The fastchat conv_template only needs (user_role_delm, response_delm).
    _delm = DELIMITERS[frontend_delimiters]
    inst_delm = _delm[1] if len(_delm) == 4 else _delm[0]  # user/instruction delimiter
    resp_delm = _delm[-1]                                    # response delimiter always last

    if "struq" in model_path or "secalign" in model_path:
        fastchat.conversation.register_conv_template(
            CustomConversation(
                name="struq",
                system_message="",
                roles=(inst_delm, resp_delm),
                sep="\n\n",
                sep2="</s>",
            )
        )
        conv_template = "struq"

    elif "llama" in model_path:
        fastchat.conversation.register_conv_template(
            CustomConversation(
                name="llama3",
                system_message="",
                roles=(inst_delm, resp_delm),
                sep="\n\n",
                sep2="<|eot_id|>",
            )
        )
        conv_template = "llama3"

    elif "mistral" in model_path:
        fastchat.conversation.register_conv_template(
            CustomConversation(
                name="mistral03",
                system_message="",
                roles=(inst_delm, resp_delm),
                sep="\n\n",
                sep2="</s>",
            )
        )
        conv_template = "mistral03"


    question_file = f"datasets/{args.bench_name}.jsonl"
    answer_file = f"{log_path}/predictions_on_mtbench.jsonl"
    if os.path.exists(answer_file):
        os.remove(answer_file)
    print(f"Output to {answer_file}")

    run_eval(
        model=model,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        conv_template=conv_template
    )

    reorg_answer_file(answer_file)