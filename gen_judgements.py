"""
Usage:
python gen_judgment.py 
"""
from utils_model import *
import argparse
import jsonlines
import concurrent.futures
import openai
import tqdm
import random
import os

def read_jsonl_from_path(file_path):
    with jsonlines.open(file_path) as reader:
        dataset = [obj for obj in reader]
    print('Num of data:', len(dataset))
    return dataset


def write_jsonl_to_path(data, file_path):
    print('Num of data:', len(data))
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)


type2aspects = read_jsonl_from_path('./data/priority-aspects.jsonl')[0]
print(type2aspects['coding'])


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

def format_eval(data, judge_prompt, model_name, turn):
    messages = []

    for conv in data:
        if turn == 1:
            sample = {
                "question": conv['user_1'],
                "answer": conv[model_name + '_1'],
                "reference": conv['reference_1'],
                "priority_aspect": type2aspects[conv['type']]    
            }
            template = judge_prompt["seabench-turn1-ref-updated"]
        elif turn == 2:
            sample = {
                "question_1": conv['user_1'],
                "answer_1": conv[model_name + '_1'],
                "question_2": conv['user_2'],
                "answer_2": conv[model_name + '_2'],
                "reference": conv['reference_2'],
                "priority_aspect": type2aspects[conv['type']]    
            }
            template = judge_prompt["seabench-turn2-ref-updated"]
        else:
            raise NotImplementedError

        filled_template = template["prompt_template"].format(**sample)

        message = [
            {"role": "system", "content": template["system_prompt"]},
            {"role": "user", "content": filled_template}
        ]
    
        messages.append(message)
    
    #print('\n\n', messages, '\n\n')
    return messages


def make_judge_single(data, judge_prompts, args):

    # use reference answer by default
    if args.reference_answer:
        print("\nWill judge with reference answer\n")
    else:
        raise NotImplementedError

    for turn in range(2):
        # eval the first turn
        print(f"begin the {turn}-th turn eval...")

        eval_prompts = format_eval(data, judge_prompts, args.testing_model, turn+1)
        # prompt_args = [(api_key, p, args) for p in eval_prompts]
        prompt_args = [(api_key, p, args.judge_model,args.max_tokens, args.temperature ) for p in eval_prompts]

        if args.judge_model_type == 'openai':
            parallel_call = parallel_query_chatgpt_model
        else: 
            parallel_call = parallel_query_chatgpt_model_azure

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            predictions = list(tqdm.tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))
        
        # merge the answers to the data
        for idx in range(len(predictions)):
            data[idx][f'eval_{turn+1}'] = predictions[idx]

        write_jsonl_to_path(data, f"./model_judgement/{args.judge_model}_eval_{args.testing_model}_single_turn{turn}.jsonl")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_file", type=str, default="data/judge_prompts.jsonl")
    parser.add_argument("--judge_model", type=str, default="gpt-4-turbo-2024-04-09")
    parser.add_argument("--judge_model_type",default='openai', choices=['openai', 'azure'],
                        type=str, help='judge model endpoint, openai or azure')
    parser.add_argument("--testing_model", type=str, default="chatgpt")
    parser.add_argument("--baseline_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="single", \
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument("--reference_answer", type=bool, default=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_workers", type=int, default=5)

    args = parser.parse_args()

    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.environ["OPENAI_API_KEY"] if args.judge_model_type == 'openai' else os.environ["AZURE_OPENAI_KEY"]

    # load data (questions, refs, model gens)
    args.testing_model = args.testing_model.split('/')[-1]
    model_name = args.testing_model
    data_path = f"./outputs/{model_name}.jsonl"
    data = read_jsonl_from_path(data_path)
    
    # debug: only take part of the data for checking the sanity
    if args.debug:
        data = random.sample(data, 10)

    # load judge
    judge_prompts_lines = read_jsonl_from_path(args.judge_file)
    judge_prompts = {}
    for p in judge_prompts_lines:
        judge_prompts[p['name']] = p

    os.makedirs("./model_judgement", exist_ok=True)

    if args.mode == "single":
        judges = make_judge_single(data, judge_prompts, args)
        # output_file = f"./model_judgement/gpt4t_eval_{args.testing_model}_single.jsonl"
        output_file = f"./model_judgement/{args.judge_model}_eval_{args.testing_model}_single.jsonl"
        write_jsonl_to_path(judges, output_file)

    else:
        raise NotImplementedError

