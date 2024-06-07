from utils_model import *
import argparse
from tqdm import tqdm
import numpy as np
import concurrent.futures
import jsonlines
from datetime import datetime
import json
now = datetime.now()
currentDateTime = now.strftime("%d/%m/%Y")


def jsonl_to_list(path):
    with jsonlines.open(path) as reader:
        dataset = [obj for obj in reader]
    return dataset


def list_to_jsonl(dataset,path):
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(dataset)


def print_json(obj):
    print(json.dumps(obj, indent=2))


dic_api_functions = {
    "openai": parallel_query_chatgpt_model,
    "claude": parallel_query_claude_model,
    "together": parallel_query_together_model
}

system_message_map = {
    "SeaLLM-7B-v2.5": "You are a helpful assistant.",
    "SeaLLM-7B-v2": "You are helful assistant.",
    "Sailor-7B-Chat": "You are a helpful assistant.",
    "gpt-3.5-turbo-0125": "You are a helpful assistant.",
    "claude-3-haiku-20240307": ""
}

dic_api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "azure": os.getenv("AZURE_OPENAI_KEY"),
    "vllm": "vllm",
    "hf": "hf"
}


def process_model(questions, model_id, model_type="vllm", system_prompt="", 
                  api_key=None, max_tokens=2048, temperature=0, output_dir='outputs'):
    
    # create output_dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = model_id.split('/')[-1]
    if model_type == "vllm":
        # run model locally with vllm
        llm, sampling_params, tokenizer = prepare_vllm(model_id, max_tokens=max_tokens, temperature=temperature)
    elif model_type == "hf":
        # run model locally with hf
        llm, tokenizer = prepare_hf_model(model_id)
    else:
        # call model api
        parallel_call = dic_api_functions[model_type]

    # get system prompt
    if system_prompt == '':
        default_system_prompt = 'You are a helpful assistant.'
        system_prompt = system_message_map.get(model_name, default_system_prompt)
    
    print(system_prompt)
    list_message = [input_to_messages("system", system_prompt, [], model_id=model_id) 
                    if system_prompt != "" else []
                    for _ in range(len(questions))]

    # turn 1 
    list_input = [q['user_1'] for q in questions]
    list_message = [input_to_messages("user", prompt, message, model_id=model_id) for prompt, message in zip(list_input, list_message)]
    
    print_json(list_message[0])
    # exit()
    if model_type == "vllm":
        list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
        responses = get_vllm_completion(llm, list_prompt, sampling_params)
    elif model_type == "hf":
        list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
        responses = get_hf_completion(llm, tokenizer, list_prompt, max_new_tokens=max_tokens, batch_size=1)
    else:
        prompt_args = [(api_key, message, model_id, max_tokens, temperature) for message in list_message]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

    # save model generation as assistant
    list_message = [input_to_messages("assistant", prompt, message, model_id=model_id) for prompt, message in zip(responses, list_message)]

    # turn 2
    list_message = [input_to_messages("user", prompt_for_ans, message, model_id=model_id) for prompt_for_ans, message in zip([q['user_2'] for q in questions], list_message)]
    print_json(list_message[0])
    if model_type == "vllm":
        list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
        responses2 = get_vllm_completion(llm, list_prompt, sampling_params)
    elif model_type == "hf":
        list_prompt = [messages_to_prompt(message, tokenizer) for message in list_message]
        responses2 = get_hf_completion(llm, tokenizer, list_prompt, max_new_tokens=max_tokens, batch_size=1)
    else:
        prompt_args = [(api_key, message, model_id, max_tokens, temperature) for message in list_message]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            responses2 = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

    # add responses of two turns to questions
    for i, q in enumerate(questions):
        q[f'{model_name}_1'] = responses[i]
        q[f'{model_name}_2'] = responses2[i]
        # q['prompt'] = list_prompt[i]

    # save the questions to a jsonl file
    list_to_jsonl(questions, f'{output_dir}/{model_name}.jsonl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on public questions')
    parser.add_argument('--data_path', type=str, default= "data/public-questions.jsonl",help='dataset path')
    parser.add_argument('--model_id', type=str, default= "SeaLLMs/SeaLLM-7B-v2.5" ,help='model id')
    parser.add_argument('--model_type', type=str, default="default", choices=["default","vllm","hf", "openai", "claude", "together"], help='model type')
    parser.add_argument('--system_prompt', type=str, default= "",help='system prompt for chat model')
    parser.add_argument('--api_key', type=str, default=None, help='api key')
    parser.add_argument('--max_tokens', type=int, default=2048, help='max tokens')
    parser.add_argument('--temperature', type=float, default=0, help='temperature')
    parser.add_argument('--output_dir', type=str, default='outputs', help='output directory')
    args = parser.parse_args()

    # guess the detailed model type if not specifically given
    if args.model_type == "default":
        if "gpt-3.5" in args.model_id or "gpt-4" in args.model_id:
            args.model_type = "openai"
        elif "gemini" in args.model_id:
            args.model_type = "gemini"
        elif "claude" in args.model_id:
            args.model_type = "claude"
        elif 'aisingapore/sea-lion-7b' in args.model_id:
            # ["aisingapore/sea-lion-7b", "aisingapore/sea-lion-7b-instruct"]
            args.model_type = "hf"
        else:
            args.model_type = "vllm"

    args.api_key = dic_api_keys[args.model_type] if args.api_key is None else args.api_key

    # todo - add hf dataset loading
    questions = jsonl_to_list(args.data_path)
    # print(questions[0])
    # questions = questions[:2]
    
    process_model(questions, args.model_id, args.model_type, args.system_prompt, 
                  args.api_key, args.max_tokens, args.temperature, args.output_dir)