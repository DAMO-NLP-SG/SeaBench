from utils_model import *
import argparse
from tqdm import tqdm
import numpy as np
import concurrent.futures
import jsonlines
from datetime import datetime
import json
from datasets import load_dataset

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
    "together": parallel_query_together_model,
    "azure": parallel_query_chatgpt_model_azure,
    "openrouter": parallel_query_openrouter_model,
}

system_message_map = {
    "SeaLLM-7B-v2.5": "You are a helpful assistant.",
    "SeaLLM-7B-v2": "You are a helful assistant.",
    "Sailor-7B-Chat": "You are a helpful assistant.",
    "gpt-3.5-turbo-0125": "You are a helpful assistant.",
    "claude-3-haiku-20240307": "",
    "sea-lion-7b-instruct":"",
    "gemma-2-9b-it":"",
    "gemma-2-27b-it":"",
    "Mistral-7B-Instruct-v0.3":"",
}

dic_api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "azure": os.getenv("AZURE_OPENAI_KEY"),
    "vllm": "vllm",
    "hf": "hf",
    'openrouter': os.getenv("OPENROUTER_API_KEY"),
}


def process_model(questions, model_id, model_type="vllm", system_prompt="", 
                  api_key=None, max_tokens=2048, temperature=0, output_dir='outputs',args=None):
    
    # create output_dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = model_id.split('/')[-1]

    # check whether output file exists
    if os.path.exists(os.path.join(output_dir, f"{model_name}.jsonl")):
        if args.update:
            if args.unit_ids is not None:
                unit_ids = set(args.unit_ids.split(','))
                print(f"Output file {os.path.join(output_dir, f'{model_name}.jsonl')} exists. Updating the responses with unit_ids {unit_ids}")
                questions_old = jsonl_to_list(os.path.join(output_dir, f"{model_name}.jsonl"))
                questions = [q for q in questions if q['unit_id'] in unit_ids]
                questions_old = [q for q in questions_old if q['unit_id'] not in unit_ids]
            else: 
                print(f"Output file {os.path.join(output_dir, f'{model_name}.jsonl')} exists. Updating all the responses")
        else:
            print(f"Output file {os.path.join(output_dir, f'{model_name}.jsonl')} exists. Skipping the model")
            return

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            responses2 = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

    # add responses of two turns to questions
    for i, q in enumerate(questions):
        q[f'{model_name}_1'] = responses[i]
        q[f'{model_name}_2'] = responses2[i]
        # q['prompt'] = list_prompt[i]

    # save the questions to a jsonl file
    if args.update and args.unit_ids is not None:
        questions = questions_old + questions
        questions = sorted(questions, key=lambda x: int(x['unit_id'].split('-')[-1]))
        list_to_jsonl(questions, f'{output_dir}/{model_name}.jsonl')
    else:
        list_to_jsonl(questions, f'{output_dir}/{model_name}.jsonl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on public questions')
    parser.add_argument('--data_path', type=str, default= "SeaLLMs/SeaBench",help='dataset path')
    parser.add_argument('--model_id', type=str, default= "SeaLLMs/SeaLLMs-v3-7B-Chat" ,help='model id')
    parser.add_argument('--model_type', type=str, default="default", choices=["default","vllm","hf", "openai", "azure","claude", "together",'openrouter'], help='model type')
    parser.add_argument('--system_prompt', type=str, default= "",help='system prompt for chat model')
    parser.add_argument('--api_key', type=str, default=None, help='api key')
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument('--max_tokens', type=int, default=2048, help='max tokens')
    parser.add_argument('--temperature', type=float, default=0, help='temperature')
    parser.add_argument('--output_dir', type=str, default='outputs', help='output directory')
    parser.add_argument('--update', type=int, default=0, help='whether update output file')
    parser.add_argument('--unit_ids', type=str, default=None, help='unit ids to update, if None, update all the questions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # guess the detailed model type if not specifically given
    if args.model_type == "default":
        if "gpt-3.5" in args.model_id or "gpt-4" in args.model_id:
            args.model_type = "openai"
        # elif "gemini" in args.model_id:
        #     args.model_type = "gemini"
        # elif "claude" in args.model_id:
        #     args.model_type = "claude"
        elif "gemini" in args.model_id or "claude" in args.model_id:
            args.model_type = "openrouter"
        else:
            args.model_type = "vllm"

    args.api_key = dic_api_keys[args.model_type] if args.api_key is None else args.api_key

    ds = load_dataset(args.data_path)['train']
    questions = ds.to_pandas().to_dict(orient='records')
    
    if args.debug:
        questions = questions[:10]
    
    process_model(questions, args.model_id, args.model_type, args.system_prompt, 
                  args.api_key, args.max_tokens, args.temperature, args.output_dir,args)