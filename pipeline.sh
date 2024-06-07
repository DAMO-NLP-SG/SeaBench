testing_model=SeaLLMs/SeaLLM-7B-v2.5
judge_model=gpt-4-turbo-2024-04-09
export OPENAI_API_KEY=xxx  # set the OpenAI API key

CUDA_VISIBLE_DEVICES=0 python gen_responses.py \
  --model_id $testing_model \
  --model_type vllm \
  --system_prompt "You are helful assistant." \
  --max_token 2048

export OPENAI_API_KEY=xxx  # set the OpenAI API key
python gen_judgements.py \
  --judge_model $judge_model \
  --testing_model $testing_model \
  --mode single \
  --judge_model_type openai

python gen_results.py \
--testing_model $testing_model \
--judge_model $judge_model

#### Use Azure API instead of OpenAI API
# export AZURE_OPENAI_ENDPOINT=xxx # set the OpenAI Azure API endpoint
# export AZURE_OPENAI_KEY=xxx  # set the OpenAI Azure API key
# python gen_judgements.py \
#   --judge_model gpt4-1106 \
#   --testing_model SeaLLMs/SeaLLM-7B-v2.5 \
#   --mode single \
#   --judge_model_type azure

# python gen_results.py \
# --testing_model SeaLLMs/SeaLLM-7B-v2.5 \
# --judge_model gpt4-1106
