data_path=data/public-questions.jsonl
output_dir=outputs
judgement_dir=model_judgement
results_folder=results

judge_model=gpt-4o-2024-08-06
judge_model_type=openai

for testing_model in \
google/gemma-2-27b-it \
qwen/qwen-2.5-72b-instruct \
qwen/qwen-2-72b-instruct \
meta-llama/llama-3.1-70b-instruct \
meta-llama/llama-3-70b-instruct \
; do

CUDA_VISIBLE_DEVICES=0 python gen_responses.py \
  --model_id $testing_model \
  --model_type openrouter \
  --max_token 2048 \
  --data_path $data_path \
  --output_dir $output_dir 

# export OPENAI_API_KEY_test=xxx  # set the OpenAI API key
python gen_judgements.py \
  --judge_model $judge_model \
  --testing_model $testing_model \
  --mode single \
  --judge_model_type $judge_model_type \
  --response_dir $output_dir \
  --judgement_dir $judgement_dir
done 

python gen_results.py --judge_model $judge_model \
      --judgement_dir $judgement_dir \
      --results_folder $results_folder


# ------------------------------------------------------------------------------
# export AZURE_OPENAI_ENDPOINT=xxx  # set the OpenAI Azure API endpoint
# export AZURE_OPENAI_KEY=xxx  # set the OpenAI Azure API key
# judge_model=gpt-4o-xxx

# python gen_judgements.py \
#   --judge_model $judge_model \
#   --testing_model $testing_model \
#   --mode single \
#   --judge_model_type azure \
#   --response_dir $output_dir \
#   --judgement_dir $judgement_dir

# python gen_results.py --judge_model $judge_model \
#       --judgement_dir $judgement_dir \
#       --results_folder $results_folder