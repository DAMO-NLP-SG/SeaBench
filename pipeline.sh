data_path=SeaLLMs/SeaBench
output_dir=outputs
judgement_dir=model_judgement
results_folder=results

judge_model=gpt-4o-2024-08-06
judge_model_type=openai

for testing_model in \
SeaLLMs/SeaLLMs-v3-7B-Chat \
; do

CUDA_VISIBLE_DEVICES=0 python gen_responses.py \
  --model_id $testing_model \
  --model_type default \
  --max_token 2048 \
  --data_path $data_path \
  --output_dir $output_dir

# export OPENAI_API_KEY=xxx  # set the OpenAI API key
CUDA_VISIBLE_DEVICES=0 python gen_judgements.py \
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