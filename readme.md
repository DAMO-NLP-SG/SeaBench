
# 1. Data
All the data is under the `data` folder. Currently, only `public-questions.jsonl` here (private questions are hidden)

# 2. run inference to get model's prediction
You need to first generate model's response, you can directly run `python gen_responses.py`.
Currently it supports some models (both open-source or commercial), feel free to add support for models.

Example: 
```
python gen_responses.py --model_id SeaLLMs/SeaLLM-7B-v2.5 
```

Please pay attention to:
* different models have different ways to express `system_prompt`, `user_turn`, and `assistant_turn`
* it needs to at least support a two-turn inference

Either way, the model prediction would be in `./outputs/model_name.jsonl`
* It should add two keys for each row compared to `publuc-questions.jsonl`: modelname_1 and modelname_2, which are the model's responses at the 1st and 2nd turn

# 3. run model evaluation
run `python gen_judgements.py --testing_model model_name`
* by default, it will use gpt-4-turbo as the evaluator
* the predictions will be written to `./model_judgement/` directory

# 4. extract and summarize the results
run `python gen_results.py` will give the results of a certain model

# Pipeline
You can also specify the testing_model, judge_model and OPENAI_API_KEY in pipeline.sh and quickly run
```
source pipeline.sh
```