# SeaBench: Benchmarking LLMs for Southeast Aisa languages with Open-ended Questions

<p align="center">
<a href="https://huggingface.co/spaces/SeaLLMs/SeaExam_leaderboard" target="_blank" rel="noopener"> 🤗 Leaderboard</a>
</p>

This repository contains code and data for SeaBench, a comprehensive benchmark designed to evaluate large language models' (LLMs) capabilities in Southeast Asian (SEA) languages. Specifically, SeaBench assesses models' multi-turn and instruction-following capabilities across Indonesian, Thai, and Vietnamese languages through carefully crafted evaluation tasks.

# 1. Data
All the data is under the `data` folder. Currently, only `public-questions.jsonl` here (private questions are hidden).

# Evaluation
## 1. run inference to get model's prediction
You need to first generate model's response, you can directly run `python gen_responses.py`.
Currently it supports some models (both open-source or commercial), feel free to add support for models.

Example:
```
python gen_responses.py --model_id SeaLLMs/SeaLLMs-v3-7B-Chat
```

Please pay attention to:
* different models have different ways to express `system_prompt`, `user_turn`, and `assistant_turn`
* it needs to at least support a two-turn inference

Either way, the model prediction would be in `./outputs/{model_name}.jsonl`
* It should add two keys for each row compared to `publuc-questions.jsonl`: modelname_1 and modelname_2, which are the model's responses at the 1st and 2nd turn

## 2. judge model evaluation
run `python gen_judgements.py --testing_model model_name`
* by default, it will use gpt-4o-2024-08-06 as the evaluator
* the predictions will be written to `./model_judgement/` directory

## 3. extract and summarize the results
run `python gen_results.py` will give the results of a certain model

## Pipeline
You can also specify the testing_model, judge_model and OPENAI_API_KEY in pipeline.sh and quickly run
```
source pipeline.sh
```

# Leaderboard
You can find our interactive leaderboard [🤗 Here](https://huggingface.co/spaces/SeaLLMs/SeaExam_leaderboard). The leaderboard showcases results from two complementary benchmarks: [SeaExam](https://github.com/DAMO-NLP-SG/SeaExam) and [SeaBench](https://github.com/DAMO-NLP-SG/SeaBench). Each benchmark evaluates different aspects of model capabilities through distinct question types, providing a comprehensive assessment of model performance.

# Citation
If you find SeaBench useful for your research, please consider citing our papers:
```
@article{damonlp2024seallm3,
  author = {Wenxuan Zhang*, Hou Pong Chan*, Yiran Zhao*, Mahani Aljunied*,
            Jianyu Wang*, Chaoqun Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu,
            Yew Ken Chia, Xin Li, Lidong Bing},
  title = {SeaLLMs 3: Open Foundation and Chat Multilingual Large Language Models for Southeast Asian Languages},
  year = {2024},
  url = {https://arxiv.org/abs/2407.19672}
}

@article{damonlpsg2023seallm,
  author = {Xuan-Phi Nguyen*, Wenxuan Zhang*, Xin Li*, Mahani Aljunied*,
            Zhiqiang Hu, Chenhui Shen, Yew Ken Chia, Xingxuan Li, Jianyu Wang,
            Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang,
            Chaoqun Liu, Hang Zhang, Lidong Bing},
  title = {SeaLLMs - Large Language Models for Southeast Asia},
  year = {2024},
  booktitle = {ACL 2024 System Demonstrations},
  url = {https://arxiv.org/pdf/2312.00738},
}
```
