import argparse
import sys
import os
import re
import jsonlines


def read_jsonl_from_path(file_path):
    with jsonlines.open(file_path) as reader:
        dataset = [obj for obj in reader]
    print('Num of data:', len(dataset))
    return dataset


def compute_scores(judgements):
    fail = 0
    for sample in judgements:
        try:
            sample['score_1'] = re.search(r'\[\[(\d*\.*\d*)\]\]', sample['eval_1']).group(1)
        except:
            sample['score_1'] = -1
            print(sample['eval_1'])
            fail += 1
        try:
            sample['score_2'] = re.search(r'\[\[(\d*\.*\d*)\]\]', sample['eval_2']).group(1)
        except:
            sample['score_2'] = -1
            print(sample['eval_2'])
            fail += 1
    langs = set([s['lang'] for s in judgements])
    scores_by_lang = {key: [[], []] for key in langs}

    types = set([s['type'] for s in judgements])
    scores_by_type = {key: [[], []] for key in types}

    for sample in judgements:
        if sample['score_1'] != -1:
            scores_by_lang[sample['lang']][0].append(float(sample['score_1']))
            scores_by_type[sample['type']][0].append(float(sample['score_1']))
        if sample['score_2'] != -1:
            scores_by_lang[sample['lang']][1].append(float(sample['score_2']))
            scores_by_type[sample['type']][1].append(float(sample['score_2']))
    print("number of failure in parsing the eval results:", fail)
    return scores_by_lang, scores_by_type


def summarize_scores(scores_by_lang, scores_by_type):
    print("Overall:")
    first_turn = sum(sum(scores_by_lang[lang][0]) for lang in sorted(scores_by_lang.keys()))
    second_turn = sum(sum(scores_by_lang[lang][1]) for lang in sorted(scores_by_lang.keys()))
    all_len = sum(len(scores_by_lang[lang][0]) for lang in sorted(scores_by_lang.keys()))
    all_len2 = sum(len(scores_by_lang[lang][1]) for lang in sorted(scores_by_lang.keys()))
    first_turn = first_turn / all_len
    second_turn = second_turn / all_len2
    all_turn = (first_turn + second_turn) / 2
    print(f"1st turn - {first_turn:.2f}, 2nd turn - {second_turn:.2f}, avg - {all_turn:.2f}")
    print('-'*30)

    print("Results by lang:")
    for lang in sorted(scores_by_lang.keys()):
        first_turn = sum(scores_by_lang[lang][0])/len(scores_by_lang[lang][0])
        second_turn = sum(scores_by_lang[lang][1])/len(scores_by_lang[lang][1])
        all_turn = sum(scores_by_lang[lang][0] + scores_by_lang[lang][1]) / (len(scores_by_lang[lang][0])+len(scores_by_lang[lang][1]))
        print(f"{lang}: 1st turn - {first_turn:.2f}, 2nd turn - {second_turn:.2f}, avg - {all_turn:.2f}")

    print('-'*30)

    print("Results by types:")
    for type in sorted(scores_by_type.keys()):
        first_turn = sum(scores_by_type[type][0])/len(scores_by_type[type][0])
        second_turn = sum(scores_by_type[type][1])/len(scores_by_type[type][1])
        all_turn = sum(scores_by_type[type][0] + scores_by_type[type][1]) / (len(scores_by_type[type][0])+len(scores_by_type[type][1]))
        print(f"{type:<12}: 1st turn - {first_turn:.2f}, 2nd turn - {second_turn:.2f}, avg - {all_turn:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--judge_model", type=str, default="gpt-4-turbo-2024-04-09")
    parser.add_argument("--testing_model", type=str, default="chatgpt")
    
    args = parser.parse_args()

    args.testing_model = args.testing_model.split('/')[-1]
    judgement_file = f'./model_judgement/{args.judge_model}_eval_{args.testing_model}_single.jsonl'
    # judgement_file = f'./model_judgement/gpt4t_eval_{args.testing_model}_single.jsonl'
    judgements = read_jsonl_from_path(judgement_file)

    scores_by_lang, scores_by_type = compute_scores(judgements)
    summarize_scores(scores_by_lang, scores_by_type)
