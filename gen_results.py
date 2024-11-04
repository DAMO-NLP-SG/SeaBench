import argparse
import sys
import os
import re
import jsonlines
import pandas as pd


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

    dic_results_lang = {}
    print("Results by lang:")
    for lang in sorted(scores_by_lang.keys()):
        first_turn = sum(scores_by_lang[lang][0])/len(scores_by_lang[lang][0])
        second_turn = sum(scores_by_lang[lang][1])/len(scores_by_lang[lang][1])
        all_turn = sum(scores_by_lang[lang][0] + scores_by_lang[lang][1]) / (len(scores_by_lang[lang][0])+len(scores_by_lang[lang][1]))
        print(f"{lang}: 1st turn - {first_turn:.2f}, 2nd turn - {second_turn:.2f}, avg - {all_turn:.2f}")
        dic_results_lang[f'{lang}_1'] = first_turn
        dic_results_lang[f'{lang}_2'] = second_turn
        dic_results_lang[f'{lang}_avg'] = all_turn        

    print('-'*30)

    dic_results_type = {}
    print("Results by types:")
    for type in sorted(scores_by_type.keys()):
        first_turn = sum(scores_by_type[type][0])/len(scores_by_type[type][0])
        second_turn = sum(scores_by_type[type][1])/len(scores_by_type[type][1])
        all_turn = sum(scores_by_type[type][0] + scores_by_type[type][1]) / (len(scores_by_type[type][0])+len(scores_by_type[type][1]))
        print(f"{type:<12}: 1st turn - {first_turn:.2f}, 2nd turn - {second_turn:.2f}, avg - {all_turn:.2f}")
        dic_results_type[f'{type}_1'] = first_turn
        dic_results_type[f'{type}_2'] = second_turn
        dic_results_type[f'{type}_avg'] = all_turn

    return dic_results_lang, dic_results_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--judge_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument("--judgement_dir", type=str, default='./model_judgement/')  
    # parser.add_argument("--testing_model", type=str, default="chatgpt")
    args = parser.parse_args()

    args.judge_model = args.judge_model.split('/')[-1]

    # args.testing_model = args.testing_model.split('/')[-1]
    # judgement_file = f'./model_judgement/{args.judge_model}_eval_{args.testing_model}_single.jsonl'
    # judgement_file = f'./model_judgement/gpt4t_eval_{args.testing_model}_single.jsonl'

    # find all eval files ending with _single.jsonl
    judgement_folder = args.judgement_dir
    judgement_files = [f for f in os.listdir(judgement_folder) if f.endswith('_single.jsonl') and f.startswith(args.judge_model)]

    list_results_lang = []
    list_results_lang_avg = []
    list_results_type = []
    list_results_type_avg = []
    for eval_file in judgement_files:
        print(eval_file)
        testing_model = eval_file.split('_')[2]
        judge_model = eval_file.split('_')[0]
        
        judgements = read_jsonl_from_path(judgement_folder +'/'+ eval_file)

        scores_by_lang, scores_by_type = compute_scores(judgements)
        dic_results_lang, dic_results_type = summarize_scores(scores_by_lang, scores_by_type)
        dic_results_type_avg = {k: v for k, v in dic_results_type.items() if '_avg' in k}
        dic_results_lang_avg = {k: v for k, v in dic_results_lang.items() if '_avg' in k}

        print('*'*50)
        
        dic_results_lang['model'] = testing_model
        dic_results_lang['judge_model'] = judge_model
        dic_results_lang_avg['model'] = testing_model
        dic_results_lang_avg['judge_model'] = judge_model
        dic_results_type['model'] = testing_model
        dic_results_type['judge_model'] = judge_model
        dic_results_type_avg['model'] = testing_model
        dic_results_type_avg['judge_model'] = judge_model

        list_results_lang.append(dic_results_lang)
        list_results_type.append(dic_results_type)
        list_results_type_avg.append(dic_results_type_avg)
        list_results_lang_avg.append(dic_results_lang_avg)


    # save the two list to df and save to csv
    df_results_lang = pd.DataFrame(list_results_lang)
    df_results_type = pd.DataFrame(list_results_type)
    df_results_type_avg = pd.DataFrame(list_results_type_avg)
    df_results_lang_avg = pd.DataFrame(list_results_lang_avg)

    # put 'model' and 'judge_model' to the first two columns
    def move_columns(df):
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        return df[cols]
    
    df_results_lang = move_columns(df_results_lang)
    df_results_type = move_columns(df_results_type)
    df_results_type_avg = move_columns(df_results_type_avg)
    df_results_lang_avg = move_columns(df_results_lang_avg)

    df_results_lang_avg['avg'] = df_results_lang_avg[['id_avg','th_avg','vi_avg']].mean(axis=1)
    df_results_lang_avg = df_results_lang_avg.sort_values(by='avg', ascending=False)

    # save to results folder
    os.makedirs(args.results_folder, exist_ok=True)
    # df_results_lang.to_csv(args.results_folder + 'results_lang.csv', index=False)
    # df_results_type.to_csv(args.results_folder + 'results_type.csv', index=False)
    # df_results_type_avg.to_csv(args.results_folder + 'results_type_avg.csv', index=False)
    # df_results_lang_avg.to_csv(args.results_folder + 'results_lang_avg.csv', index=False)

    df_results_lang.to_csv(f"{args.results_folder}/results_lang_{args.judge_model}.csv", index=False)
    df_results_type.to_csv(f"{args.results_folder}/results_type_{args.judge_model}.csv", index=False)
    df_results_type_avg.to_csv(f"{args.results_folder}/results_type_avg_{args.judge_model}.csv", index=False)
    df_results_lang_avg.to_csv(f"{args.results_folder}/results_lang_avg_{args.judge_model}.csv", index=False)

