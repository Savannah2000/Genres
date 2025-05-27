import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

model_map = {
    'Janus':"janus",
    'Phi4':"phi4mm",
    'Qwen2.5VL3b': "qwen2v5VL3b",
    'Qwen2.5VL7b': "qwen2v5VL7b",
}

male_names = [
            'Levi', 'Henry', 'William', 'Oliver', 'Jack', 'Michael', 'Elijah',
            'Noah', 'Theodore', 'Samuel', 'Liam', 'James', 'Mateo', 'Lucas', 'Benjamin'
        ]

female_names = [
            'Mila', 'Emma', 'Eleanor', 'Evelyn', 'Sofia', 'Elizabeth', 'Luna',
            'Olivia', 'Scarlett', 'Amelia', 'Charlotte', 'Amelia', 'Isabella', 'Ava', 'Mia'
        ]

def safe_float(x):
    try:
        if isinstance(x, str) and x.strip().lower() == 'na':
            return 0.0
        return float(x)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    model = [
        'Janus',
        'Phi4',
        'Qwen2.5VL3b',
        'Qwen2.5VL7b',
    ]
    rs_cate = ['cs','em','mp','ar']

    ## subject-object sentences count
    for m in model:
        model_name = model_map[m]
        print(f"Model: {m}")
        print("--------------------------------")
        relationship_data = {}  # Dictionary to store data for each relationship
        male_n_subjects = []
        female_n_subjects = []
        delta_subjects = []
        male_n_objects = []
        female_n_objects = []
        delta_objects = []
        print("Relationship & Male & Female & Delta")
        for idx, rs in enumerate(rs_cate):
            ana_file_path = f"../analysis/{m}/analysis_{model_name}_mf_{rs}.jsonl"
            male_data_list = []
            female_data_list = []
            with open(ana_file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    names = data['names']
                    if names[0] in male_names:
                        male_key = f'male_{names[0]}_analysis'
                        male_data = data[male_key]
                        female_key = f'female_{names[1]}_analysis'
                        female_data = data[female_key]
                    else:
                        male_key = f'male_{names[1]}_analysis'
                        male_data = data[male_key]
                        female_key = f'female_{names[0]}_analysis'
                        female_data = data[female_key]
                    male_data_list.append(male_data)
                    female_data_list.append(female_data)
            female_sentence_positive_ratio_data = [data['EE']['n_positive']/(data['EE']['n_positive']+data['EE']['n_negative'])*100 if data['EE']['n_positive']+data['EE']['n_negative'] > 0 else -1 for data in female_data_list]
            female_sentence_positive_ratio_data = [x for x in female_sentence_positive_ratio_data if x != -1]
            male_sentence_positive_ratio_data = [data['EE']['n_positive']/(data['EE']['n_positive']+data['EE']['n_negative'])*100 if data['EE']['n_positive']+data['EE']['n_negative'] > 0 else -1 for data in male_data_list]
            male_sentence_positive_ratio_data = [x for x in male_sentence_positive_ratio_data if x != -1]
            female_avg = round(sum(female_sentence_positive_ratio_data)/len(female_sentence_positive_ratio_data), 2)
            male_avg = round(sum(male_sentence_positive_ratio_data)/len(male_sentence_positive_ratio_data), 2)

            print(f'{rs.upper()} & {male_avg} & {female_avg} & {round(male_avg-female_avg, 2)}')
            relationship_data[rs] = (male_avg, female_avg)
            
            # Calculate and print model averages after processing all relationships
            if idx == len(rs_cate) - 1:  # On last relationship
                male_model_avg = round(sum(data[0] for data in relationship_data.values()) / len(relationship_data), 2)
                female_model_avg = round(sum(data[1] for data in relationship_data.values()) / len(relationship_data), 2)
                print(f'Avg. & {male_model_avg} & {female_model_avg} & {round(male_model_avg-female_model_avg, 2)}')

            