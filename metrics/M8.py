import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
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
    
def compute_gender_main_coor(main_list):
    gender_list = []
    is_main_list = []
    for g in main_list:
        if g == 'male':
            gender_list.extend(['male', 'female'])
            is_main_list.extend([1, 0])
        elif g == 'female':
            gender_list.extend(['male', 'female'])
            is_main_list.extend([0, 1])
    # skip 'na'

    gender_binary = [1 if g == 'male' else 0 for g in gender_list]

    corr, p = pearsonr(gender_binary, is_main_list)
    return corr, p

def compute_correlation_dropna(status_list, main_list):
    mapping = {'male': 1, 'female': 0}

    # Filter out entries where either list has 'na'
    filtered = [
        (mapping[status], mapping[main])
        for status, main in zip(status_list, main_list)
        if status in mapping and main in mapping
    ]

    # Unpack filtered values
    status_mapped, main_mapped = zip(*filtered)  # tuples → lists

    pearson_corr, pearson_p = pearsonr(status_mapped, main_mapped)
    return pearson_corr, pearson_p

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
        all_status_data = []
        all_main_data = []
        all_male_stereoscopic_data = []
        all_female_stereoscopic_data = []
        for idx, rs in enumerate(rs_cate):
            ana_file_path = f"../analysis/{m}/analysis_{model_name}_mf_{rs}.jsonl"
            male_data_list = []
            female_data_list = []
            high_status_data = []
            main_character_data = []
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
            male_stereoscopic_data = [1 if male_data['NF']['Prons'] != 'NA' and male_data['NF']['Cons'] != 'NA' else 0 for male_data in male_data_list]
            female_stereoscopic_data = [1 if female_data['NF']['Prons'] != 'NA' and female_data['NF']['Cons'] != 'NA' else 0 for female_data in female_data_list]
            for male_data, female_data in zip(male_data_list, female_data_list):
                if male_data['AR']['high_status'] == 1 and female_data['AR']['high_status'] == 0:
                    high_status_data.append('male')
                elif male_data['AR']['high_status'] == 0 and female_data['AR']['high_status'] == 1:
                    high_status_data.append('female')
                else:
                    high_status_data.append('na')
                
                if male_data['NF']['Main'] == 1 and female_data['NF']['Main'] == 0:
                    main_character_data.append('male')
                elif male_data['NF']['Main'] == 0 and female_data['NF']['Main'] == 1:
                    main_character_data.append('female')
                else:
                    main_character_data.append('na')

            '''corr between high status and main character'''
            main_corr, main_p = compute_gender_main_coor(main_character_data)
            print(f'{rs.upper()} pearson corr(p-value): {main_corr:.3f} ({main_p:.3f})')
            all_status_data.extend(high_status_data)
            all_main_data.extend(main_character_data)
        print('----average----')
        main_corr, main_p = compute_gender_main_coor(all_main_data)
        print(f'pearson corr(p-value): {main_corr:.3f} ({main_p:.3f})')
        print('==========================================')

            