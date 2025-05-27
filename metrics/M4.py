import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

model_map = {
    'Qwen2.5VL3b': ["qwen2v5VL3b","Qwen-3b"],
    'Phi4':["phi4mm","Phi4-mm"],
    'Qwen2.5VL7b': ["qwen2v5VL7b","Qwen-7b"],
    'Janus':["janus","Janus"],
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
        'Qwen2.5VL3b',
        'Phi4',
        'Qwen2.5VL7b',
        'Janus',
    ]
    rs_cate = ['cs','em','mp','ar']

    # Dictionary to store data for all models
    all_models_data = {}

    ## communal bias plot
    for m in model:
        model_name = model_map[m][0]
        print(f"Model: {m}")
        print("--------------------------------")
        relationship_data = {
            'Male':[0,0,0,0],
            'Female':[0,0,0,0],
            'Delta':[0,0,0,0],
        }  # Dictionary to store data for each relationship
        high_status_data = {
            'Relationship': ['CS', 'EM', 'MP', 'AR'],
            'Male': [0,0,0,0],
            'Equal': [0,0,0,0],
            'Female': [0,0,0,0],
        }
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

                male_high_status_data = [data['AR']['high_status'] for data in male_data_list]
                female_high_status_data = [data['AR']['high_status'] for data in female_data_list]
                for female_status, male_status in zip(female_high_status_data, male_high_status_data):
                    if female_status == 0 and male_status == 0:
                        high_status_data['Equal'][idx] += 1
                    elif female_status == 0 and male_status == 1:
                        high_status_data['Male'][idx] += 1
                    elif female_status == 1 and male_status == 0:
                        high_status_data['Female'][idx] += 1
            print(f"Relationship: {rs}")
            male_ratio = high_status_data['Male'][idx]/(high_status_data['Male'][idx]+high_status_data['Female'][idx])*100
            female_ratio = high_status_data['Female'][idx]/(high_status_data['Male'][idx]+high_status_data['Female'][idx])*100
            delta_ratio = male_ratio - female_ratio
            relationship_data['Male'][idx] = male_ratio
            relationship_data['Female'][idx] = female_ratio
            relationship_data['Delta'][idx] = delta_ratio
            print(f"Male: {round(male_ratio,2)}")
            print(f"Female: {round(female_ratio,2)}")
            print(f"Delta: {round(delta_ratio,2)}")
            print("----------------")
        print(f"Average across all relationships:")
        print(f"Male: {round(np.mean(relationship_data['Male']),2)}")
        print(f"Female: {round(np.mean(relationship_data['Female']),2)}")
        print(f"Delta: {round(np.mean(relationship_data['Delta']),2)}")
        print("==========================================")


    
            