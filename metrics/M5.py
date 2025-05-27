import argparse
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
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
    
    ## emotion bias plot
    for m in model:
        print(f"Model: {m}")
        print("--------------------------------")
        model_name = model_map[m][0]
        male_overall_emodata_count = []
        female_overall_emodata_count = []
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

            # Calculate emotion stats for this relationship
            female_emo_data_count = [data['EE']['n_emotion_word'][1] for data in female_data_list]
            male_emo_data_count = [data['EE']['n_emotion_word'][1] for data in male_data_list]
            male_overall_emodata_count.extend(male_emo_data_count)
            female_overall_emodata_count.extend(female_emo_data_count)
            print(f"Relationship: {rs}")
            print(f"Male: {round(np.mean(male_emo_data_count),2)}")
            print(f"Female: {round(np.mean(female_emo_data_count),2)}")
            print(f"Delta: {round(np.mean(male_emo_data_count) - np.mean(female_emo_data_count),2)}")
            print("----------------")
        print(f"Average across all relationships:")
        print(f"Male: {round(np.mean(male_overall_emodata_count),2)}")
        print(f"Female: {round(np.mean(female_overall_emodata_count),2)}")
        print(f"Delta: {round(np.mean(male_overall_emodata_count) - np.mean(female_overall_emodata_count),2)}")
        print("==========================================")


            