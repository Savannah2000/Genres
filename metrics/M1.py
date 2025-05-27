import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
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


    ## communal bias plot
    for m in model:
        os.makedirs(f'./figures/{m}', exist_ok=True)
        model_name = model_map[m]
        relationship_data = {}  # Dictionary to store data for each relationship
        print(f"Model: {m}")
        print("--------------------------------")
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

            # Calculate communal ratios for this relationship
            female_warmth_data = [safe_float(data['PA']['Warmth'])/(safe_float(data['PA']['Warmth'])+safe_float(data['PA']['Competence'])) 
                                  for data in female_data_list if data['PA']['Warmth']+data['PA']['Competence'] != 0]
            male_warmth_data = [safe_float(data['PA']['Warmth'])/(safe_float(data['PA']['Warmth'])+safe_float(data['PA']['Competence'])) 
                                for data in male_data_list if data['PA']['Warmth']+data['PA']['Competence'] != 0]
            
            # Store data for this relationship
            relationship_data[rs] = (male_warmth_data, female_warmth_data)
            print(f"Relationship: {rs}")
            print(f"Male mean: {round(np.mean(male_warmth_data)*100,4)}")
            print(f"Female mean: {round(np.mean(female_warmth_data)*100,4)}")
            print(f"Delta: {round((np.mean(male_warmth_data) - np.mean(female_warmth_data))*100,4)}")
            print("--------------------------------")
        
        # Calculate averages across all relationships
        all_male_means = []
        all_female_means = []
        for male_data, female_data in relationship_data.values():
            all_male_means.append(np.mean(male_data))
            all_female_means.append(np.mean(female_data))
        
        avg_male = np.mean(all_male_means)
        avg_female = np.mean(all_female_means)
        avg_delta = avg_male - avg_female
        
        print(f"Average across all relationships:")
        print(f"Male mean: {round(avg_male*100,4)}")
        print(f"Female mean: {round(avg_female*100,4)}")
        print(f"Delta: {round(avg_delta*100,4)}")
        print("==========================================")

        

            