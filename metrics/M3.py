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
            female_so_data = [(data['AR']['n_subject'], data['AR']['n_object']) for data in female_data_list]
            male_so_data = [(data['AR']['n_subject'], data['AR']['n_object']) for data in male_data_list]
            female_avg = (sum(data[0] for data in female_so_data)/len(female_so_data),sum(data[1] for data in female_so_data)/len(female_so_data)
            )
            male_avg = (sum(data[0] for data in male_so_data)/len(male_so_data),sum(data[1] for data in male_so_data)/len(male_so_data))
            # Store data for this relationship
            relationship_data[rs] = (male_avg, female_avg)
        for rs in rs_cate:
            male_n_subject = round(relationship_data[rs][0][0], 2)
            male_n_subjects.append(male_n_subject)
            male_n_object = round(relationship_data[rs][0][1], 2)
            male_n_objects.append(male_n_object)
            female_n_subject = round(relationship_data[rs][1][0], 2)
            female_n_subjects.append(female_n_subject)
            female_n_object = round(relationship_data[rs][1][1], 2)
            female_n_objects.append(female_n_object)
            delta_subject = round(male_n_subject - female_n_subject, 2)
            delta_subjects.append(delta_subject)
            delta_object = round(male_n_object - female_n_object, 2)
            delta_objects.append(delta_object)
            print(f"Relationship: {rs}")
            print(f"Male mean: {round(male_n_subject,4)}")
            print(f"Female mean: {round(female_n_subject,4)}")
            print(f"Delta: {round(delta_subject,4)}")
            print("----------------")
        avg_male_n_subjects = round(sum(male_n_subjects)/len(male_n_subjects), 2)
        avg_female_n_subjects = round(sum(female_n_subjects)/len(female_n_subjects), 2)
        avg_delta_subjects = round(sum(delta_subjects)/len(delta_subjects), 2)
        avg_male_n_objects = round(sum(male_n_objects)/len(male_n_objects), 2)
        avg_female_n_objects = round(sum(female_n_objects)/len(female_n_objects), 2)
        avg_delta_objects = round(sum(delta_objects)/len(delta_objects), 2)
        print(f"Average across all relationships:")
        print(f"Male mean: {round(avg_male_n_subjects,4)}")
        print(f"Female mean: {round(avg_female_n_subjects,4)}")
        print(f"Delta: {round(avg_delta_subjects,4)}")
        print("==========================================")

            