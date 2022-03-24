from email import header
import os
import csv
import torch
import torchvision
from torchvision import transforms
import sys
import numpy as np

curr = os.path.dirname(__file__)

depression_simptoms = [
    'sadness',
    'anger',
    'irritability',
    'sleep_disturbances',
    'tiredness',
    'loss_of_appetite',
    'increased_appetite',
    'weight_loss',
    'weight_gain',
    'anxiety',
    'guilt',
    'lack_of_concentration',
    'trouble_remembering_things',
    'thoughts_of_death',
    'back_pain',
    'headache',
    'loss_of_interest'
]

difference = [
    'sadness',
    'anger',
    'sleep_disturbances',
    'tiredness',
    'guilt',
    'trouble_remembering_things',
    'thoughts_of_death',
    'loss_of_interest'
]


def build_processed_file_for_depression(input_path, output_path):
    raw = []
    headers = []
    with open(input_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, rows in enumerate(csv_reader):
            if idx > 0:
                original = list(map(int, rows[:-1]))
                original.extend([0] * 8)
                original.extend([rows[len(rows)-1]])
                raw.append(original)
            else:
                headers = rows[:-1]
                for idx1, symptom in enumerate(rows):
                    for idx2, depression_symptom in enumerate(depression_simptoms):
                        if depression_symptom not in headers:
                            headers.append(depression_symptom)
                headers.append(rows[len(rows)-1])

    headers = [headers]

    with open(output_path, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerows(headers)
        write.writerows(raw)


trainCSVPath = os.path.join(curr, r'data/csv-data/Training_depression.csv')
training_csv_output = os.path.join(curr, r'data/csv-data/Training_despression_processed.csv')

build_processed_file_for_depression(trainCSVPath, training_csv_output)

testCSVPath = os.path.join(curr, r'data/csv-data/Testing_depression.csv')
testing_csv_output = os.path.join(curr, r'data/csv-data/Testing_despression_processed.csv')

build_processed_file_for_depression(testCSVPath, testing_csv_output)
