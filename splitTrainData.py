import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import os
import torch.nn.functional as F
from buildTrainData import *
from buildTestData import *
from net import Net
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import numpy as np

curr = os.path.dirname(__file__)

trainCSVPath = os.path.join(curr, r'data/csv-data/processed_training.csv')
testCSVPath = os.path.join(curr, r'data/csv-data/processed_testing.csv')

training_csv_output = os.path.join(curr, r'data/csv-data/processed_training_2.csv')
testing_csv_output = os.path.join(curr, r'data/csv-data/processed_testing_2.csv')

diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')

raw = []
headers = []

with open(trainCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            raw.append(rows)
        else:
            headers.append(rows)

with open(testCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            raw.append(rows)

full_dataset = raw

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# print(len([i for i, j in zip(train_dataset, test_dataset) if i == j]))

with open(training_csv_output, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerows(headers)
    write.writerows(train_dataset)
with open(testing_csv_output, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerows(headers)
    write.writerows(test_dataset)