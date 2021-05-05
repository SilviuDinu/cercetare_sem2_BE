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

trainCSVPath = os.path.join(curr, r'data/csv-data/processed_training_2.csv')
testCSVPath = os.path.join(curr, r'data/csv-data/processed_testing_2.csv')

training_csv_output = os.path.join(curr, r'data/csv-data/processed_training_2.csv')
testing_csv_output = os.path.join(curr, r'data/csv-data/processed_testing_2.csv')

diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')

raw = []
headers = []
diseases = []


def has_more_than_10(key, set):
    count = 0

    for idx, item in enumerate(set):
        if key == item[len(item) - 1] == key:
            count += 1
    
    if count > 10:
        return True
    return False

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

with open(diseases_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        diseases.append(rows)

full_dataset = raw

# train_size = int(0.9 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


testset = []
trainset = []

traindata = np.loadtxt(training_csv_output, delimiter=',', dtype=np.float32, skiprows=1)
testdata = np.loadtxt(testing_csv_output, delimiter=',', dtype=np.float32, skiprows=1)

for idx, boala in enumerate(diseases):
    key = float(boala[0])
    for i, item in enumerate(raw):
        can_add = not has_more_than_10(key, testset)
        if key == item[len(item) - 1] and can_add:
            testset.append(item)
            del raw[i]

print("length raw {}".format(len(raw)))
print("length testset {}".format(len(testset)))

train_dataset = raw
test_dataset = testset


# print(len([i for i, j in zip(train_dataset, test_dataset) if i == j]))

# for i, test in enumerate(test_dataset):
#     for j, train in enumerate(train_dataset):
#         if test == train:
#             del train_dataset[j]


for i, test in enumerate(test_dataset):
    for j, train in enumerate(train_dataset):
        count = 0
        for k, elem in enumerate(test):
            if elem == train[k]:
                count += 1
        if count == len(test):
            print('i: %s, j: %s' % (i, j))
           


print("length raw {}".format(len(train_dataset)))
print("length testset {}".format(len(test_dataset)))

if not os.path.exists(training_csv_output):
    with open(training_csv_output, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerows(headers)
        write.writerows(train_dataset)
if not os.path.exists(testing_csv_output):
    with open(testing_csv_output, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerows(headers)
        write.writerows(test_dataset)

# train = []

# with open(trainCSVPath, "r") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for idx, rows in enumerate(csv_reader):
#         if idx > 0:
#             train.append(rows)


# test = []

# with open(testCSVPath, "r") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for idx, rows in enumerate(csv_reader):
#         if idx > 0:
#             test.append(rows)

# print(len(test))
# print(len(train))

# print(test[81])
# print(train[4205])
# print(test[81] == train[4205])

# print(len(test))
# print(len(train))

# for i, test in enumerate(test):
#     for j, train in enumerate(train):
#         count = 0
#         for k, elem in enumerate(test):
#             if elem == train[k]:
#                 count += 1
#         if count == len(test):
#             print('i: %s, j: %s' % (i, j))
           