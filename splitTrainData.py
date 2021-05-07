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
from random import randrange, randint

curr = os.path.dirname(__file__)

trainCSVPath = os.path.join(curr, r'data/csv-data/processed_training.csv')
testCSVPath = os.path.join(curr, r'data/csv-data/processed_testing.csv')

training_csv_output = os.path.join(curr, r'data/csv-data/processed_training_2.csv')
testing_csv_output = os.path.join(curr, r'data/csv-data/processed_testing_2.csv')

diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')

raw = []
headers = []
diseases = []

testset = []
trainset = []


def has_more_than_10(key, set):
    count = 0

    for idx, item in enumerate(set):
        if float(key) == float(item[len(item) - 1]):
            count += 1
    rand = randint(5,10)
    if count > 3:
        return True
    return False

with open(trainCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            trainset.append(rows)
        else:
            headers.append(rows)

with open(testCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            testset.append(rows)

with open(diseases_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        diseases.append(rows)


for idx, boala in enumerate(diseases):
    # print(diseases)
    key = float(boala[0])
    for i, item in enumerate(trainset):
        # print(key)
        can_add = not has_more_than_10(key, testset)
        if key == float(item[len(item) - 1]) and can_add:
            testset.append(item)
            del trainset[i]

print("length raw {}".format(len(trainset)))
print("length testset {}".format(len(testset)))

train_dataset = trainset
test_dataset = testset

# print(len([i for i, j in zip(train_dataset, test_dataset) if i == j]))
# test_array = np.array(test_dataset, dtype=float)
# train_array = np.array(train_dataset, dtype=float)
# for i, test in enumerate(test_array):
#     for j, train in enumerate(train_array):
#         if np.array_equal(test_array[i], train_array[j]):
#             train_array = np.delete(train_array, j)

# train_dataset = train_array
ok = 0
for i, test in enumerate(test_dataset):
    test_arr = np.array(test, dtype=float)
    for j, train in enumerate(train_dataset):
        train_arr = np.array(train, dtype=float)
        if np.array_equal(test_arr, train_arr):
            del train_dataset[j]

# for i, test in enumerate(test_dataset):
#     for j, train in enumerate(train_dataset):
#         if test == train:
#             del train_dataset[j]

# for i, test in enumerate(test_dataset):
#     for j, train in enumerate(train_dataset):
#         if test == train:
#             del train_dataset[j]

# for i, test in enumerate(test_dataset):
#     for j, train in enumerate(train_dataset):
#         count = 0
#         for k, elem in enumerate(test):
#             if elem == train[k]:
#                 count += 1
#         if count == len(test):
#             print('i: %s, j: %s' % (i, j))
           


print("length raw {}".format(len(train_dataset)))
print("length testset {}".format(len(test_dataset)))

# if not os.path.exists(training_csv_output):
with open(training_csv_output, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerows(headers)
    write.writerows(train_dataset)
# if not os.path.exists(testing_csv_output):
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
           