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

trainCSVPath = os.path.join(curr, r'data/csv-data/Training.csv')
testCSVPath = os.path.join(curr, r'data/csv-data/Testing.csv')

training_csv_output = os.path.join(curr, r'data/csv-data/processed_training.csv')
testing_csv_output = os.path.join(curr, r'data/csv-data/processed_testing.csv')

diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')

train = []
test = []
diseases = []


with open(trainCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            train.append(rows)

with open(testCSVPath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            test.append(rows)

traindata = np.loadtxt(training_csv_output, delimiter=',', dtype=np.float32, skiprows=1)
testdata = np.loadtxt(testing_csv_output, delimiter=',', dtype=np.float32, skiprows=1)


for i, test_item in enumerate(testdata):
    for j, train_item in enumerate(traindata):
        comparison = test_item == train_item
        equal_arrays = comparison.all()
        if equal_arrays:
            print("test: %s, train: %s" % (i, j))

# for i, test_item in enumerate(test):
#     for j, train_item in enumerate(train):
#         comparison = test_item == train_item
#         equal_arrays = comparison.all()
#         if test_item == train_item:
#             print("test: %s, train: %s" % (i, j))