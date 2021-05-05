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
from prettytable import PrettyTable

# np.set_printoptions(threshold=np.inf)

nb_classes = 41
FILE = 'model.pth'

curr = os.path.dirname(__file__)

test = TestingData()
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')


try: 
    net = Net()
    net.load_state_dict(torch.load(FILE))
    net.eval()
    # for param in net.parameters():
    #     print(param)
except:
    train = TrainingData()
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    net = Net()

    import torch.optim as optim

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10): # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y.squeeze().long())  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 



correct = 0
total = 0

preds = []
targets = []
diseases = []

with open(diseases_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        diseases.append(rows)



def get_disease_name(index):
    for disease in diseases:
        if float(disease[0]) == index:
            return (disease[1][:30] + '..') if len(disease[1]) > 30 else disease[1]

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X)
        for idx, i in enumerate(output):
            preds.append(torch.argmax(i).long())
            targets.append(y[idx].squeeze().long())
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

accuracy = round(correct/total, 3)
print("Accuracy: ", accuracy)

preds = torch.tensor(preds)
targets = torch.tensor(targets)
conf_matrix = confusion_matrix(preds, targets)
conf_matrix = torch.tensor(conf_matrix)

t = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'Precision', 'F1'])

TP = np.diag(conf_matrix)
for c in range(nb_classes):
    idx = torch.ones(nb_classes).bool()
    idx[c] = 0
    # all non-class samples classified as non-class
    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()

    sensitivity = (TP[c] / (TP[c]+FN))
    specificity = (TN / (TN+FP))
    precision = (TP[c] / (TP[c] + FP))
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    # print('Class {}\nTP {}, TN {}, FP {}, FN {}, Sensitivity: {}, Specificity: {}, Precision: {}, F1: {}'.format(
    #     c, TP[c], TN, FP, FN, sensitivity, specificity, precision, F1))
    
    t.add_row([c, get_disease_name(c), float(TP[c]), float(TN), float(FP), float(FN), float(sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])


print(t)
    

if accuracy > 0.97 and not os.path.exists(FILE):
    torch.save(net.state_dict(), FILE)

