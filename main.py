import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import os
import torch.nn.functional as F
from net import Net
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from prettytable import PrettyTable
from buildTrainData import *
from buildTestData import *

FILE = 'final_5007_521_0.944acc.pth'

curr = os.path.dirname(__file__)
path = os.path.join(curr, r'data/csv-data/Testing_despression_processed.csv')
diseases_path = os.path.join(curr, r'data/csv-data/diseases_disertatie.csv')

train = TrainingData()
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test = TestingData()
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

print(len(train), len(test))
try:
    net = Net()
    net.load_state_dict(torch.load(FILE))
    net.eval()
    # for param in net.parameters():
    # print(param)
except:
    # train = TrainingData()
    # trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    net = Net()

    import torch.optim as optim

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),  lr=0.0001, weight_decay=0.00001)

    for epoch in range(10):  # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            # X is the batch of features, y is the batch of targets.
            X, y = data
            # sets gradients to 0 before loss calc. You will do this likely every step.
            net.zero_grad()
            # pass in the reshaped batch (recall they are 28x28 atm)
            output = net(X)
            # print(X, y, output)
            # calc and grab the loss value
            loss = F.nll_loss(output, y.squeeze().long())
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        # print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!


correct = 0
total = 0
preds = []
targets = []
net.eval()

correct_GERD = 0
total_GERD = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X)
        for idx, i in enumerate(output):
            # print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
                if torch.argmax(i) == torch.tensor(0):
                    correct_GERD += 1
                    total_GERD += 1
            total += 1
            preds.append(torch.argmax(i))
            targets.append(y[idx].squeeze())

print(len(preds), len(targets), correct_GERD, total_GERD)

conf_matrix = confusion_matrix(preds, targets)

t = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
                               'Sensitivity', 'Specificity', 'Precision', 'F1'])
nb_classes = 42

TP = np.diag(conf_matrix)
for c in range(nb_classes):
    idx = torch.ones(nb_classes).bool()
    idx[c] = 0

    # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()

    FP = conf_matrix[idx, c].sum()

    FN = conf_matrix[c, idx].sum()

    sensitivity = (TP[c] / (TP[c]+FN))
    specificity = (TN / (TN+FP))
    precision = (TP[c] / (TP[c] + FP))
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    
    with open(diseases_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, rows in enumerate(csv_reader):
            if c == torch.tensor(int(rows[0])):
                disease_value = rows[1]


    t.add_row([c, disease_value, float(TP[c]), float(TN), float(FP), float(FN), round(float
        (sensitivity), 3), round(float(specificity), 3), round(float(precision), 3), round(float(F1), 3)])

accuracy = round(correct/total, 3)
print("Accuracy: ", accuracy)
print(t)

if accuracy >= 0.90 and not os.path.exists(FILE):
    torch.save(net.state_dict(), FILE)
