import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import os
import torch.nn.functional as F
from net import Net
from buildTrainData import *
from buildTestData import *

FILE = 'final_5007_521_0.944acc.pth'

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
net.eval()
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X)
        for idx, i in enumerate(output):
            # print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

accuracy = round(correct/total, 3)
print("Accuracy: ", accuracy)


if accuracy >= 0.90 and not os.path.exists(FILE):
    torch.save(net.state_dict(), FILE)
