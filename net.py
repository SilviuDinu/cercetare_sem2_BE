import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(139, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(64, 42)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)