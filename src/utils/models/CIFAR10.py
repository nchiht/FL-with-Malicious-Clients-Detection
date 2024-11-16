import torch
import torch.nn as nn
import torch.nn.functional as F 

# Define model
# 64x64 img
class MalwareDet(nn.Module):
    def __init__(self):
        super(MalwareDet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 59 * 59, 500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10_Net(nn.Module):
    def __init__(self) -> None:
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x