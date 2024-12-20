import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
    
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from logging import INFO, DEBUG, ERROR, WARNING

class mnist_Net(nn.Module):
    def __init__(self) -> None:
        super(mnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2,stride=1)
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

    # def __init__(self) -> None:
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     batch_size = x.size(0)
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(batch_size, -1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)
    
# def load_datasets(partition_id: int):
def load_datasets(partition_id: int, NUM_CLIENTS: int, BATCH_SIZE: int):
    fds = FederatedDataset(dataset="builetrongduc/CICMalDroid", partitioners={"train": NUM_CLIENTS})

    for _ in range(2):  # Try twice
        try:
            partition = fds.load_partition(partition_id)
            break
        except Exception as e:
            log(WARNING, f"Error loading partition: {e}. Retrying...")
    else:
        raise RuntimeError("Failed to load partition after 2 attempts")
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # pytorch_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] # Standard MNIST mean and std
    # )
    pytorch_transforms =transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


