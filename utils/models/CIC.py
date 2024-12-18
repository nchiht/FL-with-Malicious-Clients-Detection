import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder

from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from logging import INFO, DEBUG, ERROR, WARNING

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
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
# class mnist_Net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size = x.size(0)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(batch_size, -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # Lấy tuple (image, label) từ ImageFolder
        return {"data": image, "label": label}  # Trả về dạng dictionary
        # return self.dataset[idx]    
    
def load_data_CIC(data_path: str = "data/CICMaldroid_IMG"):
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    dataset = CustomDataset(root_dir=data_path, transform=transform)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(2024))

    return train_dataset, test_dataset
def load_dataset_CIC(num_partitions: int, batch_size: int, val_ratio: float = 0.2):
    trainset, testset = load_data_CIC()

    num_images_train = len(trainset) // num_partitions
    partition_len_train = [num_images_train] * num_partitions
    remainder_train = len(trainset) % num_partitions
    partition_len_train[-1] += remainder_train

    # num_images_val = len(valset) // num_partitions   
    # partition_len_val = [num_images_val] * num_partitions
    # remainder_val = len(valset) % num_partitions
    # partition_len_val[-1] += remainder_val

    trainsets= random_split(trainset, partition_len_train, generator=torch.Generator().manual_seed(2025))
    # valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(2025))
    trainloaders = []
    valloaders = []
    for trainset in trainsets:
        num_total = len(trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2026))

        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(valset, batch_size=batch_size, shuffle=True))
    # for valset in valsets:
    #     valloaders.append(DataLoader(valset, batch_size=batch_size, shuffle=True))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    print("Data CIC loaded")
    return trainloaders, valloaders, testloader

# def load_datasets(partition_id: int, NUM_CLIENTS: int, BATCH_SIZE: int):
#     fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})

#     for _ in range(2):  # Try twice
#         try:
#             partition = fds.load_partition(partition_id)
#             break
#         except Exception as e:
#             log(WARNING, f"Error loading partition: {e}. Retrying...")
#     else:
#         raise RuntimeError("Failed to load partition after 2 attempts")
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] # Standard MNIST mean and std
#     )

#     def apply_transforms(batch):
#         # Instead of passing transforms to CIFAR10(..., transform=transform)
#         # we will use this function to dataset.with_transform(apply_transforms)
#         # The transforms object is exactly the same
#         batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
#         return batch

#     # Create train/val for each partition and wrap it into DataLoader
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
#     valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
#     testset = fds.load_split("test").with_transform(apply_transforms)
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE)
#     return trainloader, valloader, testloader


