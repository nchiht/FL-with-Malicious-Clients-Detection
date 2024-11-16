import torch
import torch.nn as nn
import torchvision.transforms as transforms

import flwr
# from flwr.client import Client, ClientApp, NumPyClient
# from flwr.common import Metrics, Context
# from flwr.server import ServerApp, ServerConfig, ServerAppComponents
# from flwr.server.strategy import FedAvg
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader


NUM_CLIENTS = 10
BATCH_SIZE = 32

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

def train(net, trainloader, epochs: int, verbose=False, device='cpu'):
    DEVICE=torch.device(device)
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader, device='cpu'):
    DEVICE=torch.device(device)
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def inference(network, test_loader):
    if torch.cuda.is_available():
        network = network.to('cuda:0')
    network.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total_samples_val = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
          if torch.cuda.is_available():
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')
          outputs = network(inputs)
          test_loss += criterion(outputs, labels).item()
          _, predicted_val = torch.max(outputs.data, 1)
          total_samples_val += labels.size(0)
          correct += (predicted_val == labels).sum().item()

    test_loss /= total_samples_val
    accuracy = correct / total_samples_val

    print('\nTest set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_samples_val, 100. * accuracy))

    return accuracy
