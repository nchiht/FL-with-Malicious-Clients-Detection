import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List, Tuple
from datasets.utils.logging import disable_progress_bar

from flwr.client import Client, NumPyClient


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
NUM_CLIENTS = 10
BATCH_SIZE = 32
    
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:

            if(net._get_name() == "cifar10_Net"): # TODO: depends on models
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            if(net._get_name() == "mnist_Net"):
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

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

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:

            if(net._get_name() == "cifar10_Net"): # TODO: depends on models
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            if(net._get_name() == "mnist_Net"):
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
# def client_fn(context: Context) -> Client:
#     """Create a Flower client representing a single organization."""

#     # Load model
#     net = Net().to(DEVICE)

#     # Load data (CIFAR-10)
#     # Note: each client gets a different trainloader/valloader, so each client
#     # will train and evaluate on their own unique data partition
#     # Read the node_config to fetch data partition associated to this node
#     partition_id = context.node_config["partition-id"]
#     trainloader, valloader, _ = load_datasets_cifar10(partition_id=partition_id)

#     # Create a single Flower client representing a single organization
#     # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
#     # to convert it to a subclass of `flwr.client.Client`
#     return FlowerClient(net, trainloader, valloader).to_client()


# def main():
#     trainloader, valloader, testloader = load_datasets(partition_id=0)
#     net = Net().to(DEVICE)

#     for epoch in range(5):
#         train(net, trainloader, 1)
#         loss, accuracy = test(net, valloader)
#         print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

#     loss, accuracy = test(net, testloader)
#     print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

# if __name__ == '__main__':
#     # main()
#     # Run simulation
#     run_simulation(
#         server_app=server,
#         client_app=client,
#         num_supernodes=NUM_CLIENTS,
#         backend_config=backend_config,
#     )