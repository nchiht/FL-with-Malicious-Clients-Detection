import os   
os.environ['RAY_DEDUP_LOGS'] = '0'
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import Dataset

from collections import OrderedDict
from typing import List, Tuple, Dict
from datasets.utils.logging import disable_progress_bar

from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar, Parameters
from utils.models import cifar10, mnist
from flwr.common.logger import log
from logging import INFO, DEBUG

"""
Each model associated with clients would have 
different train, test, load_datasets function
"""

# DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
# NUM_CLIENTS = 10
BATCH_SIZE = 32

def train(net, trainloader, epochs: int, verbose=False, device="cpu", learning_rate=0.001):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    local_grad = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # if net._get_name() in ["cifar10_Net", "mnist_Net"]:
        
        for batch in trainloader:
            
            if(net._get_name() == "cifar10_Net"): # TODO: depends on models
                images, labels = batch["img"].to(device), batch["label"].to(device)
            if(net._get_name() == "mnist_Net"):
                images, labels = batch["image"].to(device), batch["label"].to(device)
            if(net._get_name() == "MyModel"):
                images, labels = batch["data"].to(device), batch["label"].to(device)
                log(INFO, "begin training: %s", images.shape)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Local_Grad
            if len(local_grad) == 0:
                for p in net.parameters():
                    local_grad.append(p.grad.clone())
            else:
                for idx, p in enumerate(net.parameters()):
                    local_grad[idx] += p.grad

            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total    
        # elif net._get_name() == "MyModel":
        #     for inputs, labels in trainloader:
        #         inputs, labels = inputs.to(device), labels.to(device)

        #         optimizer.zero_grad()
        #         outputs = net(inputs)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()
                
        #         # Metrics
        #         epoch_loss += loss
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        #     # Training accuracy
        #     epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    # log(INFO, "Local Grad: %s", local_grad)
    log(INFO, "End training")
    return local_grad

def test(net, testloader, device="cpu"):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        # if net._get_name() in ["cifar10_Net", "mnist_Net"]:
        for batch in testloader:

            if(net._get_name() == "cifar10_Net"): # TODO: depends on models
                images, labels = batch["img"].to(device), batch["label"].to(device)
            if(net._get_name() == "mnist_Net"):
                images, labels = batch["image"].to(device), batch["label"].to(device)
            if(net._get_name() == "MyModel"):
                images, labels = batch["data"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # elif net._get_name() == "MyModel":
        #     for inputs, labels in testloader:
        #         inputs, labels = inputs.to(device), labels.to(device)

        #         outputs = net(inputs)
        #         loss += criterion(outputs, labels).item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    print("Setted parameters")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

#Posion data
def poison_data(train_loader, poison_ratio=0, net=cifar10.cifar10_Net()):
    """
    Thay đổi nhãn dữ liệu trong train_loader bằng cách thực hiện data flipping.
    Args:
        train_loader: Dataloader chứa dữ liệu huấn luyện.
        poison_ratio: Tỉ lệ nhãn bị thay đổi (mặc định 0%).
    Returns:
        poisoned_dataset: Một đối tượng datasets.Dataset chứa dữ liệu đã bị thay đổi nhãn.
    """
    
    if(net._get_name() == "cifar10_Net"): # TODO: depends on models
        poisoned_data = {"img": [], "label": []}
        for batch in train_loader:
            inputs, labels = batch["img"], batch["label"]

            for i in range(len(labels)):
                if torch.rand(1).item() < poison_ratio:
                    labels[i] = 0 #(labels[i] + 1) % 10

            # Chuyển từng phần tử của inputs thành torch.Tensor nếu chưa phải
            inputs = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in inputs]

            poisoned_data["img"].extend(inputs)
            poisoned_data["label"].extend(labels)
        
        # Chuyển đổi dữ liệu thành torch.Tensor
        poisoned_data["img"] = [torch.tensor(img) if isinstance(img, list) else img for img in poisoned_data["img"]]
        poisoned_data["label"] = torch.tensor(poisoned_data["label"])
        
        # Tạo datasets.Dataset
        poisoned_dataset = Dataset.from_dict({"img": poisoned_data["img"], "label": poisoned_data["label"]})
        return poisoned_dataset
    if(net._get_name() == "mnist_Net"):
        poisoned_data = {"image": [], "label": []}
        for batch in train_loader:
            inputs, labels = batch["image"], batch["label"]

            for i in range(len(labels)):
                if torch.rand(1).item() < poison_ratio:
                    labels[i] = 0 #(labels[i] + 1) % 10

            # Chuyển từng phần tử của inputs thành torch.Tensor nếu chưa phải
            inputs = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in inputs]

            poisoned_data["image"].extend(inputs)
            poisoned_data["label"].extend(labels)
        
        # Chuyển đổi dữ liệu thành torch.Tensor
        poisoned_data["image"] = [torch.tensor(img) if isinstance(img, list) else img for img in poisoned_data["image"]]
        poisoned_data["label"] = torch.tensor(poisoned_data["label"])
        
        # Tạo datasets.Dataset
        poisoned_dataset = Dataset.from_dict({"image": poisoned_data["image"], "label": poisoned_data["label"]})
        return poisoned_dataset

def poison_data_utg(train_loader, poison_ratio=0, net=cifar10.cifar10_Net()):
    """
    Thay đổi nhãn dữ liệu trong train_loader bằng cách thực hiện data flipping.
    Args:
        train_loader: Dataloader chứa dữ liệu huấn luyện.
        poison_ratio: Tỉ lệ nhãn bị thay đổi (mặc định 0%).
    Returns:
        poisoned_dataset: Một đối tượng datasets.Dataset chứa dữ liệu đã bị thay đổi nhãn.
    """
    
    if(net._get_name() == "cifar10_Net"): # TODO: depends on models
        poisoned_data = {"img": [], "label": []}
        for batch in train_loader:
            inputs, labels = batch["img"], batch["label"]


            for i in range(len(labels)):
                if torch.rand(1).item() < poison_ratio:
                    labels[i] = np.random.randint(0, 10)
                    # labels[i] = (labels[i] + 1) % 10

            # Chuyển từng phần tử của inputs thành torch.Tensor nếu chưa phải
            inputs = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in inputs]

            poisoned_data["img"].extend(inputs)
            poisoned_data["label"].extend(labels)
        
        # Chuyển đổi dữ liệu thành torch.Tensor
        poisoned_data["img"] = [torch.tensor(img) if isinstance(img, list) else img for img in poisoned_data["img"]]
        poisoned_data["label"] = torch.tensor(poisoned_data["label"])
        
        # Tạo datasets.Dataset
        poisoned_dataset = Dataset.from_dict({"img": poisoned_data["img"], "label": poisoned_data["label"]})
        return poisoned_dataset
    if(net._get_name() == "mnist_Net"):
        poisoned_data = {"image": [], "label": []}
        for batch in train_loader:
            inputs, labels = batch["image"], batch["label"]

            for i in range(len(labels)):
                if torch.rand(1).item() < poison_ratio:
                    labels[i] = np.random.randint(0, 10)
                    # labels[i] = (labels[i] + 1) % 10

            # Chuyển từng phần tử của inputs thành torch.Tensor nếu chưa phải
            inputs = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in inputs]

            poisoned_data["image"].extend(inputs)
            poisoned_data["label"].extend(labels)
        
        # Chuyển đổi dữ liệu thành torch.Tensor
        poisoned_data["image"] = [torch.tensor(img) if isinstance(img, list) else img for img in poisoned_data["image"]]
        poisoned_data["label"] = torch.tensor(poisoned_data["label"])
        
        # Tạo datasets.Dataset
        poisoned_dataset = Dataset.from_dict({"image": poisoned_data["image"], "label": poisoned_data["label"]})
        return poisoned_dataset

def collate_fn(batch):
    """
    Chuẩn hóa batch để tạo Tensor từ dữ liệu.
    """
    images = torch.stack([torch.tensor(item["img"]) if isinstance(item["img"], list) else item["img"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"img": images, "label": labels}

def collate_fn_mnist(batch):
    """
    Chuẩn hóa batch để tạo Tensor từ dữ liệu.
    """
    images = torch.stack([torch.tensor(item["image"]) if isinstance(item["image"], list) else item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"image": images, "label": labels}

class FlowerClient(NumPyClient):
    def __init__(
            self,
            parition_id, 
            node_id, 
            net, 
            trainloader, 
            valloader, 
            device='cpu',
            epochs=1, 
            datapoison_ratio=0,
            target=True
        ):
        self.partition_id = parition_id 
        self.node_id = node_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device(device)
        self.epochs = epochs
        self.datapoison_ratio = datapoison_ratio
        self.target = target

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        log(INFO, "1: Client %s: Normal data, data: %s", self.node_id, self.net._get_name())
        set_parameters(self.net, parameters)
        self.partition_id = config["index"]
        # log(INFO, "dp flag client %s: %s", self.partition_id, config["dp_flags"])
        if (
            (self.datapoison_ratio > 0) and 
            # (self.partition_id in [0,1,2,3,4]) and 
            (config["dp_flags"]) and 
            (config["server_round"] > config["warmup_rounds"])
        ): #TODO: thêm client_states và server_round từ server
            if self.target:
                poisoned_data = poison_data(self.trainloader, poison_ratio=self.datapoison_ratio, net=self.net)
                print("Target")
            else:
                poisoned_data = poison_data_utg(self.trainloader, poison_ratio=self.datapoison_ratio, net=self.net)
                print("UTG")
            # Tạo lại DataLoader với dữ liệu đã bị nhiễm độc
            if(self.net._get_name() == "cifar10_Net"): # TODO: depends on models
                trainloader = DataLoader(poisoned_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)#shuffle=True, 
            if(self.net._get_name() == "mnist_Net"):
                trainloader = DataLoader(poisoned_data, batch_size=BATCH_SIZE, collate_fn=collate_fn_mnist)#shuffle=True, 
            log(INFO, "Client %s: Poisoned data", self.partition_id) 
        else:
            trainloader = self.trainloader
            log(INFO, "loaded trainloader")
        log(INFO, "2: Client %s: Normal data, data: %s", self.node_id, self.net._get_name())
        # print("learning_rate", config["learning_rate"])
        train(self.net, trainloader, epochs=self.epochs, device=self.device, learning_rate=config["learning_rate"])
        log(INFO, "3: Client %s: Normal data, data: %s", self.node_id, self.net._get_name())
        return get_parameters(self.net), len(self.trainloader), {"node_id": self.node_id, "partition_id": self.partition_id}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


