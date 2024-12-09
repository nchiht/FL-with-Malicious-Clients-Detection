import os   
os.environ['RAY_DEDUP_LOGS'] = '0'
import flwr
from flwr.simulation import run_simulation, start_simulation
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg, Krum
from flwr.server import ServerConfig, ServerAppComponents
from flwr.client import ClientApp, Client, NumPyClient
from flwr.common.logger import log
from flwr.common import Metrics, Context, NDArrays, Scalar
from flwr.common import ndarrays_to_parameters, Context

from server import EnhancedServer
from clients import FlowerClient
from clients import get_parameters, set_parameters, test
from utils import evaluation
from utils.models import cifar10, mnist
from attacks import no_attack, gaussian_attack 
from strategy import EnhancedStrategy

import torch
import argparse
from torch.utils.data import DataLoader
from datasets.utils.logging import disable_progress_bar
from logging import DEBUG, INFO
from typing import Dict, Optional, Tuple
import random


def main():
    pass


def evaluate_fn(
        server_round: int, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

    model = model_with_dataset[dataset_id][0]
    set_parameters(model, parameters)
    model.to(device)

    _,_, testset = model_with_dataset[dataset_id][1](partition_id=random.randint(1, 5), NUM_CLIENTS=NUM_CLIENTS, BATCH_SIZE=BATCH_SIZE)
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    loss, accuracy = test(model, testset, device=device)
    log(INFO, f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    # return statistics
    return loss, {"accuracy": accuracy}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(
        # strategy=strategy, 
        config=config, 
        server=EnhancedServer(
            strategy=strategy, 
            attack_fn=attck_fnc[attack], 
            magnitude=magnitude, 
            num_malicious=num_model_poisoning, 
            num_data_poisoning=num_data_poisoning, 
            window_size=window_size,
            warmup_rounds=warmup_rounds,
            steps=2
        )
    )


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = model_with_dataset[dataset_id][0].to(device)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = model_with_dataset[dataset_id][1](partition_id=partition_id, NUM_CLIENTS=NUM_CLIENTS, BATCH_SIZE=BATCH_SIZE)
    node_id = context.node_id
    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(
        partition_id, 
        node_id, net, trainloader, valloader, 
        device=device,
        epochs=epochs, 
        datapoison_ratio=datapoison_ratio, 
        target=target
    ).to_client()


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="cifar10", type=str, help="Indicate dataset for the simulation")
    parser.add_argument("-n", "--num_clients", default=10, type=int, help="Indicate number of clients for the simulation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="Indicate batch size for data partition")
    # Server configuration
    parser.add_argument("-r", "--rounds", default=10, type=int, help="Indicate number of rounds for the simulation")
    parser.add_argument("-a", "--attack", default="gaussian_attack", type=str, help="Indicate attack type for the simulation")
    parser.add_argument("-mag", "--magnitude", default=1, type=float, help="Indicate magnitude of the attack")
    parser.add_argument("-nm", "--num_model_poisoning", default=3, type=int, help="Indicate number of model poisoning")
    parser.add_argument("-nd", "--num_data_poisoning", default=2, type=int, help="Indicate number of data poisoning")
    parser.add_argument("-wr", "--warmup_rounds", default=1, type=int, help="Indicate number of warmup rounds")
    # Defense configuration
    parser.add_argument("--window_size", default=5, type=int, help="Indicate window size for the attack")
    # Client configuration
    parser.add_argument("-e", "--epochs", default=10, type=int, help="Indicate number of epochs for training") 
    parser.add_argument("-rat", "--datapoison_ratio", default=0.5, type=float, help="Indicate ratio of data poisoning")
    parser.add_argument("-t", "--target", default=True, type=bool, help="Indicate target or untargeted attack")
    
    parser.add_argument("--device", default="cpu", type=str, help="Select device type for the process")
    args = parser.parse_args()
    run_config = vars(args)
    
    dataset_id = run_config["dataset"]
    NUM_CLIENTS = run_config["num_clients"]
    BATCH_SIZE = run_config["batch_size"]
    NUM_ROUNDS = run_config["rounds"]
    attack = run_config["attack"]
    magnitude = run_config["magnitude"]
    num_model_poisoning = run_config["num_model_poisoning"]
    num_data_poisoning = run_config["num_data_poisoning"]
    warmup_rounds = run_config["warmup_rounds"]
    window_size = run_config["window_size"]
    epochs = run_config["epochs"]
    datapoison_ratio = run_config["datapoison_ratio"]
    target = run_config["target"]
    device = torch.device(run_config["device"])

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    attck_fnc = {
        'gaussian_attack': gaussian_attack,
        'no_attack': no_attack
    }
    # When running on GPU, assign an entire GPU for each client
    if device.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

    print(f"Training on {device}")
    print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
    disable_progress_bar()

    model_with_dataset = {
        "cifar10": [cifar10.cifar10_Net(), cifar10.load_datasets],
        "mnist": [mnist.mnist_Net(), mnist.load_datasets]
        # TODO: add mnist
    }

    # strategy = FedAvg(
    #     fraction_fit=1.0,  # Sample 100% of available clients for training
    #     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    #     min_fit_clients=10,  # Never sample less than 10 clients for training
    #     min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    #     min_available_clients=10,  # Wait until all 10 clients are available
    #     initial_parameters=ndarrays_to_parameters(get_parameters(model_with_dataset[dataset_id][0])),
    #     evaluate_fn=evaluate_fn
    # )

    # strategy = Krum(
    #     fraction_fit = 1,
    #     fraction_evaluate = 0.5,
    #     min_fit_clients = 10,
    #     min_evaluate_clients = 5,
    #     min_available_clients = 10,
    #     num_malicious_clients = 5,
    #     num_clients_to_keep = 20,
    #     evaluate_fn = evaluate_fn,
    #     initial_parameters = ndarrays_to_parameters(get_parameters(model_with_dataset[dataset_id][0])),
    # )

    strategy = EnhancedStrategy(
        fraction_fit = 1,
        fraction_evaluate = 0.5,
        min_fit_clients = 10,
        min_evaluate_clients = 5,
        min_available_clients = 10,
        num_malicious_clients = 10,
        num_clients_to_keep = 30,
        evaluate_fn = evaluate_fn,
        initial_parameters = ndarrays_to_parameters(get_parameters(model_with_dataset[dataset_id][0]))
    )
    

    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)


    log(INFO, f"Federated learning for {dataset_id}")
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        verbose_logging=True
    )
    # history = start_simulation(
    #     client_fn=client_fn,
    #     num_clients=NUM_CLIENTS,
    #     # client_resources=cfg.client_resources, # TODO: Add again when we manually set the resources
    #     #ray_init_args={"num_cpus": 1, "num_gpus": 16},
    #     server=EnhancedServer(strategy=strategy),
    #     # config=flwr.server.ServerConfig(num_rounds=cfg.server.num_rounds),
    #     strategy=strategy,
    # )