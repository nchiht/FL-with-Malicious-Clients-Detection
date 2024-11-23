from flwr.common import ndarrays_to_parameters

from flwr.server.strategy import FedAvg
from clients import get_parameters, Net


strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    initial_parameters=ndarrays_to_parameters(get_parameters(Net()))
)